"""Batch export orchestrator — Adobe / Shutterstock / both, IPTC + FTP.

Single entry point :func:`run_export_batch` that chains:

    paths → expert reports → [optional IPTC write] → CSV → [optional FTP]

Designed to be cheap on resources by default (no AI), so it runs on
low-power machines. Every side-effect (IPTC write, FTP push) is opt-in
and reported separately in the result so the caller (UI) can render
per-step success badges.

The function never raises on per-file errors; it accumulates them
into :class:`BatchExportResult.errors` and continues, matching the
project-wide "let the reviewer be the final gate" posture.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..analysis.expert_report import (
    build_expert_report,
    enrich_with_ai_result,
)
from ..models.metadata_models import ExpertMetadataReport, IPTCFields
from .csv_exporter import (
    ExportResult,
    export_double_csv,
    write_adobe_csv,
    write_shutterstock_csv,
)
from .ftp_uploader import FtpConfig, UploadResult, upload_files

logger = logging.getLogger(__name__)


class Platform(str, Enum):
    """Which contributor portal the export targets."""

    ADOBE = "adobe"
    SHUTTERSTOCK = "shutterstock"
    BOTH = "both"


class FileStatus(str, Enum):
    """Per-file lifecycle, surfaced live to the UI via the callback."""

    PENDING = "pending"
    ANALYZING = "analyzing"
    WRITING_IPTC = "writing_iptc"
    DONE = "done"
    FAILED = "failed"


@dataclass
class FileProgress:
    """Lifecycle of one file inside the batch."""

    path: Path
    status: FileStatus = FileStatus.PENDING
    report: Optional[ExpertMetadataReport] = None
    iptc_written: bool = False
    error: Optional[str] = None


@dataclass
class BatchExportResult:
    """End-to-end batch outcome — what was analyzed, written, uploaded."""

    platform: Platform
    output_dir: Path
    file_progress: List[FileProgress] = field(default_factory=list)
    adobe_csv: Optional[Path] = None
    shutterstock_csv: Optional[Path] = None
    iptc_written_count: int = 0
    ftp_result: Optional[UploadResult] = None
    duration_s: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def reports(self) -> List[ExpertMetadataReport]:
        return [fp.report for fp in self.file_progress if fp.report is not None]

    @property
    def success_count(self) -> int:
        return sum(1 for fp in self.file_progress if fp.status == FileStatus.DONE)

    @property
    def failure_count(self) -> int:
        return sum(1 for fp in self.file_progress if fp.status == FileStatus.FAILED)

    @property
    def csv_paths(self) -> List[Path]:
        return [p for p in (self.adobe_csv, self.shutterstock_csv) if p is not None]


ProgressCallback = Callable[[FileProgress], None]


def run_export_batch(
    paths: Sequence[Path],
    output_dir: Path,
    *,
    platform: Platform = Platform.BOTH,
    write_iptc: bool = False,
    use_ai: bool = False,
    basename: str = "metadata",
    iptc_writer: Optional[Any] = None,  # Has .write_iptc(path, IPTCFields)
    ai_runner: Optional[Callable[[Path], Dict[str, Any]]] = None,
    metadata_reader: Optional[Any] = None,  # Has .read(path) returning ImageMetadata
    ftp_config: Optional[FtpConfig] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> BatchExportResult:
    """Run the full pipeline on *paths*.

    Args:
        paths: Image files to process.
        output_dir: Where the CSV(s) go.
        platform: Which CSV to produce — Adobe, Shutterstock, or both.
        write_iptc: If True, also writes IPTC metadata back into each
            file (requires *iptc_writer*).
        use_ai: If True AND *ai_runner* is provided, overlay an AI
            analysis on top of the heuristic baseline.
        basename: Prefix for the CSV filenames.
        iptc_writer: Object with ``write_iptc(path, IPTCFields)`` —
            usually ``MetadataWriter`` from the engines layer. If
            None and ``write_iptc=True``, IPTC step is skipped and a
            warning is logged.
        ai_runner: Callable ``path -> dict`` returning an AI result.
            Only consulted when ``use_ai=True``.
        metadata_reader: Object with ``read(path)`` returning
            ``ImageMetadata``. Used to seed the heuristic builder
            with existing IPTC. If None, builder runs from filename
            heuristics only.
        ftp_config: If set, uploads the produced CSVs to the FTP
            after a successful export.
        on_progress: Per-file lifecycle callback. Called at every
            status transition so the UI can render the live table.

    Returns:
        :class:`BatchExportResult` — never raises on per-file errors.
    """
    paths_list = [Path(p) for p in paths]
    result = BatchExportResult(
        platform=platform,
        output_dir=Path(output_dir),
    )
    if not paths_list:
        result.errors.append("Aucun fichier sélectionné pour l'export.")
        return result

    t0 = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-populate the progress list so the UI can show all rows
    # immediately, even before any per-file work has started.
    for p in paths_list:
        fp = FileProgress(path=p)
        result.file_progress.append(fp)
        _emit(on_progress, fp)

    for fp in result.file_progress:
        _process_one(
            fp,
            use_ai=use_ai,
            write_iptc=write_iptc,
            iptc_writer=iptc_writer,
            ai_runner=ai_runner,
            metadata_reader=metadata_reader,
            on_progress=on_progress,
        )
        if fp.iptc_written:
            result.iptc_written_count += 1
        if fp.error:
            result.errors.append(f"{fp.path.name}: {fp.error}")

    # -------- CSV --------
    reports = result.reports
    if not reports:
        result.errors.append("Aucun rapport produit — CSV non écrit.")
        result.duration_s = time.perf_counter() - t0
        return result

    try:
        if platform == Platform.BOTH:
            csv_result: ExportResult = export_double_csv(reports, output_dir, basename=basename)
            result.adobe_csv = csv_result.adobe_csv
            result.shutterstock_csv = csv_result.shutterstock_csv
        elif platform == Platform.ADOBE:
            result.adobe_csv = output_dir / f"{basename}_adobe.csv"
            write_adobe_csv(reports, result.adobe_csv)
        elif platform == Platform.SHUTTERSTOCK:
            result.shutterstock_csv = output_dir / f"{basename}_shutterstock.csv"
            write_shutterstock_csv(reports, result.shutterstock_csv)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"Écriture CSV échouée : {exc}")
        logger.exception("CSV write failed")
        result.duration_s = time.perf_counter() - t0
        return result

    # -------- FTP --------
    if ftp_config is not None and result.csv_paths:
        try:
            result.ftp_result = upload_files(result.csv_paths, ftp_config)
            if result.ftp_result.error:
                result.errors.append(f"FTP : {result.ftp_result.error}")
            elif result.ftp_result.failure_count:
                result.errors.append(
                    f"FTP : {result.ftp_result.failure_count} fichier(s) en échec sur "
                    f"{len(result.ftp_result.items)}"
                )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"FTP : {exc}")
            logger.exception("FTP upload raised")

    result.duration_s = time.perf_counter() - t0
    logger.info(
        "Batch export done: %d ok / %d failed / %.2fs",
        result.success_count, result.failure_count, result.duration_s,
    )
    return result


# ============================================================================
# Per-file pipeline
# ============================================================================


def _process_one(
    fp: FileProgress,
    *,
    use_ai: bool,
    write_iptc: bool,
    iptc_writer: Optional[Any],
    ai_runner: Optional[Callable[[Path], Dict[str, Any]]],
    metadata_reader: Optional[Any],
    on_progress: Optional[ProgressCallback],
) -> None:
    """Build the report, optionally enrich + write IPTC, update FP."""
    fp.status = FileStatus.ANALYZING
    _emit(on_progress, fp)

    try:
        # 1. Existing IPTC seeds the heuristic builder.
        iptc = _read_iptc(fp.path, metadata_reader)

        # 2. Heuristic baseline (no AI).
        report = build_expert_report(fp.path, iptc=iptc)

        # 3. Optional AI enrichment.
        if use_ai and ai_runner is not None:
            try:
                ai_result = ai_runner(fp.path)
                if ai_result:
                    enrich_with_ai_result(report, ai_result)
            except Exception as exc:  # noqa: BLE001
                # Soft-fail on AI errors — heuristic baseline is enough.
                logger.warning("AI step failed on %s: %s", fp.path.name, exc)

        fp.report = report

        # 4. Optional IPTC write-back.
        if write_iptc and iptc_writer is not None:
            fp.status = FileStatus.WRITING_IPTC
            _emit(on_progress, fp)
            _write_iptc(fp, iptc_writer)
        elif write_iptc and iptc_writer is None:
            logger.warning("write_iptc=True but no iptc_writer — skipping write")

        fp.status = FileStatus.DONE
        _emit(on_progress, fp)
    except Exception as exc:  # noqa: BLE001
        fp.error = str(exc)
        fp.status = FileStatus.FAILED
        logger.exception("Per-file pipeline failed for %s", fp.path)
        _emit(on_progress, fp)


def _read_iptc(path: Path, reader: Optional[Any]) -> IPTCFields:
    """Best-effort IPTC read. Returns empty fields when reader is None
    or anything fails — the builder degrades gracefully."""
    if reader is None:
        return IPTCFields()
    try:
        md = reader.read(path)
        return md.iptc if md and md.iptc else IPTCFields()
    except Exception as exc:  # noqa: BLE001
        logger.debug("IPTC read failed on %s: %s", path.name, exc)
        return IPTCFields()


def _write_iptc(fp: FileProgress, writer: Any) -> None:
    """Project the expert report's keywords/title/description into
    IPTC and write back. Uses headline + object_name + caption +
    keywords + supplemental_categories."""
    if fp.report is None:
        return
    report = fp.report
    iptc = IPTCFields(
        headline=report.title_shutterstock or report.title_adobe or None,
        object_name=(report.title_shutterstock or report.title_adobe or "")[:64] or None,
        caption=report.description or None,
        keywords=list(report.keywords),
        supplemental_categories=list(report.categories_shutterstock),
    )
    writer.write_iptc(fp.path, iptc)
    fp.iptc_written = True


def _emit(callback: Optional[ProgressCallback], fp: FileProgress) -> None:
    if callback is None:
        return
    try:
        callback(fp)
    except Exception:  # noqa: BLE001
        logger.exception("on_progress callback raised")
