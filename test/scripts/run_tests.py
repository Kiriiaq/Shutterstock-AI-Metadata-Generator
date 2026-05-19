"""Run the expert-report pipeline on every test/inputs/* image.

Drops the produced CSVs into test/outputs_reels/. Bypasses the
``ShutterstockAIv2`` facade so we don't need ExifTool / Ollama for
the regression run — we go straight to the analysis + export
modules.

Usage::

    python test/scripts/run_tests.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Make the repo root importable when launched from anywhere.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules.analysis.expert_report import build_expert_report  # noqa: E402
from src.modules.engines.metadata_reader import (  # noqa: E402
    ExifToolNotFoundError,
    MetadataReader,
)
from src.modules.export.csv_exporter import export_double_csv  # noqa: E402
from src.modules.models.metadata_models import IPTCFields  # noqa: E402

INPUTS = ROOT / "test" / "inputs"
OUT = ROOT / "test" / "outputs_reels"
SUMMARY_JSON = OUT / "_summary.json"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _read_iptc(path: Path, reader: MetadataReader | None) -> IPTCFields:
    """Read existing IPTC if ExifTool is available, else return empty."""
    if reader is None:
        return IPTCFields()
    try:
        md = reader.read(path)
        return md.iptc if md else IPTCFields()
    except Exception as exc:  # noqa: BLE001
        logger.debug("IPTC read failed on %s: %s", path.name, exc)
        return IPTCFields()


def _collect_inputs() -> list[Path]:
    """Every image under inputs/, including the sub-folder with accents."""
    if not INPUTS.exists():
        return []
    found = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        found.extend(INPUTS.rglob(ext))
    # Exclude 0-byte and corrupted shells from the pipeline — the
    # rejection IS the test, not a failure of run_tests.
    return sorted(found)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    try:
        reader = MetadataReader()
    except ExifToolNotFoundError:
        logger.warning("ExifTool absent — IPTC inputs ignored")
        reader = None

    files = _collect_inputs()
    if not files:
        logger.error("No inputs found under %s — run _make_inputs.py first", INPUTS)
        return 1

    logger.info("Found %d inputs", len(files))
    reports = []
    summary = {"total": len(files), "items": []}
    start = time.perf_counter()

    for path in files:
        item: dict = {"file": str(path.relative_to(ROOT))}
        try:
            iptc = _read_iptc(path, reader)
            report = build_expert_report(path, iptc=iptc)
            reports.append(report)
            item.update({
                "status": "ok",
                "source": report.source,
                "scores": report.scores.to_dict(),
                "keywords_count": len(report.keywords),
                "rejection_risks": len(report.rejection_risks),
                "adobe_warnings": len(report.adobe_warnings),
                "shutterstock_warnings": len(report.shutterstock_warnings),
            })
        except Exception as exc:  # noqa: BLE001
            logger.exception("FAIL %s", path.name)
            item.update({"status": "error", "error": str(exc)})
        summary["items"].append(item)

    if reports:
        result = export_double_csv(reports, OUT, basename="reels")
        summary["adobe_csv"] = str(result.adobe_csv.relative_to(ROOT))
        summary["shutterstock_csv"] = str(result.shutterstock_csv.relative_to(ROOT))
        summary["row_count"] = result.row_count
    else:
        summary["adobe_csv"] = None
        summary["shutterstock_csv"] = None
        summary["row_count"] = 0

    summary["duration_s"] = round(time.perf_counter() - start, 3)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    ok = sum(1 for it in summary["items"] if it["status"] == "ok")
    logger.info("--- Done in %ss ---", summary["duration_s"])
    logger.info("Reports built : %d / %d", ok, len(files))
    logger.info("CSV Adobe     : %s", summary["adobe_csv"])
    logger.info("CSV SH        : %s", summary["shutterstock_csv"])
    logger.info("Summary       : %s", SUMMARY_JSON.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
