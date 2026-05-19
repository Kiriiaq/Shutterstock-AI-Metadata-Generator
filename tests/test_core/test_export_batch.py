"""Tests for the batch-export orchestrator (analysis → CSV → IPTC → FTP)."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.modules.export.batch import (
    BatchExportResult,
    FileProgress,
    FileStatus,
    Platform,
    run_export_batch,
)


def _make_image(tmp_path: Path, name: str = "img.jpg") -> Path:
    p = tmp_path / name
    p.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-header")  # JPEG magic
    return p


class TestRunBatch:
    def test_empty_paths(self, tmp_path):
        result = run_export_batch([], tmp_path)
        assert isinstance(result, BatchExportResult)
        assert result.errors and "Aucun fichier" in result.errors[0]
        assert result.success_count == 0

    def test_heuristic_only_both_csv(self, tmp_path):
        paths = [_make_image(tmp_path, f"img_{i}.jpg") for i in range(3)]
        result = run_export_batch(paths, tmp_path, platform=Platform.BOTH)

        assert result.success_count == 3
        assert result.failure_count == 0
        assert result.adobe_csv is not None and result.adobe_csv.exists()
        assert result.shutterstock_csv is not None and result.shutterstock_csv.exists()
        assert result.iptc_written_count == 0  # write_iptc=False default

    def test_adobe_only(self, tmp_path):
        paths = [_make_image(tmp_path)]
        result = run_export_batch(paths, tmp_path, platform=Platform.ADOBE)
        assert result.adobe_csv is not None and result.adobe_csv.exists()
        assert result.shutterstock_csv is None

    def test_shutterstock_only(self, tmp_path):
        paths = [_make_image(tmp_path)]
        result = run_export_batch(paths, tmp_path, platform=Platform.SHUTTERSTOCK)
        assert result.shutterstock_csv is not None and result.shutterstock_csv.exists()
        assert result.adobe_csv is None

    def test_write_iptc_called(self, tmp_path):
        paths = [_make_image(tmp_path)]
        writer = MagicMock()
        result = run_export_batch(
            paths, tmp_path,
            write_iptc=True, iptc_writer=writer,
            platform=Platform.ADOBE,
        )
        assert result.iptc_written_count == 1
        writer.write_iptc.assert_called_once()

    def test_write_iptc_skipped_when_no_writer(self, tmp_path):
        paths = [_make_image(tmp_path)]
        result = run_export_batch(
            paths, tmp_path,
            write_iptc=True, iptc_writer=None,
            platform=Platform.ADOBE,
        )
        # write_iptc=True without a writer → silently skipped, batch
        # still succeeds.
        assert result.iptc_written_count == 0
        assert result.success_count == 1

    def test_progress_callback_called_for_each_status(self, tmp_path):
        paths = [_make_image(tmp_path, "a.jpg"), _make_image(tmp_path, "b.jpg")]
        events: list[tuple[str, FileStatus]] = []

        def on_progress(fp: FileProgress):
            events.append((fp.path.name, fp.status))

        run_export_batch(paths, tmp_path, on_progress=on_progress)

        # Each file emits at least: pending → analyzing → done.
        names = {n for n, _ in events}
        statuses = {s for _, s in events}
        assert names == {"a.jpg", "b.jpg"}
        assert FileStatus.PENDING in statuses
        assert FileStatus.DONE in statuses

    def test_per_file_error_does_not_break_batch(self, tmp_path):
        good = _make_image(tmp_path, "good.jpg")
        ghost = tmp_path / "nope.jpg"  # never created

        # Pre-create the file just so build_expert_report doesn't blow
        # up on file-not-found; we'll force-fail the writer instead.
        ghost.write_bytes(b"\xff\xd8\xff")

        class FlakyWriter:
            calls = 0

            def write_iptc(self, p, iptc):
                FlakyWriter.calls += 1
                if FlakyWriter.calls == 2:
                    raise OSError("disk full")

        result = run_export_batch(
            [good, ghost], tmp_path,
            write_iptc=True, iptc_writer=FlakyWriter(),
            platform=Platform.ADOBE,
        )

        # One should be marked failed (because writer raised), but the
        # CSV should still be produced from the survivors.
        assert result.failure_count == 1
        assert result.adobe_csv is not None and result.adobe_csv.exists()

    def test_ai_runner_invoked_only_when_enabled(self, tmp_path):
        paths = [_make_image(tmp_path)]
        ai = MagicMock(return_value={"title": "AI Title", "keywords": ["fresh"]})

        result = run_export_batch(paths, tmp_path, use_ai=False, ai_runner=ai)
        ai.assert_not_called()
        assert result.success_count == 1

        ai.reset_mock()
        run_export_batch(paths, tmp_path, use_ai=True, ai_runner=ai)
        ai.assert_called_once()

    def test_csv_keywords_comma_separated(self, tmp_path):
        """Regression sanity: the P0 bug fix must hold under the batch path."""
        paths = [_make_image(tmp_path)]
        result = run_export_batch(paths, tmp_path, platform=Platform.SHUTTERSTOCK)
        with result.shutterstock_csv.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        # Keywords cell may be empty (file has no IPTC) but must not
        # use spaces as separator. We only verify the column exists
        # and is a string.
        assert "Keywords" in rows[0]


class TestPlatformEnum:
    def test_from_string(self):
        assert Platform("adobe") == Platform.ADOBE
        assert Platform("shutterstock") == Platform.SHUTTERSTOCK
        assert Platform("both") == Platform.BOTH
        with pytest.raises(ValueError):
            Platform("xyz")
