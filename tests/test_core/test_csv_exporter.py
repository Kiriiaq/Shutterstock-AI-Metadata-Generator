"""Tests for the double CSV exporter (Adobe + Shutterstock)."""

from __future__ import annotations

import csv
from pathlib import Path

from src.modules.analysis.expert_report import build_expert_report
from src.modules.analysis.platform_compliance import PlatformCompliance
from src.modules.export.csv_exporter import (
    ADOBE_COLUMNS,
    SHUTTERSTOCK_COLUMNS,
    export_double_csv,
    write_adobe_csv,
    write_shutterstock_csv,
)
from src.modules.models.metadata_models import IPTCFields


def _sample_report(tmp_path: Path, name: str = "sample.jpg"):
    path = tmp_path / name
    path.write_bytes(b"fake")
    compliance = PlatformCompliance(
        file_path=path,
        file_size_mb=8.0,
        megapixels=12.0,
        format="jpeg",
        color_space="sRGB",
    )
    return build_expert_report(
        path,
        iptc=IPTCFields(
            headline="Modern business team in office",
            caption="Diverse colleagues collaborate on a strategy.",
            keywords=[
                "business",
                "team",
                "office",
                "modern",
                "colleagues",
                "diversity",
                "strategy",
                "meeting",
                "collaboration",
                "professional",
            ],
            supplemental_categories=["Business/Finance"],
        ),
        compliance=compliance,
    )


class TestAdobeCsv:
    def test_columns_and_bom(self, tmp_path):
        report = _sample_report(tmp_path)
        out = tmp_path / "out_adobe.csv"
        n = write_adobe_csv([report], out)
        assert n == 1

        raw = out.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf"), "Adobe CSV must be UTF-8 with BOM"

        # parse
        with out.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert list(rows[0].keys()) == ADOBE_COLUMNS
        assert rows[0]["Filename"] == "sample.jpg"
        assert "business" in rows[0]["Keywords"]
        assert rows[0]["Category"] == "Business"


class TestShutterstockCsv:
    def test_columns_and_keywords_comma(self, tmp_path):
        report = _sample_report(tmp_path)
        out = tmp_path / "out_sh.csv"
        n = write_shutterstock_csv([report], out)
        assert n == 1

        with out.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert list(rows[0].keys()) == SHUTTERSTOCK_COLUMNS
        # The historical bug joined keywords with spaces — verify it's
        # actually commas now.
        kw = rows[0]["Keywords"]
        assert ", " in kw, f"Keywords should be comma-separated, got: {kw!r}"
        assert rows[0]["Editorial"] in {"Yes", "No"}
        assert rows[0]["Illustration"] in {"Yes", "No"}


class TestDoubleExport:
    def test_both_files_written(self, tmp_path):
        reports = [_sample_report(tmp_path, f"img_{i}.jpg") for i in range(3)]
        result = export_double_csv(reports, tmp_path, basename="batch1")

        assert result.adobe_csv == tmp_path / "batch1_adobe.csv"
        assert result.shutterstock_csv == tmp_path / "batch1_shutterstock.csv"
        assert result.adobe_csv.exists()
        assert result.shutterstock_csv.exists()
        assert result.row_count == 3
