"""Tests for platform compliance checks (lax warnings)."""

from __future__ import annotations

from pathlib import Path

from src.modules.analysis.platform_compliance import (
    PlatformCompliance,
    check_adobe_compliance,
    check_shutterstock_compliance,
)


class TestAdobeCompliance:
    def test_clean_image_no_warnings(self):
        ready, warnings = check_adobe_compliance(
            Path("x.jpg"),
            size_mb=10.0,
            megapixels=12.0,
            fmt="jpeg",
            color_space="sRGB",
        )
        assert ready is True
        assert warnings == []

    def test_low_resolution_warns_but_lax(self):
        ready, warnings = check_adobe_compliance(
            Path("x.jpg"), size_mb=2.0, megapixels=2.5, fmt="jpeg"
        )
        # Lax posture — never blocks
        assert ready is True
        assert any("MP" in w for w in warnings)

    def test_too_large_warns(self):
        ready, warnings = check_adobe_compliance(
            Path("x.jpg"), size_mb=60.0, megapixels=20.0, fmt="jpeg"
        )
        assert ready is True
        assert any("45" in w or "Mo" in w for w in warnings)

    def test_too_many_megapixels_warns(self):
        ready, warnings = check_adobe_compliance(
            Path("x.jpg"), size_mb=10.0, megapixels=120.0, fmt="jpeg"
        )
        assert ready is True
        assert any("100" in w for w in warnings)

    def test_cmyk_warns(self):
        _, warnings = check_adobe_compliance(
            Path("x.jpg"), size_mb=5.0, megapixels=10.0, fmt="jpeg", color_space="CMYK"
        )
        assert any("CMJN" in w or "CMYK" in w for w in warnings)

    def test_non_jpeg_warns(self):
        _, warnings = check_adobe_compliance(
            Path("x.tif"), size_mb=5.0, megapixels=10.0, fmt="tif"
        )
        assert any("JPEG" in w for w in warnings)


class TestShutterstockCompliance:
    def test_clean_image_no_warnings(self):
        ready, warnings = check_shutterstock_compliance(
            Path("x.jpg"), size_mb=10.0, megapixels=12.0, fmt="jpeg"
        )
        assert ready is True
        assert warnings == []

    def test_size_over_50mb_warns(self):
        ready, warnings = check_shutterstock_compliance(
            Path("x.jpg"), size_mb=55.0, megapixels=15.0, fmt="jpeg"
        )
        assert ready is True
        assert any("50" in w for w in warnings)

    def test_low_mp_warns(self):
        ready, warnings = check_shutterstock_compliance(
            Path("x.jpg"), size_mb=3.0, megapixels=2.0, fmt="jpeg"
        )
        assert any("4" in w for w in warnings)


class TestComplianceDataclass:
    def test_all_warnings_dedup(self):
        pc = PlatformCompliance(file_path=Path("x.jpg"))
        pc.adobe_warnings = ["Adobe : warning A", "Common warning"]
        pc.shutterstock_warnings = ["Common warning", "Shutterstock : warning B"]
        all_w = pc.all_warnings()
        assert all_w.count("Common warning") == 1
        assert len(all_w) == 3
