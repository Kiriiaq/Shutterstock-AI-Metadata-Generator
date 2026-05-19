"""Tests for the heuristic expert-report builder."""

from __future__ import annotations

from pathlib import Path

from src.modules.analysis.expert_report import (
    _clean_keywords,
    _merge_keywords,
    build_expert_report,
    enrich_with_ai_result,
)
from src.modules.analysis.platform_compliance import PlatformCompliance
from src.modules.models.metadata_models import (
    IPTCFields,
)


def _make_compliance(
    *,
    mp: float = 12.0,
    size_mb: float = 8.0,
    fmt: str = "jpeg",
    color_space: str = "sRGB",
) -> PlatformCompliance:
    return PlatformCompliance(
        file_path=Path("dummy.jpg"),
        file_size_mb=size_mb,
        megapixels=mp,
        format=fmt,
        color_space=color_space,
    )


class TestKeywordCleaning:
    def test_dedup_lowercase(self):
        out = _clean_keywords(["Cat", "cat", "DOG", " dog "], title="")
        assert out == ["cat", "dog"]

    def test_drops_brands(self):
        out = _clean_keywords(["nike", "running", "shoes"], title="")
        assert "nike" not in out
        assert "running" in out

    def test_keeps_stuffing_if_in_title(self):
        out = _clean_keywords(["photo", "lake", "nature"], title="Lake Photo")
        # "photo" is in title, so it survives the stuffing filter
        assert "photo" in out

    def test_drops_stuffing_if_not_in_title(self):
        out = _clean_keywords(["stock", "image", "lake"], title="Mountain Lake")
        assert "stock" not in out
        assert "image" not in out
        assert "lake" in out

    def test_caps_at_50(self):
        many = [f"kw{i}" for i in range(80)]
        out = _clean_keywords(many, title="")
        assert len(out) == 50

    def test_merge_preserves_base_order(self):
        merged = _merge_keywords(["lake", "mountain"], ["forest", "lake"], title="")
        assert merged[0] == "lake"
        assert merged[1] == "mountain"
        assert "forest" in merged


class TestHeuristicBuilder:
    def test_minimal_iptc_no_image(self, tmp_path):
        path = tmp_path / "sample.jpg"
        path.write_bytes(b"fake")  # not a real JPEG; PIL probe fails silently

        iptc = IPTCFields(
            headline="Sunset over mountain lake",
            caption="Calm evening at high altitude in the Alps",
            keywords=["lake", "mountain", "sunset", "nature", "calm", "alps", "outdoor"],
            supplemental_categories=["Nature"],
        )
        compliance = _make_compliance()

        report = build_expert_report(path, iptc=iptc, compliance=compliance)

        assert report.source == "heuristic"
        assert report.title_adobe == "Sunset over mountain lake"
        assert report.title_shutterstock == report.title_adobe
        assert report.description.startswith("Calm evening")
        assert len(report.keywords) >= 7
        assert "Nature" in report.categories_shutterstock
        assert report.category_adobe_primary == "Landscapes"

    def test_filename_fallback_title(self, tmp_path):
        path = tmp_path / "snow_white-mountain_07.jpg"
        path.write_bytes(b"fake")
        report = build_expert_report(
            path,
            iptc=IPTCFields(),
            compliance=_make_compliance(),
        )
        # underscores and dashes turned into spaces, title-cased
        assert report.title_adobe == "Snow White Mountain 07"

    def test_low_resolution_triggers_blocker(self, tmp_path):
        path = tmp_path / "tiny.jpg"
        path.write_bytes(b"fake")
        compliance = _make_compliance(mp=2.5)
        report = build_expert_report(
            path,
            iptc=IPTCFields(
                headline="Tiny image",
                keywords=["a", "b", "c", "d", "e", "f", "g"],  # 7 keywords
            ),
            compliance=compliance,
        )

        blockers = [r for r in report.rejection_risks if r.severity == "blocker"]
        assert any("MP" in r.issue for r in blockers)
        assert report.scores.rejection_risk >= 3
        assert report.scores.technical <= 7

    def test_missing_keywords_triggers_blocker(self, tmp_path):
        path = tmp_path / "x.jpg"
        path.write_bytes(b"fake")
        report = build_expert_report(
            path,
            iptc=IPTCFields(headline="Some title"),
            compliance=_make_compliance(),
        )
        assert any(
            "mots-clés" in r.issue.lower() or "mots-cles" in r.issue.lower()
            for r in report.rejection_risks
        )

    def test_lax_no_warnings_for_compliant_image(self, tmp_path):
        path = tmp_path / "x.jpg"
        path.write_bytes(b"fake")
        report = build_expert_report(
            path,
            iptc=IPTCFields(
                headline="Vibrant business team brainstorming in modern office",
                caption="Diverse colleagues collaborate on a digital marketing strategy.",
                keywords=[
                    "business",
                    "team",
                    "brainstorm",
                    "office",
                    "diverse",
                    "marketing",
                    "strategy",
                    "modern",
                    "meeting",
                    "collaboration",
                ],
                supplemental_categories=["Business/Finance"],
            ),
            compliance=_make_compliance(),
        )
        # No blockers expected
        assert all(r.severity != "blocker" for r in report.rejection_risks)
        assert report.scores.commercial >= 6
        assert report.scores.seo >= 5

    def test_marketing_uses_filled(self, tmp_path):
        path = tmp_path / "x.jpg"
        path.write_bytes(b"fake")
        report = build_expert_report(
            path,
            iptc=IPTCFields(
                headline="Title",
                keywords=["k"] * 10,
                supplemental_categories=["Technology"],
            ),
            compliance=_make_compliance(),
        )
        assert "campagne IA" in report.marketing_uses or "landing page SaaS" in report.marketing_uses


class TestAIEnrichment:
    def test_overlay_ai_keywords_and_title(self, tmp_path):
        path = tmp_path / "x.jpg"
        path.write_bytes(b"fake")
        base = build_expert_report(
            path,
            iptc=IPTCFields(
                headline="Old title",
                keywords=["old1", "old2"],
            ),
            compliance=_make_compliance(),
        )
        ai_dict = {
            "title": "New refined title from AI",
            "keywords": ["fresh", "vibrant", "modern"],
            "categories": ["Business/Finance"],
            "scores": {"commercial": 9, "technical": 8, "seo": 7, "rejection_risk": 1},
        }
        enriched = enrich_with_ai_result(base, ai_dict)
        assert enriched.title_adobe == "New refined title from AI"
        assert enriched.title_shutterstock == "New refined title from AI"
        assert "fresh" in enriched.keywords
        assert "old1" in enriched.keywords  # base keywords preserved
        assert enriched.category_adobe_primary == "Business"
        assert enriched.scores.commercial == 9
        assert enriched.source in {"hybrid", "ai"}

    def test_overlay_robust_to_missing_keys(self, tmp_path):
        path = tmp_path / "x.jpg"
        path.write_bytes(b"fake")
        base = build_expert_report(
            path,
            iptc=IPTCFields(headline="Original", keywords=["k1", "k2"]),
            compliance=_make_compliance(),
        )
        enriched = enrich_with_ai_result(base, {})  # empty dict
        assert enriched.title_adobe == "Original"
        assert enriched.keywords == base.keywords

    def test_overlay_ignores_non_dict(self, tmp_path):
        path = tmp_path / "x.jpg"
        path.write_bytes(b"fake")
        base = build_expert_report(
            path,
            iptc=IPTCFields(headline="Original"),
            compliance=_make_compliance(),
        )
        enriched = enrich_with_ai_result(base, "not a dict")  # type: ignore[arg-type]
        assert enriched.title_adobe == "Original"

    def test_overlay_clamps_score(self, tmp_path):
        path = tmp_path / "x.jpg"
        path.write_bytes(b"fake")
        base = build_expert_report(
            path,
            iptc=IPTCFields(headline="x"),
            compliance=_make_compliance(),
        )
        enriched = enrich_with_ai_result(base, {"scores": {"commercial": 99, "technical": -5}})
        assert enriched.scores.commercial == 10
        assert enriched.scores.technical == 0


class TestReportSerialisation:
    def test_to_dict_round_trip(self, tmp_path):
        path = tmp_path / "x.jpg"
        path.write_bytes(b"fake")
        report = build_expert_report(
            path,
            iptc=IPTCFields(
                headline="Hello",
                keywords=["a", "b", "c", "d", "e", "f", "g", "h"],
                supplemental_categories=["Nature"],
            ),
            compliance=_make_compliance(),
        )
        d = report.to_dict()
        assert d["scores"]["technical"] >= 0
        assert d["categories_shutterstock"] == ["Nature"]
        assert d["category_adobe_primary"] == "Landscapes"

    def test_adobe_csv_row_columns(self, tmp_path):
        path = tmp_path / "x.jpg"
        path.write_bytes(b"fake")
        report = build_expert_report(
            path,
            iptc=IPTCFields(headline="Title", keywords=["alpha", "beta"]),
            compliance=_make_compliance(),
        )
        row = report.to_adobe_csv_row()
        assert set(row.keys()) == {"Filename", "Title", "Keywords", "Category", "Releases"}
        assert row["Filename"] == "x.jpg"
        assert row["Keywords"] == "alpha, beta"

    def test_shutterstock_csv_keywords_use_comma(self, tmp_path):
        path = tmp_path / "x.jpg"
        path.write_bytes(b"fake")
        report = build_expert_report(
            path,
            iptc=IPTCFields(headline="t", keywords=["alpha", "beta"]),
            compliance=_make_compliance(),
        )
        row = report.to_shutterstock_csv_row()
        assert row["Keywords"] == "alpha, beta"
        assert "Editorial" in row
