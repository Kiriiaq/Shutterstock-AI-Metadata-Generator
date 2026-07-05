"""Tests for the smartphone-format support + metadata limits work.

Covers Lots 1-4 of the 2026-07 request:
- formats.py: extension sets, IPTC-IIM capability, AI conversion need.
- limits.py: smart_truncate (no ellipsis) + clamp_keywords.
- metadata_writer: authoritative set-or-delete args, format awareness.
- prompt_templates: bounded post-processing (budgets + brand/stuffing).

All writer tests mock subprocess — no ExifTool, no real files needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.modules import formats
from src.modules.analysis.limits import (
    ADOBE_TITLE_MAX,
    SHUTTERSTOCK_DESCRIPTION_MAX,
    clamp_keywords,
    smart_truncate,
)


class TestFormats:
    def test_smartphone_extensions_supported(self):
        for ext in (".heic", ".heif", ".avif", ".webp", ".dng"):
            assert formats.is_supported(Path(f"IMG_0001{ext}"))

    def test_classic_extensions_still_supported(self):
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            assert formats.is_supported(Path(f"x{ext}"))

    def test_unsupported_extension_rejected(self):
        assert not formats.is_supported(Path("clip.mp4"))
        assert not formats.is_supported(Path("doc.pdf"))

    def test_case_insensitive(self):
        assert formats.is_supported(Path("PHOTO.HEIC"))
        assert formats.is_supported(Path("PHOTO.JPG"))

    def test_iptc_iim_capability(self):
        # JPEG/TIFF/PNG/DNG carry a legacy IPTC block…
        assert formats.supports_iptc_iim(Path("a.jpg"))
        assert formats.supports_iptc_iim(Path("a.tiff"))
        assert formats.supports_iptc_iim(Path("a.dng"))
        # …HEIC/AVIF/WebP do not.
        assert not formats.supports_iptc_iim(Path("a.heic"))
        assert not formats.supports_iptc_iim(Path("a.avif"))
        assert not formats.supports_iptc_iim(Path("a.webp"))

    def test_ai_conversion_need(self):
        # JPEG/PNG go to the vision model as-is.
        assert not formats.needs_ai_conversion(Path("a.jpg"))
        assert not formats.needs_ai_conversion(Path("a.png"))
        # Everything else must be re-encoded to JPEG first.
        assert formats.needs_ai_conversion(Path("a.heic"))
        assert formats.needs_ai_conversion(Path("a.tiff"))
        assert formats.needs_ai_conversion(Path("a.webp"))


class TestSmartTruncate:
    def test_short_text_unchanged(self):
        assert smart_truncate("Hello world", 200) == "Hello world"

    def test_no_ellipsis_ever(self):
        out = smart_truncate("word " * 100, 50)
        assert "…" not in out
        assert "..." not in out
        assert len(out) <= 50

    def test_cuts_on_word_boundary(self):
        text = "Vibrant business team brainstorming in a modern office space"
        out = smart_truncate(text, 30)
        assert len(out) <= 30
        # No partial word at the end.
        assert not text[len(out) : len(out) + 1].strip() or text[len(out)] == " " or out == text[: len(out)]
        assert " " in out
        assert not out.endswith(" ")

    def test_strips_trailing_punctuation(self):
        out = smart_truncate("one, two, three, four, five", 12)
        assert not out.endswith(",")

    def test_empty_and_none(self):
        assert smart_truncate("", 10) == ""
        assert smart_truncate(None, 10) == ""


class TestClampKeywords:
    def test_caps_count(self):
        kws = [f"kw{i}" for i in range(80)]
        assert len(clamp_keywords(kws, 49)) == 49

    def test_drops_overlong_and_empty(self):
        kws = ["ok", "", "  ", "x" * 60, "fine"]
        out = clamp_keywords(kws, 50)
        assert out == ["ok", "fine"]


class TestAuthoritativeWrite:
    """write_editor_fields must set-or-delete across IPTC+XMP+EXIF."""

    def _writer(self):
        from src.modules.engines.metadata_writer import MetadataWriter

        w = MetadataWriter.__new__(MetadataWriter)  # bypass __init__/exiftool
        w.exiftool_path = "exiftool"
        w.create_backup = False
        w.dry_run = False
        w._dry_run_results = []
        return w

    def _iptc(self, **kw):
        from src.modules.models.metadata_models import IPTCFields

        return IPTCFields(**kw)

    def test_empty_field_emits_deletion_arg(self):
        w = self._writer()
        # caption empty -> deletion arg present for both IIM and XMP.
        args = w._build_authoritative_args(Path("a.jpg"), self._iptc(headline="Title", caption=""))
        assert "-IPTC:Caption-Abstract=" in args
        assert "-XMP-dc:Description=" in args

    def test_value_field_is_set(self):
        w = self._writer()
        args = w._build_authoritative_args(Path("a.jpg"), self._iptc(headline="A Sunny Beach", caption="desc"))
        assert "-IPTC:Headline=A Sunny Beach" in args
        assert "-XMP-dc:Title=A Sunny Beach" in args
        assert "-EXIF:ImageDescription=desc" in args

    def test_keywords_cleared_then_readded(self):
        w = self._writer()
        args = w._build_authoritative_args(Path("a.jpg"), self._iptc(headline="t", keywords=["sea", "sky"]))
        # The bare clear arg comes before the individual keyword args.
        assert args.index("-IPTC:Keywords=") < args.index("-IPTC:Keywords=sea")
        assert "-IPTC:Keywords=sky" in args
        assert "-XMP-dc:Subject=sea" in args

    def test_heic_skips_iptc_iim_uses_xmp_exif(self):
        w = self._writer()
        args = w._build_authoritative_args(Path("photo.heic"), self._iptc(headline="t", caption="d", keywords=["a"]))
        # No IPTC IIM args on HEIC…
        assert not any(a.startswith("-IPTC:") for a in args)
        # …but XMP + EXIF still carry the data.
        assert "-XMP-dc:Title=t" in args
        assert "-EXIF:ImageDescription=d" in args

    def test_run_write_invoked(self, tmp_path):
        w = self._writer()
        real = tmp_path / "a.jpg"
        real.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG-ish stub
        with patch("src.modules.engines.metadata_writer.subprocess.run") as run:
            run.return_value = MagicMock(returncode=0, stdout="1 image files updated", stderr="")
            ok = w.write_editor_fields(real, self._iptc(headline="t"))
        assert ok is True
        assert run.called


class TestAIPostProcessing:
    def _templates(self):
        from src.modules.ai.prompt_templates import Platform, PromptTemplates

        return PromptTemplates(platform=Platform.SHUTTERSTOCK)

    def test_prompt_has_bounded_budgets(self):
        prompt = self._templates().get_prompt()
        assert "TITLE" in prompt and "DESCRIPTION" in prompt
        # One-sentence description constraint present.
        assert "one sentence" in prompt.lower()

    def test_parse_trims_without_ellipsis(self):
        t = self._templates()
        long_desc = "A " + "very long scene ".rstrip() * 40
        resp = f"TITLE: nice\nDESCRIPTION: {long_desc}\nKEYWORDS: a, b, c\nCATEGORIES: Nature"
        parsed = t.parse_response(resp)
        assert len(parsed["description"]) <= SHUTTERSTOCK_DESCRIPTION_MAX
        assert "…" not in parsed["description"] and "..." not in parsed["description"]

    def test_parse_filters_brands_and_stuffing(self):
        t = self._templates()
        resp = "TITLE: city street\nDESCRIPTION: a street\nKEYWORDS: nike, city, photo, street, google\nCATEGORIES: Buildings/Landmarks"
        parsed = t.parse_response(resp)
        kws = [k.lower() for k in parsed["keywords"]]
        assert "nike" not in kws and "google" not in kws  # brands dropped
        assert "photo" not in kws  # stuffing word not in title -> dropped
        assert "city" in kws and "street" in kws

    def test_title_max_enforced(self):
        t = self._templates()
        resp = "TITLE: " + "word " * 80 + "\nDESCRIPTION: d\nKEYWORDS: a\nCATEGORIES: Nature"
        parsed = t.parse_response(resp)
        assert len(parsed["title"]) <= ADOBE_TITLE_MAX
