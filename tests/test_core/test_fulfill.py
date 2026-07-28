"""Tests for the sale-fulfilment helper (tools/fulfill.py).

The script is what stands between a paying customer and their licence
key, so the parts that can silently go wrong are pinned here: buyer
field extraction (Gumroad has renamed those fields over time), the
ledger that prevents double-issuing, and the SMTP gate that must stay
shut unless every credential is present.

Nothing here signs a real key or touches the network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_fulfill():
    """Import tools/fulfill.py by path — tools/ is not a package."""
    spec = importlib.util.spec_from_file_location("fulfill", REPO_ROOT / "tools" / "fulfill.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fulfill"] = module
    spec.loader.exec_module(module)
    return module


fulfill = _load_fulfill()


class TestBuyerExtraction:
    """Gumroad has used several names for the buyer email over time."""

    @pytest.mark.parametrize("field", ["email", "purchase_email", "buyer_email"])
    def test_email_found_under_each_known_field(self, field):
        assert fulfill.sale_email({field: "buyer@example.com"}) == "buyer@example.com"

    def test_email_is_stripped(self):
        assert fulfill.sale_email({"email": "  buyer@example.com  "}) == "buyer@example.com"

    def test_missing_email_returns_none(self):
        # A sale we cannot email must not raise — it gets skipped and logged.
        assert fulfill.sale_email({"id": "abc"}) is None

    def test_empty_email_treated_as_missing(self):
        assert fulfill.sale_email({"email": ""}) is None

    @pytest.mark.parametrize("field", ["full_name", "purchaser_name", "name"])
    def test_name_found_under_each_known_field(self, field):
        assert fulfill.sale_name({field: "Jane Doe"}) == "Jane Doe"

    def test_name_absent_is_empty_string(self):
        assert fulfill.sale_name({}) == ""


class TestEmailRendering:
    def test_key_is_embedded_verbatim(self):
        key = '{\n  "email": "a@b.c",\n  "tier": "lifetime"\n}'
        body = fulfill.render_email(key, "Jane Doe")
        assert key in body

    def test_first_name_only_in_greeting(self):
        body = fulfill.render_email("{}", "Jane Doe")
        assert body.startswith("Hi Jane,")

    def test_anonymous_greeting_has_no_dangling_space(self):
        body = fulfill.render_email("{}", "")
        assert body.startswith("Hi,")

    def test_activation_path_matches_the_app(self):
        # If the UI path changes, this test should fail loudly — a wrong
        # path in the email strands the buyer on activation.
        body = fulfill.render_email("{}", "")
        assert "PARAMETRES" in body
        assert "Licence" in body
        assert "Activer" in body


class TestLedger:
    def test_missing_ledger_starts_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fulfill, "LEDGER_PATH", tmp_path / "nope.json")
        assert fulfill.load_ledger() == {"fulfilled": {}}

    def test_corrupt_ledger_does_not_crash(self, tmp_path, monkeypatch):
        bad = tmp_path / "ledger.json"
        bad.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(fulfill, "LEDGER_PATH", bad)
        # Better to re-issue than to abort fulfilment entirely.
        assert fulfill.load_ledger() == {"fulfilled": {}}

    def test_round_trip(self, tmp_path, monkeypatch):
        path = tmp_path / "keys" / "_fulfilled.json"
        monkeypatch.setattr(fulfill, "LEDGER_PATH", path)
        monkeypatch.setattr(fulfill, "KEYS_DIR", path.parent)
        fulfill.save_ledger({"fulfilled": {"sale_1": {"email": "a@b.c"}}})
        assert fulfill.load_ledger()["fulfilled"]["sale_1"]["email"] == "a@b.c"


class TestSmtpGate:
    """Sending must stay off unless every credential is supplied."""

    REQUIRED = ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM")

    def _clear(self, monkeypatch, tmp_path):
        for key in (*self.REQUIRED, "SMTP_PORT"):
            monkeypatch.delenv(key, raising=False)
        # Point at a repo root with no .env so the file fallback is empty.
        monkeypatch.setattr(fulfill, "REPO_ROOT", tmp_path)

    def test_no_config_means_no_sending(self, monkeypatch, tmp_path):
        self._clear(monkeypatch, tmp_path)
        assert fulfill.smtp_settings() is None

    @pytest.mark.parametrize("missing", REQUIRED)
    def test_one_missing_credential_disables_sending(self, monkeypatch, tmp_path, missing):
        self._clear(monkeypatch, tmp_path)
        for key in self.REQUIRED:
            if key != missing:
                monkeypatch.setenv(key, "value")
        assert fulfill.smtp_settings() is None

    def test_complete_config_enables_sending(self, monkeypatch, tmp_path):
        self._clear(monkeypatch, tmp_path)
        for key in self.REQUIRED:
            monkeypatch.setenv(key, "value")
        settings = fulfill.smtp_settings()
        assert settings is not None
        assert settings["SMTP_PORT"] == "587"  # sensible default

    def test_env_file_supplies_credentials(self, monkeypatch, tmp_path):
        self._clear(monkeypatch, tmp_path)
        (tmp_path / ".env").write_text(
            "SMTP_HOST=smtp.example.com\nSMTP_USER=u\nSMTP_PASSWORD=p\nSMTP_FROM=f@example.com\n",
            encoding="utf-8",
        )
        assert fulfill.smtp_settings() is not None


class TestSalesParsing:
    def test_sales_payload_shape(self):
        # Guards the key we read out of the CLI response.
        payload = json.loads('{"success": true, "sales": [{"id": "1", "email": "a@b.c"}]}')
        assert payload["sales"][0]["id"] == "1"
        assert fulfill.sale_email(payload["sales"][0]) == "a@b.c"
