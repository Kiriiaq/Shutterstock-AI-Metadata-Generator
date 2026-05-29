"""Tests for the licensing module — Community vs Pro, HMAC verify, export quota.

Model (v2.2.0): a single paid tier (``LIFETIME``, 10 €) unlocks the one
gated capability — ``data_export``. Everything else is free. Community
gets ``COMMUNITY_EXPORT_QUOTA`` free export runs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.modules.licensing import (
    COMMUNITY_EXPORT_QUOTA,
    PRO_FEATURES,
    License,
    Tier,
    generate_license_key,
    load_license,
    verify_license_payload,
)


class TestCommunityDefault:
    def test_community_constructor(self):
        lic = License.community()
        assert lic.tier == Tier.COMMUNITY
        assert lic.is_pro() is False
        assert lic.has_feature("data_export") is False

    def test_load_license_missing_file_returns_community(self, tmp_path):
        lic = load_license(tmp_path / "nonexistent.json")
        assert lic.tier == Tier.COMMUNITY

    def test_load_license_unreadable_file_returns_community(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("not json", encoding="utf-8")
        lic = load_license(path)
        assert lic.tier == Tier.COMMUNITY


class TestKeyGeneration:
    def test_generated_key_passes_verification(self):
        key = generate_license_key(email="alice@example.com", tier=Tier.LIFETIME)
        assert verify_license_payload(key) is True

    def test_string_tier_accepted(self):
        key = generate_license_key(email="bob@example.com", tier="lifetime")
        assert key["tier"] == "lifetime"

    def test_lifetime_has_no_expiration(self):
        key = generate_license_key(email="carol@example.com", tier=Tier.LIFETIME)
        assert key["expires_at"] is None

    def test_features_match_pro_features_when_unspecified(self):
        key = generate_license_key(email="eve@example.com", tier=Tier.LIFETIME)
        assert set(key["features"]) == PRO_FEATURES

    def test_data_export_is_the_single_gated_feature(self):
        # Pins the model: the only thing the paid tier unlocks is the export.
        assert PRO_FEATURES == {"data_export"}

    def test_signature_is_deterministic(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        k1 = generate_license_key(email="x", tier=Tier.LIFETIME, issued_at=ts)
        k2 = generate_license_key(email="x", tier=Tier.LIFETIME, issued_at=ts)
        assert k1["signature"] == k2["signature"]


class TestVerification:
    def test_tampered_email_breaks_signature(self):
        key = generate_license_key(email="alice@example.com", tier=Tier.LIFETIME)
        key["email"] = "attacker@example.com"
        assert verify_license_payload(key) is False

    def test_tampered_features_break_signature(self):
        key = generate_license_key(email="alice@example.com", tier=Tier.LIFETIME)
        key["features"] = ["data_export", "smuggled_feature"]
        assert verify_license_payload(key) is False

    def test_missing_signature_fails(self):
        key = generate_license_key(email="alice@example.com", tier=Tier.LIFETIME)
        del key["signature"]
        assert verify_license_payload(key) is False

    def test_wrong_secret_fails(self):
        key = generate_license_key(email="alice@example.com", tier=Tier.LIFETIME)
        assert verify_license_payload(key, secret=b"wrong-secret") is False

    def test_non_dict_input_returns_false(self):
        assert verify_license_payload("not a dict") is False
        assert verify_license_payload(None) is False
        assert verify_license_payload([]) is False


class TestFeatureGating:
    def test_lifetime_unlocks_data_export(self, tmp_path):
        key = generate_license_key(email="alice@example.com", tier=Tier.LIFETIME)
        path = tmp_path / "license.json"
        path.write_text(json.dumps(key), encoding="utf-8")
        lic = load_license(path)
        assert lic.is_pro()
        assert lic.has_feature("data_export") is True

    def test_community_has_no_data_export(self):
        lic = License.community()
        assert lic.has_feature("data_export") is False

    def test_explicit_feature_list_takes_precedence(self, tmp_path):
        key = generate_license_key(
            email="x",
            tier=Tier.LIFETIME,
            features=["some_other_feature"],  # explicit list, no data_export
        )
        path = tmp_path / "license.json"
        path.write_text(json.dumps(key), encoding="utf-8")
        lic = load_license(path)
        assert lic.has_feature("some_other_feature") is True
        assert lic.has_feature("data_export") is False


class TestLifetime:
    def test_lifetime_never_expires(self, tmp_path):
        # Issued years ago, yet still valid — lifetime ignores expires_at.
        past = datetime(2020, 1, 1, tzinfo=timezone.utc)
        key = generate_license_key(
            email="carol@example.com",
            tier=Tier.LIFETIME,
            issued_at=past,
        )
        path = tmp_path / "license.json"
        path.write_text(json.dumps(key), encoding="utf-8")
        lic = load_license(path)
        assert lic.tier == Tier.LIFETIME
        assert lic.is_pro()
        assert lic.is_expired() is False


class TestFacadeIntegration:
    """The ShutterstockAIv2 facade exposes the license + activation."""

    def test_facade_starts_in_community(self, tmp_path):
        from src.modules.integration import ShutterstockAIv2

        api = ShutterstockAIv2(db_path=tmp_path / "test.db")
        try:
            assert hasattr(api, "license")
            assert api.license is not None
        finally:
            api.close()

    def test_activate_with_invalid_key_returns_false(self, tmp_path):
        from src.modules.integration import ShutterstockAIv2

        api = ShutterstockAIv2(db_path=tmp_path / "test.db")
        try:
            ok, msg = api.activate_license("not even json")
            assert ok is False
            assert "JSON" in msg or "invalide" in msg.lower()

            ok, msg = api.activate_license("")
            assert ok is False

            ok, msg = api.activate_license({"tier": "lifetime"})  # no signature
            assert ok is False
        finally:
            api.close()


class TestExportQuota:
    """The COMMUNITY_EXPORT_QUOTA free export runs (the only paywall).

    Exercises the facade helpers (``export_quota_remaining`` /
    ``consume_export_quota``) headless, rather than the UI.
    """

    def _fresh_api(self, tmp_path):
        from src.modules.integration import ShutterstockAIv2

        api = ShutterstockAIv2(db_path=tmp_path / "quota.db")
        # Some dev machines have a real license.json under ~ — force
        # Community for the duration of the test so we test the gate.
        api._license = License.community()
        api.reset_export_quota()
        return api

    def test_community_starts_with_full_quota(self, tmp_path):
        api = self._fresh_api(tmp_path)
        try:
            assert api.export_quota_remaining() == COMMUNITY_EXPORT_QUOTA
        finally:
            api.close()

    def test_consume_decrements_and_stops_at_zero(self, tmp_path):
        api = self._fresh_api(tmp_path)
        try:
            for expected in range(COMMUNITY_EXPORT_QUOTA - 1, -1, -1):
                assert api.consume_export_quota() == expected
            # Further consumes stay clamped at 0 (no underflow).
            assert api.consume_export_quota() == 0
            assert api.export_quota_remaining() == 0
        finally:
            api.close()

    def test_quota_persists_across_facade_restart(self, tmp_path):
        api = self._fresh_api(tmp_path)
        db_path = tmp_path / "quota.db"
        try:
            api.consume_export_quota()
        finally:
            api.close()

        from src.modules.integration import ShutterstockAIv2

        api2 = ShutterstockAIv2(db_path=db_path)
        api2._license = License.community()
        try:
            assert api2.export_quota_remaining() == (COMMUNITY_EXPORT_QUOTA - 1)
        finally:
            api2.close()

    def test_pro_user_has_infinite_quota(self, tmp_path):
        from src.modules.integration import ShutterstockAIv2

        api = ShutterstockAIv2(db_path=tmp_path / "pro.db")
        try:
            key = generate_license_key(email="pro@example.com", tier=Tier.LIFETIME)
            ok, _ = api.activate_license(key)
            assert ok is True
            assert api.export_quota_remaining() == -1
            # Pro consume is a no-op — returns sentinel without touching
            # the persisted counter.
            assert api.consume_export_quota() == -1
        finally:
            api.deactivate_license()
            api.close()

    def test_reset_brings_counter_back_to_full_quota(self, tmp_path):
        api = self._fresh_api(tmp_path)
        try:
            api.consume_export_quota()
            api.consume_export_quota()
            assert api.export_quota_remaining() == COMMUNITY_EXPORT_QUOTA - 2
            api.reset_export_quota()
            assert api.export_quota_remaining() == COMMUNITY_EXPORT_QUOTA
        finally:
            api.close()
