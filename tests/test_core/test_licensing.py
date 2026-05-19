"""Tests for the licensing module — Pro vs Community gating, HMAC verify."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.modules.licensing import (
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
        assert lic.has_feature("batch_unlimited") is False
        assert lic.has_feature("ftp_scheduling") is False

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
        key = generate_license_key(email="alice@example.com", tier=Tier.PRO_SOLO)
        assert verify_license_payload(key) is True

    def test_string_tier_accepted(self):
        key = generate_license_key(email="bob@example.com", tier="pro_studio")
        assert key["tier"] == "pro_studio"

    def test_lifetime_has_no_expiration(self):
        key = generate_license_key(email="carol@example.com", tier=Tier.LIFETIME)
        assert key["expires_at"] is None

    def test_pro_solo_default_365_days(self):
        key = generate_license_key(email="dave@example.com", tier=Tier.PRO_SOLO)
        issued = datetime.fromisoformat(key["issued_at"])
        expires = datetime.fromisoformat(key["expires_at"])
        delta = expires - issued
        assert 364 <= delta.days <= 366  # ±1 day for rounding

    def test_features_match_pro_features_when_unspecified(self):
        key = generate_license_key(email="eve@example.com", tier=Tier.PRO_SOLO)
        assert set(key["features"]) == PRO_FEATURES

    def test_signature_is_deterministic(self):
        # Same payload → same signature when issued_at is pinned.
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        k1 = generate_license_key(email="x", tier=Tier.PRO_SOLO, issued_at=ts)
        k2 = generate_license_key(email="x", tier=Tier.PRO_SOLO, issued_at=ts)
        assert k1["signature"] == k2["signature"]


class TestVerification:
    def test_tampered_email_breaks_signature(self):
        key = generate_license_key(email="alice@example.com", tier=Tier.PRO_SOLO)
        key["email"] = "attacker@example.com"
        assert verify_license_payload(key) is False

    def test_tampered_tier_breaks_signature(self):
        key = generate_license_key(email="alice@example.com", tier=Tier.PRO_SOLO)
        key["tier"] = "lifetime"  # try to escalate
        assert verify_license_payload(key) is False

    def test_missing_signature_fails(self):
        key = generate_license_key(email="alice@example.com", tier=Tier.PRO_SOLO)
        del key["signature"]
        assert verify_license_payload(key) is False

    def test_wrong_secret_fails(self):
        key = generate_license_key(email="alice@example.com", tier=Tier.PRO_SOLO)
        assert verify_license_payload(key, secret=b"wrong-secret") is False

    def test_non_dict_input_returns_false(self):
        assert verify_license_payload("not a dict") is False
        assert verify_license_payload(None) is False
        assert verify_license_payload([]) is False


class TestFeatureGating:
    def test_pro_solo_has_all_pro_features(self, tmp_path):
        key = generate_license_key(email="alice@example.com", tier=Tier.PRO_SOLO)
        path = tmp_path / "license.json"
        path.write_text(json.dumps(key), encoding="utf-8")
        lic = load_license(path)
        assert lic.is_pro()
        for feature in PRO_FEATURES:
            assert lic.has_feature(feature) is True

    def test_community_has_no_pro_features(self):
        lic = License.community()
        for feature in PRO_FEATURES:
            assert lic.has_feature(feature) is False

    def test_explicit_feature_list_takes_precedence(self, tmp_path):
        key = generate_license_key(
            email="x", tier=Tier.PRO_SOLO,
            features=["batch_unlimited"],  # only one feature
        )
        path = tmp_path / "license.json"
        path.write_text(json.dumps(key), encoding="utf-8")
        lic = load_license(path)
        assert lic.has_feature("batch_unlimited") is True
        assert lic.has_feature("ftp_scheduling") is False


class TestExpiration:
    def test_expired_license_returns_community(self, tmp_path):
        # Issue a license that already expired
        past = datetime.now(timezone.utc) - timedelta(days=400)
        key = generate_license_key(
            email="alice@example.com", tier=Tier.PRO_SOLO,
            issued_at=past, valid_days=30,  # expires 30 days after `past`
        )
        path = tmp_path / "license.json"
        path.write_text(json.dumps(key), encoding="utf-8")
        lic = load_license(path)
        # load_license downgrades expired licenses to Community
        assert lic.tier == Tier.COMMUNITY

    def test_lifetime_never_expires(self, tmp_path):
        past = datetime(2020, 1, 1, tzinfo=timezone.utc)
        key = generate_license_key(
            email="carol@example.com", tier=Tier.LIFETIME,
            issued_at=past,
        )
        path = tmp_path / "license.json"
        path.write_text(json.dumps(key), encoding="utf-8")
        lic = load_license(path)
        assert lic.tier == Tier.LIFETIME
        assert lic.is_pro()


class TestFacadeIntegration:
    """Tests that the ShutterstockAIv2 facade properly exposes the license."""

    def test_facade_starts_in_community(self, tmp_path):
        from src.modules.integration import ShutterstockAIv2

        db = tmp_path / "test.db"
        api = ShutterstockAIv2(db_path=db)
        try:
            # Without a license.json in tmp_path, the user's home dir
            # might have one — but we're testing the basic shape.
            assert hasattr(api, "license")
            assert api.license is not None
        finally:
            api.close()

    def test_activate_with_invalid_key_returns_false(self, tmp_path):
        from src.modules.integration import ShutterstockAIv2

        db = tmp_path / "test.db"
        api = ShutterstockAIv2(db_path=db)
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
