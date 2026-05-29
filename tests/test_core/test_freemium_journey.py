"""End-to-end freemium journey through the ShutterstockAIv2 facade.

Walks the exact path a real user takes — Community with a teaser quota,
the quota running out, pasting a Pro key, every quality feature
unlocking, then removing the key and falling back to Community. This is
the single test that ties criterion 29 ("parcours gratuit → pro testé
de bout en bout") together; the finer-grained rules live in
``test_licensing.py``.

The autouse ``_isolate_license_file`` fixture (``tests/conftest.py``)
redirects ``DEFAULT_LICENSE_PATH`` to a tmp file, so the activate /
deactivate calls here never touch the real ``~/.shutterstock_ai``.
"""

from __future__ import annotations

from src.modules.licensing import (
    COMMUNITY_EXPERT_REPORT_QUOTA,
    Tier,
    generate_license_key,
)

# The three features the 2026-05-27 pivot put behind the Pro wall.
_QUALITY_FEATURES = ("expert_report", "dual_csv_export", "ai_enrichment")


def _facade(tmp_path):
    from src.modules.integration import ShutterstockAIv2

    return ShutterstockAIv2(db_path=tmp_path / "journey.db")


def test_full_free_to_pro_to_free_journey(tmp_path):
    api = _facade(tmp_path)
    try:
        # 1. Fresh install = Community, every quality feature locked.
        assert api.license.is_pro() is False
        for feat in _QUALITY_FEATURES:
            assert api.license.has_feature(feat) is False
        assert api.expert_report_quota_remaining() == COMMUNITY_EXPERT_REPORT_QUOTA

        # 2. Burn the teaser quota → the report would now show the upsell.
        for _ in range(COMMUNITY_EXPERT_REPORT_QUOTA):
            api.consume_expert_report_quota()
        assert api.expert_report_quota_remaining() == 0
        assert api.license.has_feature("expert_report") is False

        # 3. Paste a valid Pro key → everything unlocks, quota goes infinite.
        key = generate_license_key(email="buyer@example.com", tier=Tier.PRO_SOLO)
        ok, msg = api.activate_license(key)
        assert ok is True, msg
        assert api.license.is_pro() is True
        for feat in _QUALITY_FEATURES:
            assert api.license.has_feature(feat) is True
        assert api.expert_report_quota_remaining() == -1

        # 4. Remove the key → back to Community, features re-lock.
        ok, _ = api.deactivate_license()
        assert ok is True
        assert api.license.is_pro() is False
        for feat in _QUALITY_FEATURES:
            assert api.license.has_feature(feat) is False
    finally:
        api.close()


def test_tampered_pro_key_is_rejected_end_to_end(tmp_path):
    """A forged key (tier escalated after signing) must never unlock Pro."""
    api = _facade(tmp_path)
    try:
        key = generate_license_key(email="buyer@example.com", tier=Tier.PRO_SOLO)
        key["tier"] = "lifetime"  # tamper after the HMAC was computed
        ok, _ = api.activate_license(key)
        assert ok is False
        assert api.license.is_pro() is False
    finally:
        api.close()
