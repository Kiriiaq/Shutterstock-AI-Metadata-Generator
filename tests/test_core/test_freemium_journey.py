"""End-to-end freemium journey through the ShutterstockAIv2 facade.

Walks the exact path a real user takes in the v2.2.0 model — Community
with a free export quota, the quota running out, pasting the 10 €
lifetime key, the data export unlocking, then removing the key and
falling back to Community. This is the single test that ties
criterion 29 ("parcours gratuit → pro testé de bout en bout")
together; finer-grained rules live in ``test_licensing.py``.

The autouse ``_isolate_license_file`` fixture (``tests/conftest.py``)
redirects ``DEFAULT_LICENSE_PATH`` to a tmp file, so the activate /
deactivate calls here never touch the real ``~/.shutterstock_ai``.
"""

from __future__ import annotations

from src.modules.licensing import (
    COMMUNITY_EXPORT_QUOTA,
    Tier,
    generate_license_key,
)


def _facade(tmp_path):
    from src.modules.integration import ShutterstockAIv2

    return ShutterstockAIv2(db_path=tmp_path / "journey.db")


def test_full_free_to_pro_to_free_journey(tmp_path):
    api = _facade(tmp_path)
    try:
        api.reset_export_quota()

        # 1. Fresh install = Community; the data export is gated by quota.
        assert api.license.is_pro() is False
        assert api.license.has_feature("data_export") is False
        assert api.export_quota_remaining() == COMMUNITY_EXPORT_QUOTA

        # 2. Burn the free export runs → the next export shows the upsell.
        for _ in range(COMMUNITY_EXPORT_QUOTA):
            api.consume_export_quota()
        assert api.export_quota_remaining() == 0

        # 3. Paste the 10 € lifetime key → export unlocks, quota infinite.
        key = generate_license_key(email="buyer@example.com", tier=Tier.LIFETIME)
        ok, msg = api.activate_license(key)
        assert ok is True, msg
        assert api.license.is_pro() is True
        assert api.license.has_feature("data_export") is True
        assert api.export_quota_remaining() == -1

        # 4. Remove the key → back to Community, export re-locks.
        ok, _ = api.deactivate_license()
        assert ok is True
        assert api.license.is_pro() is False
        assert api.license.has_feature("data_export") is False
    finally:
        api.close()


def test_tampered_key_is_rejected_end_to_end(tmp_path):
    """A forged key (mutated after signing) must never unlock the export."""
    api = _facade(tmp_path)
    try:
        key = generate_license_key(email="buyer@example.com", tier=Tier.LIFETIME)
        key["email"] = "attacker@example.com"  # tamper after the HMAC was computed
        ok, _ = api.activate_license(key)
        assert ok is False
        assert api.license.is_pro() is False
        assert api.license.has_feature("data_export") is False
    finally:
        api.close()
