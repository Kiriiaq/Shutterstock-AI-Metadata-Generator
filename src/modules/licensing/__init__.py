"""Licence management — Community vs Pro tiering, offline-verified.

Public surface:

- :class:`License` — dataclass loaded from JSON, exposes ``is_pro()``,
  ``has_feature(name)``, ``tier()``, ``expires_at()``.
- :func:`load_license` — find + parse + verify the user's license file.
- :func:`generate_license_key` — server-side helper to produce a signed
  payload (used by ``tools/generate_license.py``, never by the app).

Cryptography: HMAC-SHA256 of the canonical JSON payload (sorted keys),
signed with a secret known only to the project maintainer. The EXE
ships **without** the secret — only the verify path needs the matching
key, which is **embedded as the public verification key** (the same
secret, since HMAC is symmetric). The trade-off is documented in
:func:`generate_license_key`.

For a hardened release, swap to ed25519 signatures (public verify key
in the EXE, private signing key on the dev machine). Not done here to
keep the dep footprint at zero — `hmac` is stdlib.
"""

from .license import (
    COMMUNITY_EXPORT_QUOTA,
    DEFAULT_LICENSE_PATH,
    PRO_FEATURES,
    License,
    LicenseError,
    LicenseExpiredError,
    LicenseInvalidError,
    Tier,
    generate_license_key,
    load_license,
    verify_license_payload,
)

__all__ = [
    "COMMUNITY_EXPORT_QUOTA",
    "DEFAULT_LICENSE_PATH",
    "License",
    "LicenseError",
    "LicenseExpiredError",
    "LicenseInvalidError",
    "PRO_FEATURES",
    "Tier",
    "generate_license_key",
    "load_license",
    "verify_license_payload",
]
