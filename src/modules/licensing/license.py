"""License dataclass + HMAC verification + key generation.

The license payload is a JSON object with this schema::

    {
        "email": "alice@example.com",
        "tier": "pro_solo",        # community | pro_solo | pro_studio | lifetime
        "features": ["batch_unlimited", "ftp_scheduling", ...],
        "issued_at": "2026-05-19T18:00:00",
        "expires_at": "2027-05-19T18:00:00",   # null for "lifetime"
        "signature": "..."   # HMAC-SHA256 hex digest of the rest, sorted-keys
    }

The signing secret lives in an env var ``SSA_LICENSE_SECRET`` on the
maintainer's machine and inside the bundled EXE (compiled as a Python
constant). HMAC is symmetric, so anyone with the binary can in
principle extract the secret and forge keys — this is **honour-system
licensing**, not DRM. Acceptable for a 29 €/an product where the
target audience pays out of convenience, not out of inability to crack.

Hardening upgrade path (v2.2.0+): replace HMAC with ed25519 (PyNaCl)
so the EXE only carries the public verification key.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# Default secret used if SSA_LICENSE_SECRET env var isn't set. Replace
# at build time with a real secret (e.g. via PyInstaller --runtime-hook
# or a CI-injected constant). For now this is just a placeholder so
# the test suite and the dev loop work without env config.
#
# Anyone who reads this file can forge Pro keys — that's by design at
# this stage. See module docstring for the hardening path.
DEFAULT_DEV_SECRET = "ShutterstockAnalyzer-Dev-Secret-Replace-In-Production-v2"

# Default location of the license file on the user's machine. Same dir
# as the SQLite settings DB, so users find both in one place.
DEFAULT_LICENSE_PATH = Path.home() / ".shutterstock_ai" / "license.json"


class Tier(str, Enum):
    """Subscription tier — Community is the default unlicensed state."""

    COMMUNITY = "community"
    PRO_SOLO = "pro_solo"
    PRO_STUDIO = "pro_studio"
    LIFETIME = "lifetime"


# Features gated by the Pro tiers. The 2026-05-27 pivot reframed the
# Pro proposition around **quality evaluation** (the headline value
# the app actually delivers) instead of around batch/scheduling
# add-ons that nobody had asked for yet:
#
# - ``expert_report``     : full multi-section microstock audit
#   (4 scores, rejection risks, improvements, marketing/buyer
#   profiles, trends). Community gets a teaser quota — see
#   ``COMMUNITY_EXPERT_REPORT_QUOTA``.
# - ``dual_csv_export``   : Adobe + Shutterstock side-by-side export.
#   Community can still pick one platform at a time (simple CSV).
# - ``ai_enrichment``     : Ollama vision overlay on the heuristic
#   report. Local model, but the gating recognises the IA pass as
#   premium quality work.
# - ``batch_unlimited``   : already in v2.0 ; > 50 images per run.
# - ``ftp_scheduling``, ``ftp_multi_account``, ``iptc_templates``,
#   ``prompt_profiles``, ``priority_support`` : roadmap features
#   (not enforced yet, reserved so generated keys carry them).
PRO_FEATURES: set[str] = {
    "expert_report",     # full expert microstock audit (4 scores + risks + uses)
    "dual_csv_export",   # Adobe + Shutterstock side-by-side CSV
    "ai_enrichment",     # Ollama vision overlay on the heuristic report
    "batch_unlimited",   # > 50 images per export_batch run
    "ftp_scheduling",    # background recurring FTP push
    "ftp_multi_account", # multiple FTP profiles
    "iptc_templates",    # save/load custom IPTC templates
    "prompt_profiles",   # category-aware Ollama prompts
    "priority_support",  # 48h support SLA
}


# Number of expert reports a Community user may consume before the
# upsell modal kicks in. Tracked across sessions in the settings
# table (``community_expert_reports_used``). Two is the sweet spot
# observed in similar freemium tools: enough to demonstrate value
# on a couple of real images, low enough that a working contributor
# hits the wall on the same day they install the app.
COMMUNITY_EXPERT_REPORT_QUOTA = 2


class LicenseError(Exception):
    """Base exception for license problems."""


class LicenseInvalidError(LicenseError):
    """License signature is invalid or the JSON is malformed."""


class LicenseExpiredError(LicenseError):
    """License signature is valid but ``expires_at`` is in the past."""


@dataclass
class License:
    """Parsed, verified license. Always represents a *valid* license.

    Use :func:`load_license` (or :func:`License.community`) rather than
    constructing this dataclass directly — that constructor doesn't
    verify the signature.
    """

    email: str
    tier: Tier
    features: list[str] = field(default_factory=list)
    issued_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    @classmethod
    def community(cls) -> "License":
        """The implicit unlicensed state — every install starts here."""
        return cls(email="", tier=Tier.COMMUNITY, features=[])

    def is_pro(self) -> bool:
        """True if the user is on any paying tier."""
        return self.tier != Tier.COMMUNITY

    def has_feature(self, name: str) -> bool:
        """True if *name* is unlocked by the current tier.

        Lifetime + Studio + Solo all carry the same feature set. The
        distinction lives in seat count (Studio = up to 5) and pricing,
        not in the feature gate.
        """
        if self.is_expired():
            return False
        if self.is_pro():
            # Pro tiers grant ALL Pro features. The ``features`` field
            # exists for future granularity (per-feature licensing).
            if not self.features:
                return name in PRO_FEATURES
            return name in self.features
        return False

    def is_expired(self) -> bool:
        """Lifetime never expires; everything else honours expires_at."""
        if self.tier == Tier.LIFETIME:
            return False
        if self.expires_at is None:
            return False
        # Compare in UTC to avoid tz surprises across machines.
        now = datetime.now(timezone.utc)
        exp = self.expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return exp < now

    def to_dict(self) -> dict:
        return {
            "email": self.email,
            "tier": self.tier.value,
            "features": list(self.features),
            "issued_at": self.issued_at.isoformat() if self.issued_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


# ============================================================================
# HMAC verify / sign
# ============================================================================


def _canonical_payload(payload: dict) -> bytes:
    """Deterministic JSON bytes for HMAC. Drops ``signature`` if present."""
    body = {k: v for k, v in payload.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _secret() -> bytes:
    """Read the signing secret. Sources in priority order:

    1. ``SSA_LICENSE_SECRET`` env var (CI / dev override / build hook).
    2. ``_secret_compiled.PROD_SECRET`` constant — populated by
       ``build.py`` at PyInstaller time so the EXE ships with the real
       secret embedded. The file is gitignored.
    3. ``.env`` file at repo root with a ``SSA_LICENSE_SECRET=…`` line
       (dev convenience, ignored if absent).
    4. ``DEFAULT_DEV_SECRET`` — bundled fallback. Logs a one-time
       WARNING because licenses signed/verified with this secret are
       trivially forgeable.

    Returns:
        UTF-8 encoded bytes ready to hand to ``hmac.new``.
    """
    # 1) env var
    env = os.environ.get("SSA_LICENSE_SECRET")
    if env:
        return env.encode("utf-8")

    # 2) compiled-in constant (set by build.py before PyInstaller runs)
    try:
        from . import _secret_compiled  # type: ignore[attr-defined]
        prod = getattr(_secret_compiled, "PROD_SECRET", "")
        if prod:
            return str(prod).encode("utf-8")
    except ImportError:
        pass  # not built yet — that's fine, we have fallbacks

    # 3) .env file at repo root
    try:
        env_file = Path(__file__).resolve().parents[3] / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("SSA_LICENSE_SECRET=") and not line.startswith("#"):
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if value and value != "replace-with-secrets-token_urlsafe-32-output":
                        return value.encode("utf-8")
    except (OSError, IndexError):
        pass

    # 4) DEFAULT_DEV_SECRET fallback (logged once for visibility)
    _warn_dev_secret_once()
    return DEFAULT_DEV_SECRET.encode("utf-8")


_dev_secret_warned = False


def _warn_dev_secret_once() -> None:
    """Log the dev-secret warning at most once per Python process."""
    global _dev_secret_warned
    if _dev_secret_warned:
        return
    _dev_secret_warned = True
    logger.warning(
        "License secret falling back to the bundled DEV value — "
        "keys signed this way are trivially forgeable. "
        "Set SSA_LICENSE_SECRET in your .env or environment before "
        "shipping production builds."
    )


def _sign(payload: dict, secret: Optional[bytes] = None) -> str:
    secret = secret or _secret()
    digest = hmac.new(secret, _canonical_payload(payload), hashlib.sha256)
    return digest.hexdigest()


def verify_license_payload(payload: dict, *, secret: Optional[bytes] = None) -> bool:
    """Return True if *payload* carries a valid HMAC signature.

    Uses :func:`hmac.compare_digest` to avoid timing leaks.
    """
    if not isinstance(payload, dict):
        return False
    provided = str(payload.get("signature", ""))
    if not provided:
        return False
    expected = _sign(payload, secret=secret)
    return hmac.compare_digest(provided, expected)


# ============================================================================
# Load / parse
# ============================================================================


def load_license(path: Optional[Path] = None) -> License:
    """Read the user's license file and return a verified :class:`License`.

    On any error (file missing, invalid signature, expired), this falls
    back to :meth:`License.community` so the app continues to run in
    free mode. Errors are logged at WARNING level.
    """
    path = path or DEFAULT_LICENSE_PATH
    if not path.exists():
        logger.debug("No license file at %s — Community tier", path)
        return License.community()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("License file unreadable: %s", exc)
        return License.community()

    if not verify_license_payload(payload):
        logger.warning("License signature mismatch — running in Community mode")
        return License.community()

    try:
        tier = Tier(payload.get("tier", "community"))
    except ValueError:
        logger.warning("Unknown tier %r — Community", payload.get("tier"))
        return License.community()

    license_obj = License(
        email=str(payload.get("email", "")),
        tier=tier,
        features=list(payload.get("features", []) or []),
        issued_at=_parse_iso(payload.get("issued_at")),
        expires_at=_parse_iso(payload.get("expires_at")),
    )

    if license_obj.is_expired():
        logger.warning("License expired (%s) — Community mode", license_obj.expires_at)
        return License.community()

    logger.info("License active: %s (%s)", license_obj.tier.value, license_obj.email)
    return license_obj


def _parse_iso(value) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        # ``fromisoformat`` accepts the canonical strings we emit.
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


# ============================================================================
# Key generation (admin side only)
# ============================================================================


def generate_license_key(
    *,
    email: str,
    tier: Tier | str,
    features: Optional[Iterable[str]] = None,
    valid_days: Optional[int] = 365,
    issued_at: Optional[datetime] = None,
    secret: Optional[bytes] = None,
) -> dict:
    """Produce a signed license payload.

    Run on the maintainer's machine with ``SSA_LICENSE_SECRET`` set in
    the environment. The output is JSON the customer pastes into their
    app via Settings → Licence.

    Args:
        email: Customer email (returned in payload, displayed in UI).
        tier: ``Tier`` enum or string ("pro_solo", "pro_studio", "lifetime").
        features: Optional explicit feature list. If None, the customer
            gets every feature in ``PRO_FEATURES`` (Pro tiers) or nothing
            (Community).
        valid_days: Days from issue until expiration. None = no expiration
            (only valid for Lifetime tier; the helper warns otherwise).
        issued_at: Override the issue timestamp (mostly for tests).
        secret: Override the signing secret (mostly for tests).

    Returns:
        Dict ready to ``json.dumps`` and ship to the customer.
    """
    if isinstance(tier, str):
        tier = Tier(tier)

    if tier == Tier.LIFETIME and valid_days is not None:
        logger.info("Lifetime tier asked with valid_days — overriding to None")
        valid_days = None
    if tier != Tier.LIFETIME and valid_days is None:
        logger.warning(
            "Non-lifetime tier (%s) without expiration — caller intended this?",
            tier.value,
        )

    issued = issued_at or datetime.now(timezone.utc)
    expires = None
    if valid_days is not None:
        from datetime import timedelta
        expires = issued + timedelta(days=valid_days)

    feats = (
        list(features)
        if features is not None
        else (sorted(PRO_FEATURES) if tier != Tier.COMMUNITY else [])
    )

    payload: dict = {
        "email": email,
        "tier": tier.value,
        "features": feats,
        "issued_at": issued.replace(microsecond=0).isoformat(),
        "expires_at": expires.replace(microsecond=0).isoformat() if expires else None,
    }
    payload["signature"] = _sign(payload, secret=secret)
    return payload
