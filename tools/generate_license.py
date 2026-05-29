"""Admin-side license key generator.

Run on the maintainer's machine ONLY. Reads the signing secret from
the ``SSA_LICENSE_SECRET`` environment variable, falls back to the
dev placeholder if absent.

Usage examples
--------------

    # Lifetime key — the only paid tier (10 € one-shot, never expires)
    python tools\\generate_license.py --email alice@example.com --tier lifetime

    # Write to a file instead of stdout
    python tools\\generate_license.py --email eve@example.com --tier lifetime \
        --output keys\\eve.json

The output is the JSON payload to paste into the customer's app via
Settings → Licence. Send it as a plain-text email attachment or via
the Gumroad post-purchase email.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make the repo root importable when launched from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules.licensing import Tier, generate_license_key  # noqa: E402


def _read_dot_env_secret() -> str | None:
    """Look for SSA_LICENSE_SECRET in the project root .env file."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("SSA_LICENSE_SECRET=") and not line.startswith("#"):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value and value != "replace-with-secrets-token_urlsafe-32-output":
                return value
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a signed license key for ShutterstockAnalyzer Pro.",
    )
    parser.add_argument("--email", required=True, help="Customer email (displayed in app).")
    parser.add_argument(
        "--tier",
        required=True,
        choices=[t.value for t in Tier if t != Tier.COMMUNITY],
        help="Subscription tier.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Days until expiration. Default: none (the lifetime tier never expires).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Write the JSON to this file. Default: stdout.",
    )
    parser.add_argument(
        "--allow-dev-secret",
        action="store_true",
        help=(
            "Allow signing with the bundled DEV secret. Required for issuing "
            "non-prod test keys when SSA_LICENSE_SECRET is not configured. "
            "DO NOT use for keys you actually sell to customers — they'll be "
            "trivially forgeable by anyone reading the source code."
        ),
    )
    args = parser.parse_args()

    # If env var isn't set, fall back to .env file at repo root (same
    # source the runtime uses). If neither is set, we refuse to sign
    # unless --allow-dev-secret is explicitly passed — better to fail
    # loudly than ship sellable keys signed with a public secret.
    env_secret = os.environ.get("SSA_LICENSE_SECRET", "").strip() or _read_dot_env_secret()
    if env_secret:
        os.environ["SSA_LICENSE_SECRET"] = env_secret
        masked = env_secret[:6] + "…" + env_secret[-3:] if len(env_secret) > 10 else "***"
        print(f"[ok] Using production secret ({masked}).", file=sys.stderr)
    elif args.allow_dev_secret:
        print(
            "[warn] --allow-dev-secret enabled — signing with the bundled DEV\n"
            "       value. Keys generated are FOR TESTING ONLY. They WILL be\n"
            "       forgeable by anyone reading the source.",
            file=sys.stderr,
        )
    else:
        print(
            "[error] No production secret found.\n"
            "        Either:\n"
            "        - set SSA_LICENSE_SECRET in your environment, OR\n"
            "        - put SSA_LICENSE_SECRET=... in a .env file at repo root, OR\n"
            "        - pass --allow-dev-secret (test keys only).\n"
            "\n"
            "        Generate a secret with:\n"
            "        python -c \"import secrets; print(secrets.token_urlsafe(32))\"",
            file=sys.stderr,
        )
        return 2

    payload = generate_license_key(
        email=args.email,
        tier=args.tier,
        valid_days=args.days,
    )

    text = json.dumps(payload, indent=2, ensure_ascii=False)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"[ok] License written to {args.output}", file=sys.stderr)
        print(f"     email   = {args.email}", file=sys.stderr)
        print(f"     tier    = {args.tier}", file=sys.stderr)
        print(f"     expires = {payload.get('expires_at') or 'never'}", file=sys.stderr)
    else:
        print(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
