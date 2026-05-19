"""Admin-side license key generator.

Run on the maintainer's machine ONLY. Reads the signing secret from
the ``SSA_LICENSE_SECRET`` environment variable, falls back to the
dev placeholder if absent.

Usage examples
--------------

    # Pro Solo 1 year — default
    python tools\\generate_license.py --email alice@example.com --tier pro_solo

    # Pro Studio (multi-poste) 1 year
    python tools\\generate_license.py --email bob@studio.com --tier pro_studio

    # Lifetime, no expiration
    python tools\\generate_license.py --email carol@example.com --tier lifetime

    # Override expiration (in days)
    python tools\\generate_license.py --email dave@example.com --tier pro_solo --days 30

    # Write to file (default = stdout)
    python tools\\generate_license.py --email eve@example.com --tier pro_solo \
        --output keys\\eve_2026.json

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
        help="Days until expiration. Default: 365 for pro_solo/pro_studio, none for lifetime.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Write the JSON to this file. Default: stdout.",
    )
    args = parser.parse_args()

    if os.environ.get("SSA_LICENSE_SECRET") is None:
        print(
            "[warn] SSA_LICENSE_SECRET env var not set — using the bundled dev secret.\n"
            "       Keys signed this way ARE compatible with the current EXE\n"
            "       but anyone reading the source can forge keys. For production,\n"
            "       set SSA_LICENSE_SECRET to a random secret and re-build the EXE.",
            file=sys.stderr,
        )

    # Default validity by tier
    if args.days is None and args.tier in {"pro_solo", "pro_studio"}:
        args.days = 365

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
