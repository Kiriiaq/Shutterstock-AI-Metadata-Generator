"""Fulfil StockMeta Pro sales: poll Gumroad, issue licence keys, draft the email.

The paid tier is delivered by hand today: a buyer pays, then waits for a
JSON licence key by email. That delay is the weak point of the funnel —
this script removes the manual steps around it.

What it does, per run:

1. Pulls the sales for the product from the Gumroad CLI.
2. Skips every sale already recorded in the local ledger.
3. Signs a lifetime key for each new buyer (via ``generate_license.py``,
   so there is exactly one signing path in the project).
4. Writes the key to ``keys/<email>.json`` and renders a ready-to-send
   email into ``keys/<email>.email.txt``.
5. Optionally sends it, when SMTP settings are present in the
   environment or ``.env``.

Sending is deliberately opt-in. With no SMTP configured the script still
does everything else and prints what to paste — it never blocks on
credentials, and it never stores them itself.

Usage::

    python tools/fulfill.py --dry-run     # show what would be issued
    python tools/fulfill.py               # issue keys + write drafts
    python tools/fulfill.py --send        # also send them over SMTP

Requires the Gumroad CLI on PATH and an authenticated session
(``gumroad auth status``). The signing secret is read exactly the way
``generate_license.py`` reads it: ``SSA_LICENSE_SECRET`` or ``.env``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import smtplib
import subprocess
import sys
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
KEYS_DIR = REPO_ROOT / "keys"
LEDGER_PATH = KEYS_DIR / "_fulfilled.json"

# The Gumroad product to fulfil. Overridable with --product.
DEFAULT_PRODUCT_ID = "uTDlb807LCxc5mfL5NiFYA=="

EMAIL_SUBJECT = "Your StockMeta Pro licence key"

EMAIL_TEMPLATE = """\
Hi{name_suffix},

Thanks for buying StockMeta Pro. Here is your lifetime licence key.

Copy everything between the lines, including the braces:

------------------------------------------------------------
{key_json}
------------------------------------------------------------

To activate:

  1. Open StockMetaPro.exe (from the zip in your Gumroad download).
  2. In the PARAMETRES panel, bottom right, click "Modifier..."
  3. Scroll to the "Licence" section.
  4. Paste the key into the text box and click "Activer".

The topbar should switch to "Edition Pro - licence a vie". From then on
exports are unlimited, it works offline, and there is nothing to renew.

Keep this email: if you reinstall or change machine, you paste the same
key again. Lost it? Just reply here and I will reissue it.

Thanks again,
Emmanuel
"""


# --------------------------------------------------------------------------
# Gumroad
# --------------------------------------------------------------------------

def _gumroad_bin() -> str:
    """Locate the Gumroad CLI, including the default user install dir."""
    found = shutil.which("gumroad")
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / "gumroad.exe"
    if fallback.exists():
        return str(fallback)
    fallback = Path.home() / ".local" / "bin" / "gumroad"
    if fallback.exists():
        return str(fallback)
    raise SystemExit(
        "Gumroad CLI not found. Install it (https://gumroad.com/install-cli.sh) "
        "and run `gumroad auth login`."
    )


def fetch_sales(product_id: str) -> List[Dict[str, Any]]:
    """Return every sale for the product, newest page first."""
    cmd = [
        _gumroad_bin(), "sales", "list",
        "--product", product_id,
        "--all", "--json", "--no-input", "--non-interactive",
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180
    )
    if proc.returncode != 0:
        raise SystemExit(f"`gumroad sales list` failed:\n{proc.stderr.strip() or proc.stdout.strip()}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse the Gumroad response: {exc}\n{proc.stdout[:400]}") from exc
    return payload.get("sales") or []


def sale_email(sale: Dict[str, Any]) -> Optional[str]:
    """Best-effort buyer email — Gumroad has used several field names."""
    for key in ("email", "purchase_email", "buyer_email"):
        value = sale.get(key)
        if value:
            return str(value).strip()
    return None


def sale_name(sale: Dict[str, Any]) -> str:
    for key in ("full_name", "purchaser_name", "name"):
        value = sale.get(key)
        if value:
            return str(value).strip()
    return ""


# --------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------

def load_ledger() -> Dict[str, Any]:
    if not LEDGER_PATH.exists():
        return {"fulfilled": {}}
    try:
        return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"[warn] ledger unreadable, starting a fresh one: {LEDGER_PATH}", file=sys.stderr)
        return {"fulfilled": {}}


def save_ledger(ledger: Dict[str, Any]) -> None:
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# Licence issuing
# --------------------------------------------------------------------------

def issue_key(email: str) -> str:
    """Sign a lifetime key by delegating to generate_license.py."""
    out_path = KEYS_DIR / f"{email.replace('@', '_at_').replace('/', '_')}.json"
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO_ROOT / "tools" / "generate_license.py"),
        "--email", email, "--tier", "lifetime", "--output", str(out_path),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "generate_license.py failed for "
            f"{email}:\n{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return out_path.read_text(encoding="utf-8").strip()


def render_email(key_json: str, name: str) -> str:
    return EMAIL_TEMPLATE.format(
        name_suffix=f" {name.split()[0]}" if name else "",
        key_json=key_json,
    )


# --------------------------------------------------------------------------
# Optional SMTP delivery
# --------------------------------------------------------------------------

def smtp_settings() -> Optional[Dict[str, str]]:
    """Read SMTP settings from the environment, falling back to .env.

    Returns None when anything required is missing — the caller then
    keeps the draft for manual sending instead of failing.
    """
    values: Dict[str, str] = {}
    dotenv = REPO_ROOT / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()

    required = ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM")
    settings = {k: os.environ.get(k) or values.get(k, "") for k in required}
    settings["SMTP_PORT"] = os.environ.get("SMTP_PORT") or values.get("SMTP_PORT", "587")
    if not all(settings[k] for k in required):
        return None
    return settings


def send_email(settings: Dict[str, str], to_addr: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = EMAIL_SUBJECT
    msg["From"] = settings["SMTP_FROM"]
    msg["To"] = to_addr
    msg.set_content(body)

    with smtplib.SMTP(settings["SMTP_HOST"], int(settings["SMTP_PORT"]), timeout=30) as server:
        server.starttls()
        server.login(settings["SMTP_USER"], settings["SMTP_PASSWORD"])
        server.send_message(msg)


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Issue StockMeta Pro licence keys for new Gumroad sales.")
    parser.add_argument("--product", default=DEFAULT_PRODUCT_ID, help="Gumroad product id.")
    parser.add_argument("--dry-run", action="store_true", help="List what would be issued, change nothing.")
    parser.add_argument("--send", action="store_true", help="Also send the emails over SMTP.")
    parser.add_argument("--reissue", metavar="EMAIL", help="Re-issue for one buyer, ignoring the ledger.")
    args = parser.parse_args()

    if args.reissue:
        key = issue_key(args.reissue)
        body = render_email(key, "")
        draft = KEYS_DIR / f"{args.reissue.replace('@', '_at_')}.email.txt"
        draft.write_text(body, encoding="utf-8")
        print(f"[ok] re-issued for {args.reissue}\n     key   : {KEYS_DIR}\n     draft : {draft}")
        return 0

    sales = fetch_sales(args.product)
    ledger = load_ledger()
    done: Dict[str, Any] = ledger.setdefault("fulfilled", {})

    pending = []
    for sale in sales:
        sale_id = str(sale.get("id") or sale.get("sale_id") or "")
        email = sale_email(sale)
        if not sale_id or not email:
            print(f"[warn] sale without id/email, skipped: {json.dumps(sale)[:120]}", file=sys.stderr)
            continue
        if sale_id in done:
            continue
        pending.append((sale_id, email, sale_name(sale)))

    print(f"{len(sales)} sale(s) total · {len(pending)} awaiting a key")
    if not pending:
        return 0

    if args.dry_run:
        for _, email, name in pending:
            print(f"  would issue -> {email}" + (f" ({name})" if name else ""))
        return 0

    smtp = smtp_settings() if args.send else None
    if args.send and smtp is None:
        print(
            "[warn] --send asked for, but SMTP_HOST / SMTP_USER / SMTP_PASSWORD / SMTP_FROM\n"
            "       are not all set in the environment or .env. Drafts are still written;\n"
            "       send them by hand this time.",
            file=sys.stderr,
        )

    failures = 0
    for sale_id, email, name in pending:
        try:
            key = issue_key(email)
        except RuntimeError as exc:
            print(f"[fail] {email}: {exc}", file=sys.stderr)
            failures += 1
            continue

        body = render_email(key, name)
        draft = KEYS_DIR / f"{email.replace('@', '_at_')}.email.txt"
        draft.write_text(body, encoding="utf-8")

        sent = False
        if smtp is not None:
            try:
                send_email(smtp, email, body)
                sent = True
            except Exception as exc:  # noqa: BLE001 — smtplib raises a zoo of types
                print(f"[warn] could not send to {email} ({exc}); draft kept at {draft}", file=sys.stderr)

        done[sale_id] = {
            "email": email,
            "issued_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "emailed": sent,
        }
        print(f"[ok] {email} — key issued" + (" and sent" if sent else f", draft at {draft}"))

    save_ledger(ledger)

    if failures:
        print(f"\n{failures} sale(s) failed — rerun after fixing, the ledger keeps them pending.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
