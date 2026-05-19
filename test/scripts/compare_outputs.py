"""Diff ``outputs_reels/`` against ``outputs_reference/`` cell by cell.

Exit code:
- 0 when reels match references row-for-row, cell-for-cell.
- 1 when a difference is detected. Prints the diff to stdout and
  writes ``test/outputs_reels/_diff.txt``.

Usage::

    python test/scripts/compare_outputs.py
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "test" / "outputs_reference"
REELS = ROOT / "test" / "outputs_reels"
DIFF = REELS / "_diff.txt"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _diff_rows(name: str, ref_rows: list[dict], rel_rows: list[dict]) -> list[str]:
    out = []
    if len(ref_rows) != len(rel_rows):
        out.append(f"[{name}] row count differs: ref={len(ref_rows)} reels={len(rel_rows)}")
        return out
    # Match by Filename column when present, else by index.
    ref_by_key = {r.get("Filename", str(i)): r for i, r in enumerate(ref_rows)}
    for i, rel in enumerate(rel_rows):
        key = rel.get("Filename", str(i))
        ref = ref_by_key.get(key)
        if ref is None:
            out.append(f"[{name}] row missing in ref: {key}")
            continue
        for col in ref.keys():
            a, b = ref.get(col, ""), rel.get(col, "")
            if a != b:
                out.append(f"[{name}] {key} · column '{col}'\n  ref:  {a!r}\n  reel: {b!r}")
    return out


def main() -> int:
    if not REF.exists() or not any(REF.iterdir()):
        logger.error("Reference dir empty: %s", REF)
        logger.error("Run scripts/_make_reference.py first to seed references.")
        return 2
    if not REELS.exists() or not any(REELS.iterdir()):
        logger.error("Reels dir empty: %s — run run_tests.py first", REELS)
        return 2

    diffs = []
    for ref_file in sorted(REF.glob("*.csv")):
        # Match reference 'ref_<name>.csv' with reels 'reels_<name>.csv'.
        stem = ref_file.stem
        if stem.startswith("ref_"):
            counterpart = REELS / f"reels_{stem[4:]}.csv"
        else:
            counterpart = REELS / ref_file.name
        ref_rows = _load_csv(ref_file)
        rel_rows = _load_csv(counterpart)
        if not rel_rows:
            diffs.append(f"[{ref_file.name}] no counterpart in reels ({counterpart.name})")
            continue
        diffs.extend(_diff_rows(ref_file.name, ref_rows, rel_rows))

    if diffs:
        text = "\n".join(diffs)
        DIFF.write_text(text, encoding="utf-8")
        print(text)
        logger.error("DIFF: %d delta(s) — see %s", len(diffs), DIFF.relative_to(ROOT))
        return 1

    if DIFF.exists():
        DIFF.unlink()
    logger.info("OK — outputs_reels match outputs_reference cell-for-cell")
    return 0


if __name__ == "__main__":
    sys.exit(main())
