"""Seed test/outputs_reference/ from a pristine run_tests.py output.

Idempotent: re-running OVERWRITES the references. Only run when you
explicitly want to refresh the baseline (e.g. after an intentional
behaviour change). For day-to-day regression, use compare_outputs.py.

Usage::

    python test/scripts/_make_reference.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "test" / "outputs_reference"
REELS = ROOT / "test" / "outputs_reels"


def main() -> int:
    # 1. Run the pipeline to populate outputs_reels/
    res = subprocess.run([sys.executable, str(ROOT / "test" / "scripts" / "run_tests.py")],
                         cwd=str(ROOT))
    if res.returncode != 0:
        print(f"run_tests.py exited with {res.returncode}", file=sys.stderr)
        return res.returncode

    # 2. Copy outputs_reels/*.csv to outputs_reference/ref_<name>.csv
    REF.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(REELS.glob("*.csv")):
        stem = src.stem  # e.g. 'reels_adobe'
        if stem.startswith("reels_"):
            dst_name = f"ref_{stem[len('reels_'):]}.csv"
        else:
            dst_name = f"ref_{stem}.csv"
        dst = REF / dst_name
        shutil.copy2(src, dst)
        print(f"  ref: {dst.relative_to(ROOT)}")
        copied += 1
    print(f"Done. {copied} reference file(s) written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
