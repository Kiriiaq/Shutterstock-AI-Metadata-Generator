"""Smoke test the freshly built EXEs.

Starts each EXE from a clean tempdir, waits ~6s, force-kills the whole
process tree via ``taskkill /F /T /PID``, and reports whether the
process was alive (== didn't crash) at the kill point.

The Windows-specific kill is needed because PyInstaller's onefile EXE
spawns a child process that holds stdout open; ``proc.terminate()``
only kills the bootloader, leaving the child alive (and the parent's
stdout pipe blocked on read).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _kill_tree(pid: int) -> None:
    """Force-kill *pid* and every descendant. Windows-only."""
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _smoke(label: str, exe: Path, hold_seconds: float = 6.0) -> dict:
    if not exe.exists():
        return {"label": label, "status": "MISSING", "startup_s": 0.0, "size_mb": 0.0, "tail": ""}

    size_mb = exe.stat().st_size / (1024 * 1024)
    with tempfile.TemporaryDirectory(prefix=f"sa_smoke_{label}_") as tmp:
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        env.pop("PYTHONPATH", None)
        log_path = Path(tmp) / "smoke.log"
        t0 = time.time()
        with log_path.open("wb") as logf:
            proc = subprocess.Popen(
                [str(exe)],
                cwd=tmp,
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=env,
            )
        # Hold the EXE alive for `hold_seconds` so it has time to render
        # the window and reach mainloop, then nuke the process tree so
        # we don't leak windows to the desktop.
        deadline = t0 + hold_seconds
        crashed_early = False
        while time.time() < deadline:
            if proc.poll() is not None:
                crashed_early = True
                break
            time.sleep(0.25)
        startup = time.time() - t0

        if crashed_early:
            status = f"CRASHED rc={proc.returncode}"
        else:
            _kill_tree(proc.pid)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            status = "ALIVE"

        try:
            tail = log_path.read_text("utf-8", errors="replace").splitlines()[-12:]
        except Exception:
            tail = []
    return {
        "label": label,
        "status": status,
        "startup_s": startup,
        "size_mb": size_mb,
        "tail": "\n".join(tail),
    }


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    cases = [
        ("debug", repo / "dist" / "ShutterstockAnalyzer-debug.exe"),
        ("release", repo / "dist" / "ShutterstockAnalyzer.exe"),
    ]
    any_failed = False
    for label, exe in cases:
        r = _smoke(label, exe)
        ok = r["status"] == "ALIVE"
        any_failed = any_failed or not ok
        print(f"=== {label.upper()} ===")
        print(f"  exe       : {exe.name}")
        print(f"  size      : {r['size_mb']:.1f} MB")
        print(f"  status    : {r['status']}")
        print(f"  hold_time : {r['startup_s']:.2f} s")
        if r["tail"]:
            print("  log tail  :")
            for line in r["tail"].splitlines()[-6:]:
                print(f"    {line}")
        print()
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
