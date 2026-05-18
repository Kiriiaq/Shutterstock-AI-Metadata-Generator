"""Helpers for subprocess calls.

Sole purpose: provide ``SUBPROCESS_NO_WINDOW`` — a kwargs dict to splat
into ``subprocess.run`` / ``Popen`` to prevent a console window from
flashing up every time an external binary (ExifTool, ffmpeg, …) is
invoked on Windows. No-op on POSIX.

Phase G (2026-05-18) — added to fix the "exiftool.exe console window
flashes on every image scan" UX nuisance.

Usage::

    from src.utils.subprocess_helper import SUBPROCESS_NO_WINDOW

    subprocess.run(
        ["exiftool", "-json", "image.jpg"],
        capture_output=True,
        text=True,
        timeout=10,
        **SUBPROCESS_NO_WINDOW,
    )
"""

from __future__ import annotations

import subprocess
import sys

# Note: ``subprocess.CREATE_NO_WINDOW`` is defined on Windows only.
# On POSIX, we expose an empty dict so the splat is a no-op.
SUBPROCESS_NO_WINDOW: dict = (
    {"creationflags": subprocess.CREATE_NO_WINDOW}  # type: ignore[attr-defined]
    if sys.platform == "win32"
    else {}
)
