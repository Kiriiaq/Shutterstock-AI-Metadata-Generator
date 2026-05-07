"""WCAG contrast ratios for every fg/bg pair in the v3 palette.

Usage::

    python tools/wcag_check.py

Prints a table with the relative luminance ratio for each foreground /
background pair. AA compliance for normal text requires ≥ 4.5:1; for
large text (≥ 18 pt or ≥ 14 pt bold) and icons, ≥ 3:1 is enough.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow the script to run via ``python tools/wcag_check.py`` from the
# project root without a prior ``pip install``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.theme import DARK, LIGHT  # noqa: E402

# Pairs to verify. Each entry: (label, foreground_key, background_key, role).
# The ``role`` column is the WCAG requirement bucket — ``aa_normal`` needs
# ratio ≥ 4.5, ``aa_large`` (also covers icons / 3 px stroke or wider)
# needs ≥ 3.
PAIRS: list[tuple[str, str, str, str]] = [
    # Body text on the three surface tiers
    ("Body on canvas", "text_primary", "bg_primary", "aa_normal"),
    ("Body on cards", "text_primary", "bg_secondary", "aa_normal"),
    ("Body on inner frames", "text_primary", "bg_elevated", "aa_normal"),
    # Secondary text on the three surface tiers
    ("Secondary on canvas", "text_secondary", "bg_primary", "aa_normal"),
    ("Secondary on cards", "text_secondary", "bg_secondary", "aa_normal"),
    ("Secondary on inner frames", "text_secondary", "bg_elevated", "aa_normal"),
    # Accents (button surfaces) — body-strong text on accent
    ("Button text on accent", "accent_fg", "accent", "aa_normal"),
    ("Button text on accent_hover", "accent_fg", "accent_hover", "aa_normal"),
    ("Button text on accent_secondary", "accent_secondary_fg", "accent_secondary", "aa_normal"),
    ("Button text on accent_secondary_hover", "accent_secondary_fg", "accent_secondary_hover", "aa_normal"),
    # Semantic chips (white-on-color labels)
    ("Success label", "success_fg", "success", "aa_normal"),
    ("Warning label", "warning_fg", "warning", "aa_normal"),
    ("Error label", "error_fg", "error", "aa_normal"),
    # Status dots / icons (large text class is acceptable)
    ("Success dot on canvas", "success", "bg_primary", "aa_large"),
    ("Warning dot on canvas", "warning", "bg_primary", "aa_large"),
    ("Error dot on canvas", "error", "bg_primary", "aa_large"),
    # Borders against canvas (decorative, but should be visible)
    ("Border on canvas", "border", "bg_primary", "aa_large"),
]

ROLE_THRESHOLD = {"aa_normal": 4.5, "aa_large": 3.0}


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _channel_lum(c: float) -> float:
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _relative_luminance(hex_color: str) -> float:
    r, g, b = (_channel_lum(c) for c in _hex_to_rgb(hex_color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(fg: str, bg: str) -> float:
    l1 = _relative_luminance(fg)
    l2 = _relative_luminance(bg)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def _verdict(ratio: float, role: str) -> str:
    threshold = ROLE_THRESHOLD[role]
    if ratio >= 7.0:
        return "AAA"
    if ratio >= threshold:
        return "AA"
    return "FAIL"


def _print_palette(name: str, palette: dict[str, str]) -> int:
    print(f"\n=== {name} ===")
    print(f"{'pair':<42} {'fg':<11} {'bg':<11} {'ratio':>7} {'role':<10} {'verdict':<6}")
    print("-" * 91)
    failures = 0
    for label, fg_key, bg_key, role in PAIRS:
        if fg_key not in palette or bg_key not in palette:
            print(f"{label:<42} {fg_key:<11} {bg_key:<11} {'—':>7}  missing key")
            failures += 1
            continue
        fg, bg = palette[fg_key], palette[bg_key]
        ratio = _contrast_ratio(fg, bg)
        verdict = _verdict(ratio, role)
        if verdict == "FAIL":
            failures += 1
        print(f"{label:<42} {fg:<11} {bg:<11} {ratio:>6.2f}:1 {role:<10} {verdict:<6}")
    print(f"\n{name}: {failures} failure(s).")
    return failures


def main() -> int:
    fail_light = _print_palette("LIGHT", LIGHT)
    fail_dark = _print_palette("DARK", DARK)
    total = fail_light + fail_dark
    print(f"\nTotal WCAG failures across both palettes: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
