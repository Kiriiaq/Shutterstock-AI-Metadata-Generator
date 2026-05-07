"""Design system — palette, typography, spacing, radii, theme persistence.

Single source of truth for every visible style decision. Two palettes
(LIGHT, DARK) cover the same semantic names; ``get_color(name)`` reads
``customtkinter.get_appearance_mode()`` on every call so a runtime theme
switch picks up the new value without a restart.

Contrast budget: body text vs default surface ≥ 4.5:1 in both palettes
(verified against WCAG AA via the chosen slate / blue / red ramps).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from tkinter import font as tkfont
from typing import Final, Literal

import customtkinter as ctk

logger = logging.getLogger(__name__)

ThemeMode = Literal["light", "dark", "system"]

# ============================================================================
# Palettes
# ----------------------------------------------------------------------------
# Naming convention:
#   bg / bg_elevated / bg_hover / bg_active     surfaces, ascending elevation
#   fg / fg_muted / fg_subtle                   text, descending emphasis
#   border / border_strong                      separators
#   accent / accent_hover / accent_fg           primary action
#   success / warning / error / info  + _fg + _bg  semantic states (badge bg + text)
#   focus_ring                                  keyboard-focus halo (≥ 2px)
# ============================================================================

LIGHT: Final[dict[str, str]] = {
    # Surfaces — gray canvas + white cards, à la macOS / Notion / Linear.
    # Less stark than pure white, easier on the eyes, and the cards on
    # top stand out as crisp white rectangles instead of disappearing
    # into a uniform field.
    "bg": "#F1F5F9",  # slate-100 — clearly gray, the canvas
    "bg_elevated": "#FFFFFF",  # white — panels / cards stand out
    "bg_deep": "#FFFFFF",  # = bg_elevated in light mode (no visible change)
    "bg_hover": "#E2E8F0",  # slate-200
    "bg_active": "#CBD5E1",  # slate-300
    # Text — near-black for the strongest contrast on slate-100 (~16:1).
    "fg": "#0F172A",  # slate-900
    "fg_muted": "#475569",  # slate-600
    "fg_subtle": "#94A3B8",  # slate-400
    # Borders — slightly stronger than before so cards register clearly
    # against the gray canvas.
    "border": "#CBD5E1",  # slate-300
    "border_strong": "#94A3B8",  # slate-400
    # Accent (primary action)
    "accent": "#2563EB",  # blue-600
    "accent_hover": "#1D4ED8",  # blue-700
    "accent_fg": "#FFFFFF",
    # Semantics
    "success": "#15803D",  # green-700
    "success_fg": "#FFFFFF",
    "success_bg": "#DCFCE7",  # green-100
    "warning": "#B45309",  # amber-700 — warmer, more legible on white than orange-700
    "warning_fg": "#FFFFFF",
    "warning_bg": "#FEF3C7",  # amber-100
    "error": "#B91C1C",  # red-700
    "error_fg": "#FFFFFF",
    "error_bg": "#FEE2E2",  # red-100
    "info": "#0E7490",  # cyan-700
    "info_fg": "#FFFFFF",
    "info_bg": "#CFFAFE",  # cyan-100
    # Focus
    "focus_ring": "#3B82F6",
}

DARK: Final[dict[str, str]] = {
    "bg": "#0F172A",  # slate-900 — canvas
    "bg_elevated": "#1E293B",  # slate-800 — cards/panels (slightly lighter than canvas)
    "bg_deep": "#020617",  # slate-950 — for the Sources panel: deeper than canvas, "workspace floor"
    "bg_hover": "#334155",
    "bg_active": "#475569",
    "fg": "#F1F5F9",
    "fg_muted": "#94A3B8",
    "fg_subtle": "#64748B",
    "border": "#334155",
    "border_strong": "#475569",
    "accent": "#3B82F6",
    "accent_hover": "#60A5FA",
    "accent_fg": "#0B1220",
    "success": "#22C55E",
    "success_fg": "#052E16",
    "success_bg": "#14532D",
    "warning": "#F97316",
    "warning_fg": "#1C0701",
    "warning_bg": "#7C2D12",
    "error": "#EF4444",
    "error_fg": "#2A0808",
    "error_bg": "#7F1D1D",
    "info": "#06B6D4",
    "info_fg": "#042F36",
    "info_bg": "#164E63",
    "focus_ring": "#60A5FA",
}


def get_color(name: str) -> str:
    """Return the colour bound to *name* in the active palette.

    Reads ``customtkinter.get_appearance_mode()`` on every call so widgets
    can re-fetch their colours after a theme switch without restart.

    Raises ``KeyError`` if *name* is unknown — silent failure on a typo
    here would be much harder to debug than a loud crash.
    """
    mode = ctk.get_appearance_mode().lower()  # 'light' | 'dark'
    palette = DARK if mode == "dark" else LIGHT
    return palette[name]


def palette_pair(name: str) -> tuple[str, str]:
    """Return ``(light, dark)`` — useful for CTk widgets that accept tuples."""
    return LIGHT[name], DARK[name]


# ============================================================================
# Spacing scale (px)
# ----------------------------------------------------------------------------

SPACE_XS: Final[int] = 4
SPACE_SM: Final[int] = 8
SPACE_MD: Final[int] = 12
SPACE_LG: Final[int] = 16
SPACE_XL: Final[int] = 24
SPACE_XXL: Final[int] = 32
SPACE_HUGE: Final[int] = 48

# ============================================================================
# Radius scale (px)
# ----------------------------------------------------------------------------

RADIUS_SM: Final[int] = 4  # chips, badges, small status pills
RADIUS_MD: Final[int] = 6  # buttons, inputs
RADIUS_LG: Final[int] = 8  # cards, panels, modals

# ============================================================================
# Typography
# ----------------------------------------------------------------------------

_PROPORTIONAL_CANDIDATES: Final[tuple[str, ...]] = (
    "Segoe UI",  # Windows 10/11
    "SF Pro Text",  # macOS
    "Inter",  # cross-platform if installed
    "DejaVu Sans",  # Linux fallback
)

_MONOSPACE_CANDIDATES: Final[tuple[str, ...]] = (
    "Cascadia Mono",
    "Consolas",
    "SF Mono",
    "DejaVu Sans Mono",
)

# Role -> (size_px, weight). Weight 'bold' is used for medium emphasis too
# because Tk only exposes 'normal' / 'bold' reliably across platforms.
_FONT_ROLES: Final[dict[str, tuple[int, str]]] = {
    "small": (11, "normal"),
    "body": (13, "normal"),
    "body_strong": (14, "bold"),
    "h3": (16, "bold"),
    "h2": (20, "bold"),
    "h1": (24, "bold"),
    "code": (12, "normal"),
}

_resolved: dict[str, str] = {}  # cache: 'proportional' | 'monospace' -> family


def _pick_first_available(candidates: tuple[str, ...], cache_key: str) -> str:
    """Return the first installed family from *candidates*, fallback to TkDefault."""
    if cache_key in _resolved:
        return _resolved[cache_key]
    available = set(tkfont.families())
    for cand in candidates:
        if cand in available:
            _resolved[cache_key] = cand
            logger.info("UI %s font resolved to: %s", cache_key, cand)
            return cand
    fallback = tkfont.nametofont("TkDefaultFont").actual("family")
    _resolved[cache_key] = fallback
    logger.warning("No preferred %s font found; falling back to %s", cache_key, fallback)
    return fallback


def get_font(role: str = "body") -> ctk.CTkFont:
    """Return a ``CTkFont`` tuned for *role*.

    Roles: ``small``, ``body`` (default), ``body_strong``, ``h3``, ``h2``,
    ``h1``, ``code``. ``code`` uses a monospace family.
    """
    if role not in _FONT_ROLES:
        raise KeyError(f"Unknown font role: {role!r} (expected one of {sorted(_FONT_ROLES)})")
    size, weight = _FONT_ROLES[role]
    if role == "code":
        family = _pick_first_available(_MONOSPACE_CANDIDATES, "monospace")
    else:
        family = _pick_first_available(_PROPORTIONAL_CANDIDATES, "proportional")
    return ctk.CTkFont(family=family, size=size, weight=weight)


# ============================================================================
# Theme persistence
# ----------------------------------------------------------------------------

_VALID_THEMES: Final[frozenset[str]] = frozenset({"light", "dark", "system"})
# Default to "light" rather than "system": on Windows "System" resolves to
# whatever the OS-level personalization setting says, which is dark on most
# machines by default. A document-editing tool reads better in light mode at
# rest, so we make that the explicit out-of-the-box choice. Users who flip
# to dark or system have their preference persisted to ui_prefs.json.
_DEFAULT_PREFS: Final[dict[str, str]] = {"theme": "light"}


def get_prefs_path() -> Path:
    """Per-user prefs file.

    - Windows: ``%APPDATA%/ShutterstockAnalyzer/ui_prefs.json``
    - Other:   ``~/.shutterstock_analyzer/ui_prefs.json``

    Creates the parent directory if needed.
    """
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", str(Path.home()))) / "ShutterstockAnalyzer"
    else:
        base = Path.home() / ".shutterstock_analyzer"
    base.mkdir(parents=True, exist_ok=True)
    return base / "ui_prefs.json"


def load_theme_pref() -> ThemeMode:
    """Read the persisted theme choice. Returns ``'system'`` on missing/corrupt."""
    path = get_prefs_path()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return "system"
    theme = data.get("theme", "system")
    return theme if theme in _VALID_THEMES else "system"


def save_theme_pref(theme: ThemeMode) -> None:
    """Persist *theme* to the prefs file. Logs and swallows I/O errors."""
    if theme not in _VALID_THEMES:
        raise ValueError(f"Invalid theme: {theme!r} (expected one of {sorted(_VALID_THEMES)})")
    path = get_prefs_path()
    try:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = dict(_DEFAULT_PREFS)
        data["theme"] = theme
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logger.warning("Could not save theme pref: %s", e)


def apply_theme(theme: ThemeMode) -> None:
    """Apply *theme* to customtkinter and persist the choice."""
    if theme not in _VALID_THEMES:
        raise ValueError(f"Invalid theme: {theme!r}")
    ctk.set_appearance_mode("System" if theme == "system" else theme.capitalize())
    save_theme_pref(theme)


def toggle_theme() -> ThemeMode:
    """Flip the active theme between ``light`` and ``dark``.

    If currently ``system``, the actually-resolved appearance is read first
    so the toggle behaves intuitively (toggles away from whatever is shown).
    Returns the new theme.
    """
    current = load_theme_pref()
    if current == "system":
        # 'System' resolves to 'Light' or 'Dark' at runtime — flip from there.
        resolved = ctk.get_appearance_mode().lower()
        new = "dark" if resolved == "light" else "light"
    else:
        new = "dark" if current == "light" else "light"
    apply_theme(new)
    return new
