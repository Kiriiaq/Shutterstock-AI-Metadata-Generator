"""Design system — gray-blue palettes, ThemeManager, Themeable mixin.

Single source of truth for every visible style decision. Two
palettes (``LIGHT_THEME``, ``DARK_THEME``) cover the 13 semantic keys
specified in the v3 theme spec; legacy keys are kept as aliases so the
~256 existing ``palette_pair("X")`` call sites keep working unchanged.

Two refresh mechanisms coexist:

1. **CTk-native auto-switch via tuples** — ``palette_pair(key)`` returns
   ``(LIGHT[key], DARK[key])``. CTk picks one based on the active
   appearance mode and re-picks on ``set_appearance_mode(...)``. Most
   widgets in app/ use this and need no explicit refresh.

2. **Explicit observer pattern via ThemeManager** — for widgets whose
   colors depend on runtime state (status indicators, ttk.Treeview,
   custom Toplevels), inherit ``Themeable``, override ``apply_theme()``,
   and call ``register_themeable(self)`` at the end of ``__init__``.
   The ThemeManager dispatches ``notify_all()`` on every theme switch,
   which calls ``apply_theme()`` on every registered widget.

Both mechanisms see the same palette dicts. Switching themes dispatches
both: CTk's set_appearance_mode for tuple-using widgets, then the
ThemeManager observer chain for explicit ones.
"""

from __future__ import annotations

import abc
import json
import logging
import os
import weakref
from collections.abc import Callable
from pathlib import Path
from tkinter import font as tkfont
from typing import Final, Literal

import customtkinter as ctk

logger = logging.getLogger(__name__)

ThemeMode = Literal["light", "dark", "system"]

# ============================================================================
# Palettes (13 semantic keys, per the v3 theme spec)
# ----------------------------------------------------------------------------
# Light: gray-blue desaturated, no pure white. Cards (bg_secondary) are
# slightly RECESSED from the canvas (bg_primary); inner frames
# (bg_elevated) are RAISED ABOVE cards. Three-tier depth.
#
# Dark: anthracite gray-blue, no pure black. Cards are RAISED above the
# canvas; inner frames raised further. Same conceptual hierarchy,
# inverse direction.
# ============================================================================

LIGHT_THEME: Final[dict[str, str]] = {
    # Light theme softened (no pure white) — three neutral-gray tiers,
    # picked from the user-suggested palette (#F5F5F7 / #ECEEF1 /
    # #E8EAED). Less near-white than the previous near-#F5F7FA, less
    # blue cast than the older #EEF1F5 — reads as a calm soft gray
    # while keeping enough contrast for borders and text.
    "bg_primary": "#ECEEF1",  # canvas — base surface
    "bg_secondary": "#E8EAED",  # recessed (cards under canvas)
    "bg_elevated": "#F5F5F7",  # raised (inner frames on cards)
    # Border darkened from #C5CDD9 to give cards a sharper, more
    # readable edge on the gray-blue canvas (per user feedback —
    # "rajoute plus de contraste").
    "border": "#A8B2C0",
    "text_primary": "#1E2733",
    "text_secondary": "#4A5566",
    "accent": "#3B6EA5",
    "accent_hover": "#2E5A8A",
    "accent_secondary": "#6B7A8F",
    "accent_secondary_hover": "#566374",
    "success": "#2E7D5B",
    "error": "#B5413B",
    "warning": "#B8862E",
}

DARK_THEME: Final[dict[str, str]] = {
    "bg_primary": "#1B222C",
    "bg_secondary": "#252D3A",
    "bg_elevated": "#2F3847",
    "border": "#3D4759",
    "text_primary": "#E4E8EF",
    "text_secondary": "#A0AAB8",
    "accent": "#5B8FC9",
    "accent_hover": "#7BA5D8",
    "accent_secondary": "#7A8699",
    "accent_secondary_hover": "#8E99AB",
    "success": "#4FA980",
    "error": "#D9655F",
    "warning": "#D9A856",
}

# ============================================================================
# Legacy aliases (so the existing 256 palette_pair calls keep working).
# ----------------------------------------------------------------------------
# Old code expected ``bg_elevated`` to mean "card surface". In the new
# spec, "card surface" is ``bg_secondary`` and ``bg_elevated`` is an
# even higher inner surface. To preserve visual continuity for existing
# widgets, the legacy ``bg_elevated`` key stays mapped to
# ``bg_secondary``. New code should use the new key names.
# ============================================================================

_LEGACY_LIGHT: Final[dict[str, str]] = {
    "bg": LIGHT_THEME["bg_primary"],
    # Hover surface — sits between bg_secondary (#E8EAED) and the
    # border (#A8B2C0), giving a clear pressed/hover feedback without
    # dropping all the way to a near-border value.
    "bg_hover": "#D8DCE2",
    "bg_active": LIGHT_THEME["border"],
    "bg_deep": LIGHT_THEME["bg_secondary"],  # Sources panel — same as cards in light
    "fg": LIGHT_THEME["text_primary"],
    "fg_muted": LIGHT_THEME["text_secondary"],
    "fg_subtle": "#7B8696",
    # Stronger border for sharper card edges on the gray-blue canvas.
    "border_strong": "#94A0AE",
    "accent_fg": "#FFFFFF",
    "accent_secondary_fg": "#FFFFFF",
    "success_fg": "#FFFFFF",
    "success_bg": "#D6EEDF",
    "warning_fg": LIGHT_THEME["text_primary"],  # dark text on yellow — 4.96:1 (white was 3.24:1)
    "warning_bg": "#F5E6C8",
    "error_fg": "#FFFFFF",
    "error_bg": "#F5DEDB",
    "info": LIGHT_THEME["accent"],
    "info_fg": "#FFFFFF",
    "info_bg": "#D9E4F0",
    "focus_ring": LIGHT_THEME["accent"],
}

_LEGACY_DARK: Final[dict[str, str]] = {
    "bg": DARK_THEME["bg_primary"],
    "bg_hover": "#3A4458",
    "bg_active": DARK_THEME["border"],
    "bg_deep": "#0F141B",  # deeper than canvas — Sources panel
    "fg": DARK_THEME["text_primary"],
    "fg_muted": DARK_THEME["text_secondary"],
    "fg_subtle": "#6F7B8B",
    "border_strong": "#54607A",
    "accent_fg": DARK_THEME["bg_primary"],
    "accent_secondary_fg": DARK_THEME["bg_primary"],
    "success_fg": DARK_THEME["bg_primary"],
    "success_bg": "#1F3A2C",
    "warning_fg": DARK_THEME["bg_primary"],
    "warning_bg": "#3D2F1A",
    "error_fg": DARK_THEME["bg_primary"],
    "error_bg": "#3F221F",
    "info": DARK_THEME["accent"],
    "info_fg": DARK_THEME["bg_primary"],
    "info_bg": "#1A2632",
    "focus_ring": DARK_THEME["accent"],
}

# Composite palettes consumed by widgets. New keys + legacy aliases coexist.
LIGHT: Final[dict[str, str]] = {**LIGHT_THEME, **_LEGACY_LIGHT}
DARK: Final[dict[str, str]] = {**DARK_THEME, **_LEGACY_DARK}


# ============================================================================
# Color access (string or tuple form)
# ----------------------------------------------------------------------------


def get_color(name: str) -> str:
    """Return the active-theme color for *name* as a single hex string.

    Suitable for ttk.Style and other consumers that don't understand the
    CTk tuple-form. CTk widgets should prefer ``palette_pair`` so they
    auto-bascule on ``set_appearance_mode``.
    """
    mode = ctk.get_appearance_mode().lower()  # 'light' | 'dark'
    palette = DARK if mode == "dark" else LIGHT
    return palette[name]


def palette_pair(name: str) -> tuple[str, str]:
    """Return ``(light, dark)`` — the CTk-native form. Widgets initialised
    with these tuples auto-bascule when ``set_appearance_mode`` is called,
    no explicit refresh needed.
    """
    return LIGHT[name], DARK[name]


# ============================================================================
# Spacing / radius / typography (unchanged from prior revision)
# ----------------------------------------------------------------------------

SPACE_XS: Final[int] = 4
SPACE_SM: Final[int] = 8
SPACE_MD: Final[int] = 12
SPACE_LG: Final[int] = 16
SPACE_XL: Final[int] = 24
SPACE_XXL: Final[int] = 32
SPACE_HUGE: Final[int] = 48

RADIUS_SM: Final[int] = 4
RADIUS_MD: Final[int] = 6
RADIUS_LG: Final[int] = 8

_PROPORTIONAL_CANDIDATES: Final[tuple[str, ...]] = ("Segoe UI", "SF Pro Text", "Inter", "DejaVu Sans")
_MONOSPACE_CANDIDATES: Final[tuple[str, ...]] = ("Cascadia Mono", "Consolas", "SF Mono", "DejaVu Sans Mono")
_FONT_ROLES: Final[dict[str, tuple[int, str]]] = {
    "small": (11, "normal"),
    "body": (13, "normal"),
    "body_strong": (14, "bold"),
    "h3": (16, "bold"),
    "h2": (20, "bold"),
    "h1": (24, "bold"),
    "code": (12, "normal"),
}
_resolved: dict[str, str] = {}


def _pick_first_available(candidates: tuple[str, ...], cache_key: str) -> str:
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
    if role not in _FONT_ROLES:
        raise KeyError(f"Unknown font role: {role!r} (expected one of {sorted(_FONT_ROLES)})")
    size, weight = _FONT_ROLES[role]
    family = (
        _pick_first_available(_MONOSPACE_CANDIDATES, "monospace")
        if role == "code"
        else _pick_first_available(_PROPORTIONAL_CANDIDATES, "proportional")
    )
    return ctk.CTkFont(family=family, size=size, weight=weight)


# ============================================================================
# Persistence
# ----------------------------------------------------------------------------

_VALID_THEMES: Final[frozenset[str]] = frozenset({"light", "dark", "system"})
_DEFAULT_PREFS: Final[dict[str, str]] = {"theme": "light"}


def get_prefs_path() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", str(Path.home()))) / "ShutterstockAnalyzer"
    else:
        base = Path.home() / ".shutterstock_analyzer"
    base.mkdir(parents=True, exist_ok=True)
    return base / "ui_prefs.json"


def load_theme_pref() -> ThemeMode:
    path = get_prefs_path()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return "light"
    theme = data.get("theme", "light")
    return theme if theme in _VALID_THEMES else "light"


def save_theme_pref(theme: ThemeMode) -> None:
    if theme not in _VALID_THEMES:
        raise ValueError(f"Invalid theme: {theme!r}")
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


# ============================================================================
# Themeable + ThemeManager
# ----------------------------------------------------------------------------


class Themeable(abc.ABC):
    """Mixin / ABC for widgets that need explicit theme refresh.

    Inherit alongside ``ctk.CTkFrame`` (or any tk widget), define
    ``apply_theme()`` to re-pull colors from ``ThemeManager`` and call
    ``configure(...)``, and call ``register_themeable(self)`` at the
    end of ``__init__`` so the manager wires the auto-unregister on
    ``<Destroy>``.

    Widgets that don't need explicit refresh (typical CTk widgets that
    only consume tuple colors) can ignore this — they auto-bascule via
    CTk's ``set_appearance_mode``.
    """

    @abc.abstractmethod
    def apply_theme(self) -> None:
        """Re-pull colors from ``ThemeManager.get(...)`` and apply via configure."""


class ThemeManager:
    """Singleton owning the active theme + observer registry."""

    _instance: "ThemeManager | None" = None

    @classmethod
    def get_instance(cls) -> "ThemeManager":
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance._initialized = False
        if not cls._instance._initialized:
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self._mode: ThemeMode = load_theme_pref()
        # WeakSet so destroyed widgets don't leak even if unregister was missed.
        self._observers: weakref.WeakSet[Themeable] = weakref.WeakSet()
        # Optional callbacks fired alongside observers (e.g. ttk.Style refresh).
        self._global_hooks: list[Callable[[], None]] = []
        self._initialized = True
        ctk.set_appearance_mode("System" if self._mode == "system" else self._mode.capitalize())

    # ------------------------------------------------------------------ public API

    @property
    def mode(self) -> ThemeMode:
        return self._mode

    @property
    def effective_mode(self) -> str:
        if self._mode == "system":
            return ctk.get_appearance_mode().lower()
        return self._mode

    def get(self, key: str) -> str:
        """Return the active-theme color for *key* as a hex string."""
        palette = DARK if self.effective_mode == "dark" else LIGHT
        return palette[key]

    def set_mode(self, mode: ThemeMode) -> None:
        """Change the active theme, persist it, and notify all observers."""
        if mode not in _VALID_THEMES:
            raise ValueError(f"Invalid mode: {mode!r}")
        self._mode = mode
        save_theme_pref(mode)
        ctk.set_appearance_mode("System" if mode == "system" else mode.capitalize())
        self.notify_all()

    def toggle(self) -> ThemeMode:
        """Cycle light → dark → system → light."""
        nxt: dict[ThemeMode, ThemeMode] = {"light": "dark", "dark": "system", "system": "light"}
        new = nxt[self._mode]
        self.set_mode(new)
        return new

    def register(self, widget: Themeable) -> None:
        self._observers.add(widget)

    def unregister(self, widget: Themeable) -> None:
        self._observers.discard(widget)

    def add_global_hook(self, fn: Callable[[], None]) -> None:
        """Register a non-widget callback fired on every theme change.

        Useful for ``ttk.Style`` reconfiguration, theme-dependent file
        regeneration, etc.
        """
        if fn not in self._global_hooks:
            self._global_hooks.append(fn)

    def notify_all(self) -> None:
        """Call ``apply_theme()`` on every observer + every global hook.

        Errors are logged so one bad observer doesn't break the chain.
        """
        for widget in list(self._observers):
            try:
                widget.apply_theme()
            except Exception:
                logger.exception("apply_theme failed on %r", widget)
        for hook in list(self._global_hooks):
            try:
                hook()
            except Exception:
                logger.exception("Global theme hook failed: %r", hook)


def register_themeable(widget: Themeable) -> None:
    """Register *widget* with the ThemeManager and wire auto-unregister.

    Call this at the end of the widget's ``__init__`` after all child
    widgets have been built. Performs an initial ``apply_theme()`` so
    the widget shows the active theme without the caller having to.
    """
    tm = ThemeManager.get_instance()
    tm.register(widget)
    if hasattr(widget, "bind"):
        try:
            widget.bind(  # type: ignore[attr-defined]
                "<Destroy>",
                lambda _e: tm.unregister(widget),
                add="+",
            )
        except Exception:
            logger.exception("Could not bind <Destroy> on %r", widget)
    try:
        widget.apply_theme()
    except Exception:
        logger.exception("Initial apply_theme failed on %r", widget)


# ============================================================================
# Backward-compatible top-level helpers
# ----------------------------------------------------------------------------


def apply_theme(theme: ThemeMode) -> None:
    """Set the active theme and persist. Convenience wrapper around
    ``ThemeManager.set_mode`` that doesn't require importing the class.
    """
    ThemeManager.get_instance().set_mode(theme)


def toggle_theme() -> ThemeMode:
    """Cycle light → dark → system → light. Returns the new mode."""
    return ThemeManager.get_instance().toggle()
