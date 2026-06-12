"""Topbar — pilot migration to Themeable + apply_theme pattern.

Demonstrates the explicit observer pattern that the rest of the
codebase will follow in Phase B. Reads colors via
``ThemeManager.get_instance().get(key)`` (single hex string) and
reapplies them on every theme switch via ``apply_theme()``.

The other ~17 component / view files still consume tuples through
``palette_pair`` and rely on CTk's native ``set_appearance_mode``
auto-bascule. Both pathways coexist; the ThemeManager dispatches both
on ``toggle()``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import customtkinter as ctk

from app.components.tooltip import add_tooltip
from app.config.theme import (
    RADIUS_MD,
    RADIUS_SM,
    SPACE_LG,
    SPACE_SM,
    SPACE_XS,
    Themeable,
    ThemeManager,
    get_font,
    register_themeable,
)
from app.i18n.fr import t

logger = logging.getLogger(__name__)

# Provider returns {label: (value, kind)}; kind ∈ success / warning / error / muted.
HealthProvider = Callable[[], dict[str, tuple[str, str]]]
_KIND_TO_KEY: dict[str, str] = {
    "success": "success",
    "warning": "warning",
    "error": "error",
    "muted": "fg_subtle",
}


class Topbar(ctk.CTkFrame, Themeable):
    """Horizontal bar at the top of the window. Height fixed at 64 px.

    Layout: [App title]  [global health chips]  [theme] [help]

    Phase G (2026-05-17) : hauteur passée de 44 → 64 px pour permettre
    une marge verticale (SPACE_LG = 16 px) équivalente aux marges
    horizontales du titre et des boutons d'action. Avant ce changement
    la marge top/bottom n'était que d'environ 6 px (44 - 32 button
    height = 12 / 2), créant une asymétrie visible vs les 16 px à
    gauche/droite. Voir aussi le grid_rowconfigure(0, weight=1) qui
    permet le centrage vertical des widgets dans la cellule.
    """

    HEIGHT = 64

    def __init__(
        self,
        master: ctk.CTk,
        *,
        on_theme_toggle: Callable[[], None],
        on_help: Callable[[], None],
        health_provider: HealthProvider | None = None,
    ) -> None:
        ctk.CTkFrame.__init__(
            self,
            master,
            height=self.HEIGHT,
            corner_radius=0,
        )
        self.grid_propagate(False)
        self.grid_columnconfigure(1, weight=1)
        # Le centrage vertical des widgets de la topbar dépend de la
        # cellule row=0 ayant un weight ≥ 1 — sinon la cellule prend la
        # hauteur du widget le plus haut, pas HEIGHT, et le centre
        # n'opère pas. Combiné avec sticky="w"/"e" (horizontal anchor
        # uniquement) + ``pady`` symétrique, on obtient un centrage
        # vertical propre.
        self.grid_rowconfigure(0, weight=1)

        self._on_theme = on_theme_toggle
        self._on_help = on_help
        self._health_provider = health_provider
        self._chip_widgets: dict[str, tuple[ctk.CTkLabel, ctk.CTkLabel]] = {}

        self._build_title()
        self._build_health_strip()
        self._build_actions()

        # Register at the very end of __init__, once every child widget
        # exists. ``register_themeable`` wires <Destroy> for auto-
        # unregister and runs an initial ``apply_theme()`` so we don't
        # have to repeat the configure logic here.
        register_themeable(self)

    # ------------------------------------------------------------------
    # Build (structural; colors come from apply_theme())
    # ------------------------------------------------------------------

    def _build_title(self) -> None:
        # Phase G (2026-05-17) — marge SPACE_LG identique en haut/bas/gauche
        # (et SPACE_SM côté droit = espacement interne avec les chips).
        self._title = ctk.CTkLabel(
            self,
            text=t("app.topbar_title"),
            font=get_font("body_strong"),
            anchor="w",
        )
        self._title.grid(
            row=0,
            column=0,
            sticky="w",
            padx=(SPACE_LG, SPACE_SM),
            pady=SPACE_LG,
        )

    def _build_health_strip(self) -> None:
        # Phase G — espacement SPACE_SM entre éléments + même marge
        # verticale que le titre.
        self._strip = ctk.CTkFrame(self, fg_color="transparent")
        self._strip.grid(
            row=0,
            column=1,
            sticky="e",
            padx=SPACE_SM,
            pady=SPACE_LG,
        )

    def _build_actions(self) -> None:
        # Phase G — marge droite SPACE_LG identique à la marge gauche
        # du titre (16 px) + même marge verticale.
        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.grid(
            row=0,
            column=2,
            sticky="e",
            padx=(SPACE_SM, SPACE_LG),
            pady=SPACE_LG,
        )

        # Phase F (2026-05-14, audit D-06) : ajout d'une bordure 1px sur
        # les deux boutons ◐ / ? — sans elle, le ``fg_color="transparent"``
        # les fait littéralement disparaître sur le fond gris-clair de
        # la topbar en light mode. La bordure et le ``border_color`` sont
        # repris depuis le ThemeManager dans ``apply_theme`` pour rester
        # cohérents avec le palette en cours.
        self._theme_btn = ctk.CTkButton(
            actions,
            text="◐",
            width=32,
            height=32,
            corner_radius=RADIUS_MD,
            fg_color="transparent",
            border_width=1,
            font=get_font("body_strong"),
            command=self._on_theme,
        )
        add_tooltip(self._theme_btn, t("topbar.theme_toggle_tooltip"))
        self._theme_btn.pack(side="left", padx=SPACE_SM)

        self._help_btn = ctk.CTkButton(
            actions,
            text="?",
            width=32,
            height=32,
            corner_radius=RADIUS_MD,
            fg_color="transparent",
            border_width=1,
            font=get_font("body_strong"),
            command=self._on_help,
        )
        add_tooltip(self._help_btn, t("topbar.help_tooltip"))
        self._help_btn.pack(side="left", padx=SPACE_SM)

    # ------------------------------------------------------------------
    # Themeable contract
    # ------------------------------------------------------------------

    def apply_theme(self) -> None:
        """Re-pull colors from ThemeManager and reconfigure every widget.

        Called on every theme switch by ``ThemeManager.notify_all()``.
        Also called once at the end of __init__ via ``register_themeable``.
        """
        tm = ThemeManager.get_instance()
        self.configure(fg_color=tm.get("bg_secondary"))
        self._title.configure(text_color=tm.get("text_primary"))
        for btn in (self._theme_btn, self._help_btn):
            btn.configure(
                hover_color=tm.get("bg_hover"),
                text_color=tm.get("text_primary"),
                border_color=tm.get("border"),
            )
        # Health chips encode their color via the provider's "kind"; we
        # rebuild the strip wholesale so each chip picks the new palette.
        self.refresh_health()

    # ------------------------------------------------------------------
    # Health strip (rebuilt on theme change too)
    # ------------------------------------------------------------------

    def refresh_health(self) -> None:
        """Re-poll the provider and rebuild the strip."""
        for child in self._strip.winfo_children():
            child.destroy()
        self._chip_widgets.clear()
        if self._health_provider is None:
            return
        try:
            data = self._health_provider()
        except Exception:
            logger.exception("Health provider failed")
            return
        for label, (value, kind) in data.items():
            self._build_chip(label, value, kind)

    def _build_chip(self, label: str, value: str, kind: str) -> None:
        tm = ThemeManager.get_instance()
        color = tm.get(_KIND_TO_KEY.get(kind, "fg_subtle"))
        chip = ctk.CTkFrame(
            self._strip,
            fg_color=tm.get("bg_primary"),
            border_color=color,
            border_width=1,
            corner_radius=RADIUS_SM,
        )
        chip.pack(side="left", padx=SPACE_XS)
        dot = ctk.CTkLabel(
            chip,
            text="●",
            font=get_font("body"),
            text_color=color,
            width=12,
        )
        dot.pack(side="left", padx=(SPACE_SM, SPACE_XS), pady=2)
        text = ctk.CTkLabel(
            chip,
            text=f"{label} · {value}",
            font=get_font("small"),
            text_color=tm.get("text_primary"),
        )
        text.pack(side="left", padx=(0, SPACE_SM), pady=2)
        self._chip_widgets[label] = (dot, text)

    # ------------------------------------------------------------------
    # Compatibility shims
    # ------------------------------------------------------------------

    # Older code (pre Phase A) called ``refresh_theme`` explicitly. The
    # ThemeManager now handles dispatch automatically, but the public
    # method is kept as an alias for backward compatibility.
    def refresh_theme(self) -> None:
        self.apply_theme()
