"""Topbar — title, global health strip, theme toggle, help.

The search/CommandPalette button is gone in the dense-atelier layout:
each tool's panel exposes its own actions, so there is no global nav
left to search through. The freed space hosts the always-visible
health strip (Backend · ExifTool, populated by a provider callable).
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
    SPACE_MD,
    SPACE_SM,
    SPACE_XS,
    get_color,
    get_font,
)
from app.i18n.fr import t

logger = logging.getLogger(__name__)

# Provider returns {label: (value, color_kind)}; color_kind ∈ success/warning/error/muted.
HealthProvider = Callable[[], dict[str, tuple[str, str]]]
_KIND_TO_COLOR: dict[str, str] = {
    "success": "success",
    "warning": "warning",
    "error": "error",
    "muted": "fg_subtle",
}


class Topbar(ctk.CTkFrame):
    """Horizontal bar at the top of the window. Height fixed at 44 px.

    Layout: [App title]  [global health chips]  [theme] [help]
    """

    HEIGHT = 44

    def __init__(
        self,
        master: ctk.CTk,
        *,
        on_theme_toggle: Callable[[], None],
        on_help: Callable[[], None],
        health_provider: HealthProvider | None = None,
    ) -> None:
        super().__init__(
            master,
            height=self.HEIGHT,
            corner_radius=0,
            fg_color=get_color("bg_elevated"),
        )
        self.grid_propagate(False)
        self.grid_columnconfigure(1, weight=1)

        self._on_theme = on_theme_toggle
        self._on_help = on_help
        self._health_provider = health_provider
        self._chip_widgets: dict[str, tuple[ctk.CTkLabel, ctk.CTkLabel]] = {}

        self._build_title()
        self._build_health_strip()
        self._build_actions()
        self.refresh_health()

    # ------------------------------------------------------------------

    def _build_title(self) -> None:
        self._title = ctk.CTkLabel(
            self,
            text="ShutterstockAnalyzer v2.0.0 — Atelier",
            font=get_font("body_strong"),
            text_color=get_color("fg"),
            anchor="w",
        )
        self._title.grid(row=0, column=0, sticky="w", padx=SPACE_LG)

    def _build_health_strip(self) -> None:
        self._strip = ctk.CTkFrame(self, fg_color="transparent")
        self._strip.grid(row=0, column=1, sticky="e", padx=SPACE_MD)

    def _build_actions(self) -> None:
        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.grid(row=0, column=2, sticky="e", padx=SPACE_MD)

        self._theme_btn = self._action_button(actions, "◐", on_click=self._on_theme)
        add_tooltip(self._theme_btn, t("topbar.theme_toggle_tooltip"))
        self._theme_btn.pack(side="left", padx=SPACE_SM)

        self._help_btn = self._action_button(actions, "?", on_click=self._on_help)
        add_tooltip(self._help_btn, t("topbar.help_tooltip"))
        self._help_btn.pack(side="left", padx=SPACE_SM)

    def _action_button(
        self,
        parent: ctk.CTkFrame,
        text: str,
        *,
        on_click: Callable[[], None],
    ) -> ctk.CTkButton:
        return ctk.CTkButton(
            parent,
            text=text,
            width=32,
            height=32,
            corner_radius=RADIUS_MD,
            fg_color="transparent",
            hover_color=get_color("bg_hover"),
            text_color=get_color("fg"),
            font=get_font("body_strong"),
            command=on_click,
        )

    # ------------------------------------------------------------------

    def refresh_health(self) -> None:
        """Re-poll the health provider and rebuild the strip."""
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
        color = get_color(_KIND_TO_COLOR.get(kind, "fg_subtle"))
        chip = ctk.CTkFrame(
            self._strip,
            fg_color=get_color("bg"),
            border_color=color,
            border_width=1,
            corner_radius=RADIUS_SM,
        )
        chip.pack(side="left", padx=SPACE_XS)
        dot = ctk.CTkLabel(chip, text="●", font=get_font("body"), text_color=color, width=12)
        dot.pack(side="left", padx=(SPACE_SM, SPACE_XS), pady=2)
        text = ctk.CTkLabel(
            chip,
            text=f"{label} · {value}",
            font=get_font("small"),
            text_color=get_color("fg"),
        )
        text.pack(side="left", padx=(0, SPACE_SM), pady=2)
        self._chip_widgets[label] = (dot, text)

    def refresh_theme(self) -> None:
        self.configure(fg_color=get_color("bg_elevated"))
        self._title.configure(text_color=get_color("fg"))
        for btn in (self._theme_btn, self._help_btn):
            btn.configure(hover_color=get_color("bg_hover"), text_color=get_color("fg"))
        self.refresh_health()

    # ``set_breadcrumb`` kept as a no-op for API compatibility — the dense
    # atelier no longer surfaces a breadcrumb because there is no nav.
    def set_breadcrumb(self, _label: str) -> None:
        return None
