"""Topbar — breadcrumb, search-trigger, theme toggle, help button."""

from __future__ import annotations

import logging
from collections.abc import Callable

import customtkinter as ctk

from app.components.tooltip import add_tooltip
from app.config.theme import (
    RADIUS_MD,
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    get_color,
    get_font,
)
from app.i18n.fr import t

logger = logging.getLogger(__name__)


class Topbar(ctk.CTkFrame):
    """Horizontal bar at the top of the window. Height fixed at 44 px."""

    HEIGHT = 44

    def __init__(
        self,
        master: ctk.CTk,
        *,
        on_search_trigger: Callable[[], None],
        on_theme_toggle: Callable[[], None],
        on_help: Callable[[], None],
    ) -> None:
        super().__init__(
            master,
            height=self.HEIGHT,
            corner_radius=0,
            fg_color=get_color("bg_elevated"),
        )
        self.grid_propagate(False)
        self.grid_columnconfigure(1, weight=1)

        self._on_search = on_search_trigger
        self._on_theme = on_theme_toggle
        self._on_help = on_help

        self._build_breadcrumb()
        self._build_search()
        self._build_actions()

    # ------------------------------------------------------------------

    def _build_breadcrumb(self) -> None:
        self._breadcrumb = ctk.CTkLabel(
            self,
            text="",
            font=get_font("body_strong"),
            text_color=get_color("fg"),
            anchor="w",
        )
        self._breadcrumb.grid(row=0, column=0, sticky="w", padx=SPACE_LG)

    def _build_search(self) -> None:
        # Searchbox visually a CTkButton — clicking opens command palette.
        # We don't allow direct typing here on purpose: one search modality.
        search = ctk.CTkButton(
            self,
            text=t("topbar.search_placeholder"),
            anchor="w",
            font=get_font("body"),
            height=30,
            corner_radius=RADIUS_MD,
            fg_color=get_color("bg"),
            hover_color=get_color("bg_hover"),
            text_color=get_color("fg_muted"),
            border_width=1,
            border_color=get_color("border"),
            command=self._on_search,
        )
        search.grid(row=0, column=1, padx=SPACE_LG, pady=SPACE_SM, sticky="ew")
        self._search_btn = search

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

    def set_breadcrumb(self, label: str) -> None:
        self._breadcrumb.configure(text=label)

    def refresh_theme(self) -> None:
        self.configure(fg_color=get_color("bg_elevated"))
        self._breadcrumb.configure(text_color=get_color("fg"))
        self._search_btn.configure(
            fg_color=get_color("bg"),
            hover_color=get_color("bg_hover"),
            text_color=get_color("fg_muted"),
            border_color=get_color("border"),
        )
        for btn in (self._theme_btn, self._help_btn):
            btn.configure(
                hover_color=get_color("bg_hover"),
                text_color=get_color("fg"),
            )
