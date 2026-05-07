"""Sidebar navigation with collapsible mode (220 px ↔ 56 px)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Final

import customtkinter as ctk

from app.components.tooltip import add_tooltip
from app.config.theme import (
    RADIUS_MD,
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    SPACE_XS,
    get_color,
    get_font,
)
from app.i18n.fr import t

logger = logging.getLogger(__name__)


# (view_id, icon, i18n_label_key, section)
NavEntry = tuple[str, str, str, str]
NAV_ENTRIES: Final[list[NavEntry]] = [
    ("home", "🏠", "nav.home", "production"),
    ("sources", "📁", "nav.sources", "production"),
    ("analyze", "⚡", "nav.analyze", "production"),
    ("editor", "✎", "nav.editor", "production"),
    ("validate", "✓", "nav.validate", "production"),
    ("upload", "↑", "nav.upload", "production"),
    ("ai_control", "🧠", "nav.ai_control", "system"),
    ("audit", "📜", "nav.audit", "system"),
    ("settings", "⚙", "nav.settings", "system"),
]


class _NavRow(ctk.CTkFrame):
    """One sidebar row. Width follows parent; height fixed."""

    HEIGHT = 36

    def __init__(
        self,
        master: ctk.CTkFrame,
        *,
        icon: str,
        label: str,
        on_click: Callable[[], None],
    ) -> None:
        super().__init__(
            master,
            height=self.HEIGHT,
            corner_radius=RADIUS_MD,
            fg_color="transparent",
        )
        self.grid_propagate(False)
        self.grid_columnconfigure(1, weight=1)

        self._on_click = on_click
        self._is_active = False

        self._icon_label = ctk.CTkLabel(
            self,
            text=icon,
            font=get_font("h3"),
            text_color=get_color("fg_muted"),
            width=24,
            anchor="center",
        )
        self._icon_label.grid(row=0, column=0, padx=(SPACE_MD, SPACE_SM), pady=SPACE_XS)

        self._text_label = ctk.CTkLabel(
            self,
            text=label,
            font=get_font("body"),
            text_color=get_color("fg"),
            anchor="w",
        )
        self._text_label.grid(row=0, column=1, sticky="w")

        for w in (self, self._icon_label, self._text_label):
            w.bind("<Button-1>", lambda _e: self._on_click())
            w.bind("<Enter>", lambda _e: self._hover(True))
            w.bind("<Leave>", lambda _e: self._hover(False))

    def set_active(self, active: bool) -> None:
        self._is_active = active
        self._refresh_colors()

    def show_label(self, show: bool) -> None:
        if show:
            self._text_label.grid()
        else:
            self._text_label.grid_remove()

    def refresh_theme(self) -> None:
        self._refresh_colors()

    def _hover(self, hovering: bool) -> None:
        if self._is_active:
            return
        self.configure(fg_color=get_color("bg_hover") if hovering else "transparent")

    def _refresh_colors(self) -> None:
        if self._is_active:
            self.configure(fg_color=get_color("bg_active"))
            self._icon_label.configure(text_color=get_color("accent"))
            self._text_label.configure(text_color=get_color("fg"))
        else:
            self.configure(fg_color="transparent")
            self._icon_label.configure(text_color=get_color("fg_muted"))
            self._text_label.configure(text_color=get_color("fg"))


class Sidebar(ctk.CTkFrame):
    """Vertical navigation. ``Ctrl+B`` toggles collapsed mode."""

    EXPANDED_WIDTH = 220
    COLLAPSED_WIDTH = 56

    def __init__(
        self,
        master: ctk.CTk,
        *,
        on_navigate: Callable[[str], None],
    ) -> None:
        super().__init__(
            master,
            width=self.EXPANDED_WIDTH,
            corner_radius=0,
            fg_color=get_color("bg_elevated"),
        )
        self.grid_propagate(False)

        self._on_navigate = on_navigate
        self._collapsed: bool = False
        self._rows: dict[str, _NavRow] = {}
        self._section_labels: dict[str, ctk.CTkLabel] = {}
        self._build_layout()

    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(99, weight=1)  # push collapse button to bottom

        # App title
        title = ctk.CTkLabel(
            self,
            text="Shutterstock\nAnalyzer",
            font=get_font("body_strong"),
            text_color=get_color("fg"),
            justify="left",
            anchor="w",
        )
        title.grid(row=0, column=0, sticky="ew", padx=SPACE_LG, pady=(SPACE_LG, SPACE_MD))
        self._title_label = title

        row_idx = 1
        current_section: str | None = None
        for view_id, icon, label_key, section in NAV_ENTRIES:
            if section != current_section:
                section_label = ctk.CTkLabel(
                    self,
                    text=t(f"nav.section.{section}").upper(),
                    font=get_font("small"),
                    text_color=get_color("fg_subtle"),
                    anchor="w",
                )
                section_label.grid(row=row_idx, column=0, sticky="ew", padx=SPACE_LG, pady=(SPACE_MD, SPACE_XS))
                self._section_labels[section] = section_label
                current_section = section
                row_idx += 1

            row = _NavRow(
                self,
                icon=icon,
                label=t(label_key),
                on_click=lambda vid=view_id: self._on_navigate(vid),
            )
            row.grid(row=row_idx, column=0, sticky="ew", padx=SPACE_SM, pady=1)
            self._rows[view_id] = row
            row_idx += 1

        # Collapse button at the bottom
        self._collapse_btn = ctk.CTkButton(
            self,
            text="◀",
            width=32,
            height=32,
            corner_radius=RADIUS_MD,
            fg_color="transparent",
            hover_color=get_color("bg_hover"),
            text_color=get_color("fg_muted"),
            font=get_font("body_strong"),
            command=self.toggle_collapsed,
        )
        self._collapse_btn.grid(row=100, column=0, sticky="e", padx=SPACE_SM, pady=SPACE_SM)
        add_tooltip(self._collapse_btn, t("nav.collapse_tooltip"))

    # ------------------------------------------------------------------

    def set_active(self, view_id: str) -> None:
        for vid, row in self._rows.items():
            row.set_active(vid == view_id)

    def toggle_collapsed(self) -> None:
        self._collapsed = not self._collapsed
        new_width = self.COLLAPSED_WIDTH if self._collapsed else self.EXPANDED_WIDTH
        self.configure(width=new_width)

        for row in self._rows.values():
            row.show_label(not self._collapsed)
        for label in self._section_labels.values():
            if self._collapsed:
                label.grid_remove()
            else:
                label.grid()
        self._title_label.configure(text="" if self._collapsed else "Shutterstock\nAnalyzer")
        self._collapse_btn.configure(text="▶" if self._collapsed else "◀")

    def refresh_theme(self) -> None:
        """Re-pull theme colours after a theme switch."""
        self.configure(fg_color=get_color("bg_elevated"))
        self._title_label.configure(text_color=get_color("fg"))
        self._collapse_btn.configure(
            hover_color=get_color("bg_hover"),
            text_color=get_color("fg_muted"),
        )
        for row in self._rows.values():
            row.refresh_theme()
        for label in self._section_labels.values():
            label.configure(text_color=get_color("fg_subtle"))
