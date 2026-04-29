"""EmptyState — large icon + title + subtitle + optional action button.

Use when a list/section has no data, or as a placeholder for a not-yet-
selected context. Sized for the central content area.
"""

from __future__ import annotations

from collections.abc import Callable

import customtkinter as ctk

from app.config.theme import (
    RADIUS_MD,
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    get_color,
    get_font,
)


class EmptyState(ctk.CTkFrame):
    """Centred empty-state with optional action."""

    def __init__(
        self,
        master: ctk.CTkFrame,
        *,
        icon: str,
        title: str,
        subtitle: str | None = None,
        action_label: str | None = None,
        on_action: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master, fg_color=get_color("bg"))
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.grid(row=0, column=0)

        ctk.CTkLabel(
            inner,
            text=icon,
            font=ctk.CTkFont(size=48),
            text_color=get_color("fg_subtle"),
        ).pack(pady=(0, SPACE_MD))

        ctk.CTkLabel(
            inner,
            text=title,
            font=get_font("h2"),
            text_color=get_color("fg"),
            justify="center",
        ).pack(pady=(0, SPACE_SM))

        if subtitle:
            ctk.CTkLabel(
                inner,
                text=subtitle,
                font=get_font("body"),
                text_color=get_color("fg_muted"),
                justify="center",
                wraplength=420,
            ).pack(pady=(0, SPACE_LG))

        if action_label and on_action is not None:
            ctk.CTkButton(
                inner,
                text=action_label,
                font=get_font("body_strong"),
                fg_color=get_color("accent"),
                hover_color=get_color("accent_hover"),
                text_color=get_color("accent_fg"),
                corner_radius=RADIUS_MD,
                height=36,
                command=on_action,
            ).pack(pady=(0, SPACE_LG))
