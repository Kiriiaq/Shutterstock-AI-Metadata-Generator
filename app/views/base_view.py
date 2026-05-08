"""Abstract base class for views.

Subclasses set ``view_id`` and override ``on_enter(**kwargs)`` (load data)
and ``on_leave()`` (cleanup, save drafts). The Router calls those hooks
automatically — vues never need to wire their own lifecycle.
"""

from __future__ import annotations

from typing import Any

import customtkinter as ctk

from app.config.theme import (
    SPACE_LG,
    SPACE_SM,
    get_font,
    palette_pair,
)


class BaseView(ctk.CTkFrame):
    """Common scaffolding for all vues."""

    view_id: str = ""

    def __init__(self, master: ctk.CTkFrame, **kwargs: Any) -> None:
        super().__init__(master, fg_color=palette_pair("bg"), **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def on_enter(self, **kwargs: Any) -> None:
        """Called by Router right after instantiation."""

    def on_leave(self) -> None:
        """Called by Router before destruction."""


def _modal_header(parent: ctk.CTkFrame, *, icon: str, title: str, row: int = 0) -> ctk.CTkFrame:
    """Render an icon + h1 title row anchored top-left of *parent*.

    Used at the top of every modal detail view (Modèle IA, Validation,
    Historique, Paramètres) so the panel's identity sits in the same
    upper-left corner as it does in the workspace, instead of the
    title floating alone in the centre/left without a glyph.
    """
    header = ctk.CTkFrame(parent, fg_color="transparent")
    header.grid(row=row, column=0, sticky="w", pady=(0, SPACE_LG))
    ctk.CTkLabel(
        header,
        text=icon,
        font=get_font("h1"),
        text_color=palette_pair("fg_muted"),
        anchor="w",
    ).pack(side="left", padx=(0, SPACE_SM))
    ctk.CTkLabel(
        header,
        text=title,
        font=get_font("h1"),
        text_color=palette_pair("fg"),
        anchor="w",
    ).pack(side="left")
    return header
