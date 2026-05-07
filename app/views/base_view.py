"""Abstract base class for views.

Subclasses set ``view_id`` and override ``on_enter(**kwargs)`` (load data)
and ``on_leave()`` (cleanup, save drafts). The Router calls those hooks
automatically — vues never need to wire their own lifecycle.
"""

from __future__ import annotations

from typing import Any

import customtkinter as ctk

from app.config.theme import palette_pair


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
