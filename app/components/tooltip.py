"""Tooltip — Toplevel-based, themed, keyboard-accessible.

Usage::

    from app.components.tooltip import add_tooltip
    add_tooltip(my_button, "Cliquez pour scanner le dossier")
"""

from __future__ import annotations

import logging
import tkinter as tk

import customtkinter as ctk

from app.config.theme import RADIUS_SM, SPACE_SM, SPACE_XS, get_color, get_font

logger = logging.getLogger(__name__)


class Tooltip:
    """Attach to a single widget. Show on <Enter> after delay, hide on <Leave>."""

    def __init__(
        self,
        widget: tk.Misc,
        text: str,
        *,
        delay_ms: int = 500,
        wraplength: int = 260,
    ) -> None:
        self._widget = widget
        self._text = text
        self._delay_ms = max(0, int(delay_ms))
        self._wraplength = wraplength
        self._after_id: str | None = None
        self._tip: ctk.CTkToplevel | None = None

        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<FocusOut>", self._hide, add="+")

    def update_text(self, text: str) -> None:
        self._text = text

    # ------------------------------------------------------------------

    def _schedule(self, _event: object) -> None:
        self._cancel_pending()
        self._after_id = self._widget.after(self._delay_ms, self._show)

    def _cancel_pending(self) -> None:
        if self._after_id is not None:
            try:
                self._widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _show(self) -> None:
        if self._tip is not None or not self._text:
            return
        try:
            x = self._widget.winfo_rootx() + 12
            y = self._widget.winfo_rooty() + self._widget.winfo_height() + 6
        except tk.TclError:
            return

        tip = ctk.CTkToplevel(self._widget)
        tip.wm_overrideredirect(True)
        try:
            tip.attributes("-topmost", True)
        except tk.TclError:
            pass
        tip.configure(fg_color=get_color("fg"))
        tip.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            tip,
            text=self._text,
            font=get_font("small"),
            text_color=get_color("bg"),
            wraplength=self._wraplength,
            justify="left",
            corner_radius=RADIUS_SM,
        ).pack(padx=SPACE_SM, pady=SPACE_XS)
        self._tip = tip

    def _hide(self, _event: object = None) -> None:
        self._cancel_pending()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None


def add_tooltip(widget: tk.Misc, text: str, **kwargs: object) -> Tooltip:
    """Convenience: attach and return a :class:`Tooltip`."""
    return Tooltip(widget, text, **kwargs)  # type: ignore[arg-type]
