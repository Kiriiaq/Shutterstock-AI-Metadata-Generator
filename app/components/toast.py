"""Bottom-right stacked toast notifications.

Usage::

    toasts = ToastManager(root)
    toasts.show("Sauvegarde réussie", kind="success")
    toasts.show("Échec FTPS : connexion refusée", kind="error", timeout_ms=8000)

Toasts auto-dismiss after their timeout, click to dismiss early. The
manager re-stacks survivors so the column stays compact.
"""

from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable
from typing import Literal

import customtkinter as ctk

from app.config.theme import RADIUS_MD, SPACE_LG, SPACE_MD, SPACE_SM, get_color, get_font

logger = logging.getLogger(__name__)

ToastKind = Literal["success", "warning", "error", "info"]
_VALID_KINDS: frozenset[str] = frozenset({"success", "warning", "error", "info"})


class _Toast(ctk.CTkToplevel):
    """One toast; auto-destroys on timeout."""

    def __init__(
        self,
        master: tk.Misc,
        message: str,
        kind: ToastKind,
        *,
        timeout_ms: int,
        on_close: Callable[["_Toast"], None],
    ) -> None:
        super().__init__(master)
        self.wm_overrideredirect(True)
        try:
            self.attributes("-topmost", True)
        except tk.TclError:
            pass
        self._on_close = on_close
        self._after_id: str | None = None

        if kind not in _VALID_KINDS:
            kind = "info"
        bg = get_color(f"{kind}_bg")
        fg = get_color(kind)

        self.configure(fg_color=bg)
        frame = ctk.CTkFrame(
            self,
            fg_color=bg,
            border_color=fg,
            border_width=1,
            corner_radius=RADIUS_MD,
        )
        frame.pack(fill="both", expand=True)
        ctk.CTkLabel(
            frame,
            text=message,
            text_color=fg,
            font=get_font("body_strong" if kind in ("error", "warning") else "body"),
            wraplength=320,
            justify="left",
        ).pack(padx=SPACE_LG, pady=SPACE_MD)

        self.bind("<Button-1>", lambda _e: self.dismiss())
        self._after_id = self.after(max(1000, int(timeout_ms)), self.dismiss)

    def dismiss(self) -> None:
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        try:
            self.destroy()
        except tk.TclError:
            pass
        try:
            self._on_close(self)
        except Exception:
            logger.exception("Toast on_close handler failed")


class ToastManager:
    """Stacks toasts in the bottom-right of the root window."""

    GAP = SPACE_SM
    MARGIN = SPACE_LG

    def __init__(self, root: ctk.CTk) -> None:
        self._root = root
        self._toasts: list[_Toast] = []

    def show(
        self,
        message: str,
        kind: ToastKind = "info",
        *,
        timeout_ms: int = 4000,
    ) -> None:
        toast = _Toast(self._root, message, kind, timeout_ms=timeout_ms, on_close=self._on_close)
        self._toasts.append(toast)
        self._reposition()

    def _on_close(self, toast: _Toast) -> None:
        try:
            self._toasts.remove(toast)
        except ValueError:
            pass
        self._reposition()

    def _reposition(self) -> None:
        try:
            self._root.update_idletasks()
            screen_w = self._root.winfo_screenwidth()
            screen_h = self._root.winfo_screenheight()
        except tk.TclError:
            return
        y = screen_h - self.MARGIN
        for toast in reversed(self._toasts):
            try:
                toast.update_idletasks()
                w = max(toast.winfo_reqwidth(), 280)
                h = toast.winfo_reqheight()
                x = screen_w - w - self.MARGIN
                y -= h
                toast.geometry(f"{w}x{h}+{x}+{y}")
                y -= self.GAP
            except tk.TclError:
                continue
