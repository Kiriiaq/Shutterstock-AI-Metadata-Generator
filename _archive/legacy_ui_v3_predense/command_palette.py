"""Command palette — searchable list of actions, opened via Ctrl+K.

Pattern: Toplevel with no decorations, search field at top, filtered list
below. Arrow keys navigate, Enter executes, Escape dismisses. The action
list is rebuilt every time the palette opens (via a provider callable)
so caller can include context-dependent commands.
"""

from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass, field

import customtkinter as ctk

from app.config.theme import (
    RADIUS_LG,
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


@dataclass(frozen=True)
class Command:
    """One palette entry."""

    id: str
    label: str  # user-visible
    callback: Callable[[], None]
    shortcut: str = ""  # display shortcut, e.g. "Ctrl+S"
    keywords: tuple[str, ...] = field(default_factory=tuple)


CommandProvider = Callable[[], list[Command]]


class CommandPalette:
    """Manages a single Toplevel palette window. ``open()`` shows it,
    ``close()`` destroys it. Reopening builds a fresh window so we don't
    have to deal with Tk widget reuse edge-cases.
    """

    MAX_VISIBLE = 8

    def __init__(self, parent: ctk.CTk, provider: CommandProvider) -> None:
        self._parent = parent
        self._provider = provider
        self._win: ctk.CTkToplevel | None = None
        self._entry: ctk.CTkEntry | None = None
        self._list_frame: ctk.CTkScrollableFrame | None = None
        self._row_widgets: list[tuple[ctk.CTkFrame, Command]] = []
        self._highlighted: int = 0
        self._all_commands: list[Command] = []

    # ------------------------------------------------------------------

    def open(self) -> None:
        if self._win is not None and self._win.winfo_exists():
            self._win.lift()
            return
        self._all_commands = list(self._provider())
        self._highlighted = 0
        self._build_window()
        self._refresh_results("")

    def close(self) -> None:
        if self._win is None:
            return
        try:
            self._win.destroy()
        except tk.TclError:
            pass
        self._win = None
        self._entry = None
        self._list_frame = None
        self._row_widgets.clear()

    # ------------------------------------------------------------------

    def _build_window(self) -> None:
        win = ctk.CTkToplevel(self._parent)
        win.title("Palette de commandes")
        win.transient(self._parent)
        win.configure(fg_color=get_color("bg"))
        win.geometry("560x420")
        try:
            win.attributes("-topmost", True)
        except tk.TclError:
            pass
        self._center_top(win)

        wrapper = ctk.CTkFrame(
            win,
            fg_color=get_color("bg_elevated"),
            border_color=get_color("border"),
            border_width=1,
            corner_radius=RADIUS_LG,
        )
        wrapper.pack(fill="both", expand=True, padx=SPACE_SM, pady=SPACE_SM)

        self._entry = ctk.CTkEntry(
            wrapper,
            placeholder_text=t("topbar.search_placeholder"),
            font=get_font("body"),
            fg_color=get_color("bg"),
            text_color=get_color("fg"),
            border_color=get_color("border"),
            corner_radius=RADIUS_MD,
            height=36,
        )
        self._entry.pack(fill="x", padx=SPACE_MD, pady=(SPACE_MD, SPACE_SM))
        self._entry.bind("<KeyRelease>", lambda _e: self._refresh_results(self._entry.get()))
        self._entry.bind("<Down>", lambda _e: self._move_highlight(+1) or "break")
        self._entry.bind("<Up>", lambda _e: self._move_highlight(-1) or "break")
        self._entry.bind("<Return>", lambda _e: self._execute_highlighted() or "break")
        self._entry.bind("<Escape>", lambda _e: self.close())

        self._list_frame = ctk.CTkScrollableFrame(
            wrapper,
            fg_color="transparent",
            corner_radius=0,
        )
        self._list_frame.pack(fill="both", expand=True, padx=SPACE_MD, pady=(0, SPACE_MD))

        win.protocol("WM_DELETE_WINDOW", self.close)
        try:
            win.grab_set()
        except tk.TclError:
            pass
        win.after(50, self._entry.focus_set)
        self._win = win

    # ------------------------------------------------------------------

    def _refresh_results(self, query: str) -> None:
        if self._list_frame is None:
            return
        for child in self._list_frame.winfo_children():
            child.destroy()
        self._row_widgets.clear()

        matches = self._filter(query)
        if not matches:
            ctk.CTkLabel(
                self._list_frame,
                text="Aucune commande ne correspond.",
                font=get_font("body"),
                text_color=get_color("fg_muted"),
            ).pack(padx=SPACE_LG, pady=SPACE_LG)
            self._highlighted = 0
            return

        for idx, cmd in enumerate(matches):
            row = self._build_row(cmd, idx)
            self._row_widgets.append((row, cmd))

        self._highlighted = 0
        self._update_highlight_styles()

    def _filter(self, query: str) -> list[Command]:
        q = query.strip().lower()
        if not q:
            return self._all_commands
        tokens = [tok for tok in q.split() if tok]

        def matches(cmd: Command) -> bool:
            haystack = " ".join((cmd.label, *cmd.keywords)).lower()
            return all(tok in haystack for tok in tokens)

        return [c for c in self._all_commands if matches(c)]

    def _build_row(self, cmd: Command, idx: int) -> ctk.CTkFrame:
        assert self._list_frame is not None
        row = ctk.CTkFrame(
            self._list_frame,
            fg_color="transparent",
            corner_radius=RADIUS_MD,
            height=36,
        )
        row.pack(fill="x", pady=1)
        row.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            row,
            text=cmd.label,
            font=get_font("body"),
            text_color=get_color("fg"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=SPACE_MD, pady=SPACE_XS)

        if cmd.shortcut:
            ctk.CTkLabel(
                row,
                text=cmd.shortcut,
                font=get_font("small"),
                text_color=get_color("fg_subtle"),
                anchor="e",
            ).grid(row=0, column=1, sticky="e", padx=SPACE_MD)

        for w in (row, *row.winfo_children()):
            w.bind("<Button-1>", lambda _e, i=idx: self._execute_index(i))
            w.bind("<Enter>", lambda _e, i=idx: self._set_highlight(i))
        return row

    # ------------------------------------------------------------------

    def _move_highlight(self, delta: int) -> bool:
        if not self._row_widgets:
            return True
        self._highlighted = (self._highlighted + delta) % len(self._row_widgets)
        self._update_highlight_styles()
        return True

    def _set_highlight(self, idx: int) -> None:
        if 0 <= idx < len(self._row_widgets):
            self._highlighted = idx
            self._update_highlight_styles()

    def _update_highlight_styles(self) -> None:
        for i, (row, _cmd) in enumerate(self._row_widgets):
            row.configure(fg_color=get_color("bg_active") if i == self._highlighted else "transparent")

    def _execute_highlighted(self) -> bool:
        if 0 <= self._highlighted < len(self._row_widgets):
            self._execute_index(self._highlighted)
        return True

    def _execute_index(self, idx: int) -> None:
        if not (0 <= idx < len(self._row_widgets)):
            return
        _, cmd = self._row_widgets[idx]
        self.close()
        try:
            cmd.callback()
        except Exception:
            logger.exception("Command callback failed: %s", cmd.id)

    # ------------------------------------------------------------------

    def _center_top(self, win: ctk.CTkToplevel) -> None:
        try:
            self._parent.update_idletasks()
            px = self._parent.winfo_rootx()
            py = self._parent.winfo_rooty()
            pw = self._parent.winfo_width()
            win.update_idletasks()
            ww = win.winfo_reqwidth()
            x = max(0, px + (pw - ww) // 2)
            y = max(0, py + 80)
            win.geometry(f"+{x}+{y}")
        except tk.TclError:
            pass
