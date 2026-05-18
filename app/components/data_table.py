"""DataTable — themed wrapper around ttk.Treeview for dense lists.

ttk.Style is global to the Tk app, so we configure a dedicated style name
("AppTheme.Treeview") on construction and on theme switch. Vues should
hold one DataTable per logical list and call ``set_rows`` / ``get_selected``.
"""

from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass, field
from tkinter import ttk
from typing import Any, Literal

import customtkinter as ctk

from app.config.theme import (
    get_color,
    palette_pair,
)

logger = logging.getLogger(__name__)

_STYLE_NAME = "AppTheme.Treeview"
_HEADING_STYLE = "AppTheme.Treeview.Heading"


@dataclass
class Column:
    """One DataTable column."""

    id: str
    label: str
    width: int = 120
    anchor: Literal["w", "e", "center"] = "w"
    sort_key: Callable[[Any], Any] = field(default=lambda v: v)


def apply_treeview_style() -> None:
    """Configure the global ttk style used by DataTable.

    On Windows the default ttk theme is ``vista``, which **ignores**
    our ``style.configure(background=...)`` calls for Treeview — it
    falls back to native OS colors (white). The ``clam`` theme is the
    only built-in theme that respects ``style.configure`` for
    Treeview backgrounds, so we force-switch to it the first time.

    Subsequent calls (on every theme toggle) just re-issue the
    configure with the new palette colors.
    """
    style = ttk.Style()

    # First call: force 'clam' so subsequent .configure() calls take
    # effect. theme_use() is process-wide and persists.
    if "clam" in style.theme_names() and style.theme_use() != "clam":
        try:
            style.theme_use("clam")
        except tk.TclError:
            logger.exception("Could not switch ttk theme to 'clam'")

    # Pull from the v3-spec keys; legacy aliases still resolve
    # (e.g. "fg" → text_primary, "bg" → bg_primary).
    rows_bg = get_color("bg_secondary")  # rows: card surface
    field_bg = get_color("bg_primary")  # empty area below rows: canvas
    fg = get_color("text_primary")
    sel_bg = get_color("accent")
    sel_fg = get_color("accent_fg")
    border = get_color("border")
    heading_bg = get_color("bg_elevated")
    heading_active = get_color("bg_hover")

    style.configure(
        _STYLE_NAME,
        background=rows_bg,
        fieldbackground=field_bg,
        foreground=fg,
        bordercolor=border,
        rowheight=26,
        borderwidth=0,
    )
    style.map(
        _STYLE_NAME,
        background=[("selected", sel_bg)],
        foreground=[("selected", sel_fg)],
    )
    style.configure(
        _HEADING_STYLE,
        background=heading_bg,
        foreground=fg,
        relief="flat",
        borderwidth=0,
    )
    style.map(
        _HEADING_STYLE,
        background=[("active", heading_active)],
    )


class DataTable(ctk.CTkFrame):
    """ttk.Treeview + scrollbars + theme sync.

    Constructor takes a list of Column definitions and a select_mode.
    Use ``set_rows`` to load data (each row is a dict keyed by column id),
    ``get_selected`` to read selection, ``on_select`` / ``on_activate``
    to attach handlers.
    """

    def __init__(
        self,
        master: ctk.CTkFrame,
        *,
        columns: list[Column],
        select_mode: Literal["browse", "extended"] = "extended",
        height: int = 10,
    ) -> None:
        # Use palette_pair tuple (not get_color string) so the frame's
        # background auto-bascules on ``set_appearance_mode``: that's
        # the difference between a DataTable that stays at its initial
        # theme color forever and one that follows light/dark switches.
        # (refresh_theme() also re-applies it explicitly, belt + braces.)
        super().__init__(master, fg_color=palette_pair("bg_secondary"))
        self._columns = columns
        self._sort_state: dict[str, bool] = {}  # col_id -> reverse?
        self._on_select_cb: Callable[[list[dict[str, Any]]], None] | None = None
        self._on_activate_cb: Callable[[dict[str, Any]], None] | None = None
        self._row_data: dict[str, dict[str, Any]] = {}  # iid -> row
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        apply_treeview_style()

        self._tree = ttk.Treeview(
            self,
            columns=[c.id for c in columns],
            show="headings",
            selectmode=select_mode,
            style=_STYLE_NAME,
            height=height,
        )
        for col in columns:
            self._tree.heading(
                col.id,
                text=col.label,
                anchor=col.anchor,
                command=lambda cid=col.id: self._sort_by(cid),
            )
            self._tree.column(col.id, width=col.width, anchor=col.anchor, stretch=True)
        self._tree.tag_configure("alt", background=get_color("bg_elevated"))

        self._tree.grid(row=0, column=0, sticky="nsew")

        # Vertical scrollbar only — columns are configured with
        # ``stretch=True`` so horizontal overflow can't really happen,
        # and the unified-scroll spec asks for as few internal
        # scrollbars as possible. Y-scroll is kept because long row
        # lists would otherwise force the panel to grow infinitely
        # and dominate the global window scroll.
        self._yscroll = ttk.Scrollbar(self, orient="vertical", command=self._tree.yview)
        self._yscroll.grid(row=0, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=self._yscroll.set)

        self._tree.bind("<<TreeviewSelect>>", self._handle_select)
        self._tree.bind("<Double-1>", self._handle_activate)
        self._tree.bind("<Return>", self._handle_activate)
        # Phase G+4 (2026-05-19) — Ctrl+A sélectionne toutes les lignes
        # (mode select="extended" uniquement, no-op en "browse"). Bind
        # local au Treeview pour ne pas interférer avec le Ctrl+A des
        # widgets de saisie (CTkEntry, CTkTextbox).
        self._tree.bind("<Control-a>", self._handle_ctrl_a)
        self._tree.bind("<Control-A>", self._handle_ctrl_a)

    # ------------------------------------------------------------------

    def set_rows(self, rows: list[dict[str, Any]]) -> None:
        """Replace all rows. Each row is a dict keyed by column id."""
        self._tree.delete(*self._tree.get_children())
        self._row_data.clear()
        for idx, row in enumerate(rows):
            iid = f"row-{idx}"
            values = [row.get(c.id, "") for c in self._columns]
            tags = ("alt",) if idx % 2 else ()
            self._tree.insert("", "end", iid=iid, values=values, tags=tags)
            self._row_data[iid] = row

    def get_selected(self) -> list[dict[str, Any]]:
        return [self._row_data[iid] for iid in self._tree.selection() if iid in self._row_data]

    def select_all(self) -> None:
        """Phase G+4 — sélectionne toutes les lignes (mode extended)."""
        children = self._tree.get_children()
        if children:
            self._tree.selection_set(children)

    def deselect_all(self) -> None:
        """Phase G+4 — efface la sélection."""
        sel = self._tree.selection()
        if sel:
            self._tree.selection_remove(*sel)

    def _handle_ctrl_a(self, _event: object) -> str:
        """Ctrl+A → ``select_all`` puis ``break`` pour empêcher Tk de
        propager l'event à des parents."""
        self.select_all()
        return "break"

    def on_select(self, callback: Callable[[list[dict[str, Any]]], None]) -> None:
        self._on_select_cb = callback

    def on_activate(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._on_activate_cb = callback

    def refresh_theme(self) -> None:
        """Re-apply ttk style + frame background after a theme switch.

        The CTkFrame is built with a ``palette_pair`` tuple so it
        auto-switches via ``set_appearance_mode``, but we re-issue
        configure here as a belt-and-braces guard for any edge case
        (e.g. partial repaint while a Treeview is rebuilding).
        """
        apply_treeview_style()
        try:
            self.configure(fg_color=palette_pair("bg_secondary"))
            self._tree.tag_configure("alt", background=get_color("bg_elevated"))
        except Exception:
            logger.exception("DataTable refresh_theme failed")

    # ------------------------------------------------------------------

    def _sort_by(self, col_id: str) -> None:
        col = next((c for c in self._columns if c.id == col_id), None)
        if col is None:
            return
        # Default: first click on a column sorts ascending; clicking the
        # same column again flips to descending. Clicking a different
        # column resets to ascending.
        previous_reverse = self._sort_state.get(col_id)
        reverse = not previous_reverse if col_id in self._sort_state else False
        self._sort_state = {col_id: reverse}
        rows = list(self._row_data.values())
        try:
            rows.sort(key=lambda r: col.sort_key(r.get(col.id, "")), reverse=reverse)
        except TypeError:
            rows.sort(key=lambda r: str(r.get(col.id, "")), reverse=reverse)
        self.set_rows(rows)
        # Visual hint: prepend arrow to heading.
        for c in self._columns:
            label = c.label
            if c.id == col_id:
                arrow = " ▼" if reverse else " ▲"
                label += arrow
            self._tree.heading(c.id, text=label)

    def _handle_select(self, _event: object) -> None:
        if self._on_select_cb is not None:
            self._on_select_cb(self.get_selected())

    def _handle_activate(self, _event: object) -> None:
        if self._on_activate_cb is None:
            return
        selected = self.get_selected()
        if selected:
            self._on_activate_cb(selected[0])
