"""Modal confirmation dialog. Blocks the caller until user choice."""

from __future__ import annotations

import tkinter as tk

import customtkinter as ctk

from app.config.theme import (
    RADIUS_LG,
    SPACE_LG,
    SPACE_MD,
    get_font,
    palette_pair,
)
from app.i18n.fr import t


def confirm(
    parent: tk.Misc,
    *,
    title: str,
    message: str,
    destructive: bool = False,
    ok_label: str | None = None,
    cancel_label: str | None = None,
) -> bool:
    """Show a modal Yes/No dialog. Returns ``True`` if the user confirmed.

    *destructive=True* renders the OK button in error red and labels it
    "Supprimer" by default.
    """
    ok_label = ok_label or (t("dialog.delete") if destructive else t("dialog.ok"))
    cancel_label = cancel_label or t("dialog.cancel")

    dialog = ctk.CTkToplevel(parent)
    dialog.title(title or t("dialog.confirm_default_title"))
    dialog.transient(parent.winfo_toplevel())
    dialog.resizable(False, False)
    dialog.configure(fg_color=palette_pair("bg"))
    # Phase G (2026-05-16) — propage l'icône ShutterstockAnalyzer sur
    # la fenêtre de confirmation, comme sur les autres modales. On
    # remonte au winfo_toplevel() (la fenêtre App) puis on appelle son
    # helper ``_apply_modal_icon`` si présent. Aucun import circulaire :
    # on utilise getattr + try/except pour rester découplé du module
    # ``app.app``.
    top = parent.winfo_toplevel()
    apply_icon = getattr(top, "_apply_modal_icon", None)
    if callable(apply_icon):
        try:
            apply_icon(dialog)
        except Exception:
            pass

    body = ctk.CTkFrame(
        dialog,
        fg_color=palette_pair("bg_elevated"),
        corner_radius=RADIUS_LG,
        border_color=palette_pair("border"),
        border_width=1,
    )
    body.pack(fill="both", expand=True, padx=SPACE_LG, pady=SPACE_LG)

    ctk.CTkLabel(
        body,
        text=title,
        font=get_font("h3"),
        text_color=palette_pair("fg"),
        anchor="w",
        justify="left",
    ).pack(fill="x", padx=SPACE_LG, pady=(SPACE_LG, SPACE_MD))

    ctk.CTkLabel(
        body,
        text=message,
        font=get_font("body"),
        text_color=palette_pair("fg_muted"),
        wraplength=380,
        justify="left",
        anchor="w",
    ).pack(fill="x", padx=SPACE_LG, pady=(0, SPACE_LG))

    btn_row = ctk.CTkFrame(body, fg_color="transparent")
    btn_row.pack(fill="x", padx=SPACE_LG, pady=(0, SPACE_LG))

    result = {"value": False}

    def _ok() -> None:
        result["value"] = True
        dialog.destroy()

    def _cancel() -> None:
        result["value"] = False
        dialog.destroy()

    ok_color = palette_pair("error") if destructive else palette_pair("accent")
    ok_hover = palette_pair("error") if destructive else palette_pair("accent_hover")
    ok_btn = ctk.CTkButton(
        btn_row,
        text=ok_label,
        fg_color=ok_color,
        hover_color=ok_hover,
        text_color=palette_pair("accent_fg"),
        font=get_font("body_strong"),
        height=32,
        command=_ok,
    )
    ok_btn.pack(side="right", padx=(SPACE_MD, 0))

    cancel_btn = ctk.CTkButton(
        btn_row,
        text=cancel_label,
        fg_color="transparent",
        hover_color=palette_pair("bg_hover"),
        text_color=palette_pair("fg"),
        border_width=1,
        border_color=palette_pair("border_strong"),
        font=get_font("body"),
        height=32,
        command=_cancel,
    )
    cancel_btn.pack(side="right")

    dialog.bind("<Return>", lambda _e: _ok())
    dialog.bind("<Escape>", lambda _e: _cancel())
    cancel_btn.focus_set()

    _center_over_parent(dialog, parent)
    try:
        dialog.grab_set()
    except tk.TclError:
        pass
    parent.wait_window(dialog)
    return result["value"]


def _center_over_parent(dialog: ctk.CTkToplevel, parent: tk.Misc) -> None:
    parent.update_idletasks()
    try:
        px = parent.winfo_rootx()
        py = parent.winfo_rooty()
        pw = parent.winfo_width()
        ph = parent.winfo_height()
        dialog.update_idletasks()
        dw = dialog.winfo_reqwidth()
        dh = dialog.winfo_reqheight()
        x = max(0, px + (pw - dw) // 2)
        y = max(0, py + (ph - dh) // 3)
        dialog.geometry(f"+{x}+{y}")
    except tk.TclError:
        pass
