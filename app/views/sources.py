"""Sources view — folder picker + recursive scan + selectable table."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from tkinter import filedialog
from typing import TYPE_CHECKING, Any

import customtkinter as ctk

from app.components.data_table import Column, DataTable
from app.components.empty_state import EmptyState
from app.config.theme import (
    RADIUS_MD,
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    SPACE_XL,
    get_color,
    get_font,
)
from app.utils.formatters import fmt_int, fmt_size
from app.views.base_view import BaseView

if TYPE_CHECKING:
    from app.app import App

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class SourcesView(BaseView):
    view_id = "sources"

    def __init__(self, master: ctk.CTkFrame, *, app: "App") -> None:
        super().__init__(master)
        self.app = app
        self._scanning = False
        self._rows: list[dict[str, Any]] = []
        self._build()

    # ------------------------------------------------------------------

    def _build(self) -> None:
        wrapper = ctk.CTkFrame(self, fg_color="transparent")
        wrapper.grid(row=0, column=0, sticky="nsew", padx=SPACE_XL, pady=SPACE_XL)
        wrapper.grid_columnconfigure(0, weight=1)
        wrapper.grid_rowconfigure(2, weight=1)

        # Header
        ctk.CTkLabel(
            wrapper,
            text="Sources et tri",
            font=get_font("h1"),
            text_color=get_color("fg"),
        ).grid(row=0, column=0, sticky="w", pady=(0, SPACE_LG))

        self._build_picker(wrapper, row=1)
        self._build_table(wrapper, row=2)
        self._build_footer(wrapper, row=3)

    def _build_picker(self, parent: ctk.CTkFrame, row: int) -> None:
        bar = ctk.CTkFrame(parent, fg_color=get_color("bg_elevated"), corner_radius=RADIUS_MD)
        bar.grid(row=row, column=0, sticky="ew", pady=(0, SPACE_MD))
        bar.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(bar, text="Dossier :", font=get_font("body_strong"), text_color=get_color("fg")).grid(
            row=0, column=0, padx=(SPACE_LG, SPACE_SM), pady=SPACE_MD
        )
        self._folder_entry = ctk.CTkEntry(bar, font=get_font("body"))
        self._folder_entry.grid(row=0, column=1, sticky="ew", pady=SPACE_MD)

        ctk.CTkButton(bar, text="Parcourir", width=110, command=self._browse).grid(
            row=0, column=2, padx=SPACE_SM, pady=SPACE_MD
        )
        self._scan_btn = ctk.CTkButton(
            bar,
            text="Scanner",
            width=110,
            fg_color=get_color("accent"),
            hover_color=get_color("accent_hover"),
            text_color=get_color("accent_fg"),
            command=self._scan,
        )
        self._scan_btn.grid(row=0, column=3, padx=(SPACE_SM, SPACE_LG), pady=SPACE_MD)

        self._recursive_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(bar, text="Inclure les sous-dossiers", variable=self._recursive_var).grid(
            row=1, column=1, sticky="w", padx=0, pady=(0, SPACE_MD)
        )
        self._status = ctk.CTkLabel(bar, text="", font=get_font("small"), text_color=get_color("fg_muted"))
        self._status.grid(row=1, column=2, columnspan=2, sticky="e", padx=SPACE_LG, pady=(0, SPACE_MD))

    def _build_table(self, parent: ctk.CTkFrame, row: int) -> None:
        self._table_container = ctk.CTkFrame(parent, fg_color="transparent")
        self._table_container.grid(row=row, column=0, sticky="nsew", pady=(0, SPACE_MD))
        self._table_container.grid_columnconfigure(0, weight=1)
        self._table_container.grid_rowconfigure(0, weight=1)
        self._render_empty()

    def _render_empty(self) -> None:
        for child in self._table_container.winfo_children():
            child.destroy()
        EmptyState(
            self._table_container,
            icon="📁",
            title="Aucun dossier scanné",
            subtitle="Sélectionnez un dossier puis cliquez sur Scanner pour commencer.",
        ).grid(row=0, column=0, sticky="nsew")

    def _render_table(self) -> None:
        for child in self._table_container.winfo_children():
            child.destroy()
        self._table = DataTable(
            self._table_container,
            columns=[
                Column(id="name", label="Fichier", width=300),
                Column(id="size", label="Taille", width=100, anchor="e", sort_key=lambda v: int(v or 0)),
                Column(id="dimensions", label="Dimensions", width=120, anchor="center"),
                Column(id="metadata", label="Métadonnées", width=120, anchor="center"),
            ],
        )
        self._table.grid(row=0, column=0, sticky="nsew")
        rendered = [
            {
                "name": r["path"].name,
                "size": r["size_bytes"],
                "size_display": fmt_size(r["size_bytes"]),
                "dimensions": r.get("dimensions", "—"),
                "metadata": "Oui" if r.get("has_metadata") else "Non",
                "_path": r["path"],
            }
            for r in self._rows
        ]
        for row in rendered:
            row["size"] = row.pop("size_display")
        self._table.set_rows(rendered)
        self._table.on_select(self._on_table_selection)

    def _build_footer(self, parent: ctk.CTkFrame, row: int) -> None:
        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.grid(row=row, column=0, sticky="ew")

        self._count_label = ctk.CTkLabel(
            bar, text="Aucune sélection", font=get_font("body"), text_color=get_color("fg_muted")
        )
        self._count_label.pack(side="left")

        self._next_btn = ctk.CTkButton(
            bar,
            text="Étape suivante",
            font=get_font("body_strong"),
            fg_color=get_color("accent"),
            hover_color=get_color("accent_hover"),
            text_color=get_color("accent_fg"),
            state="disabled",
            command=self._goto_analyze,
        )
        self._next_btn.pack(side="right")

    # ------------------------------------------------------------------

    def _browse(self) -> None:
        path = filedialog.askdirectory(title="Choisir un dossier")
        if path:
            self._folder_entry.delete(0, "end")
            self._folder_entry.insert(0, path)

    def _scan(self) -> None:
        folder = self._folder_entry.get().strip()
        if not folder or not Path(folder).is_dir():
            self.app.toasts.show("Dossier introuvable.", kind="error")
            return
        if self._scanning:
            return
        self._scanning = True
        self._scan_btn.configure(state="disabled", text="Scan en cours…")
        self._status.configure(text="Recherche des images…", text_color=get_color("warning"))
        threading.Thread(target=self._scan_worker, args=(Path(folder),), daemon=True).start()

    def _scan_worker(self, folder: Path) -> None:
        try:
            from PIL import Image

            from src.modules.workers.worker_pool import collect_image_files

            files = collect_image_files(folder, recursive=self._recursive_var.get(), extensions=list(SUPPORTED_EXTS))
            api = self.app.api
            reader = api.metadata_reader if api else None

            rows = []
            for f in files:
                row = {"path": f, "size_bytes": f.stat().st_size, "has_metadata": False, "dimensions": "—"}
                try:
                    with Image.open(f) as im:
                        row["dimensions"] = f"{im.width}×{im.height}"
                except Exception:
                    pass
                if reader is not None:
                    try:
                        meta = reader.get_quick_info(f)
                        row["has_metadata"] = bool(meta)
                    except Exception:
                        pass
                rows.append(row)
            self.after(0, lambda: self._on_scan_complete(rows, folder))
        except Exception as e:
            logger.exception("Scan failed")
            self.after(0, lambda err=str(e): self._on_scan_failed(err))

    def _on_scan_complete(self, rows: list[dict[str, Any]], folder: Path) -> None:
        self._scanning = False
        self._scan_btn.configure(state="normal", text="Scanner")
        self._rows = rows
        if rows:
            self._status.configure(
                text=f"{fmt_int(len(rows))} images détectées dans {folder.name}",
                text_color=get_color("success"),
            )
            self._render_table()
            self.app.app_state.set("source_folder", folder)
            self.app.app_state.set("scanned_images", [r["path"] for r in rows])
        else:
            self._render_empty()
            self._status.configure(text="Aucune image trouvée.", text_color=get_color("warning"))
        self._update_count()

    def _on_scan_failed(self, error: str) -> None:
        self._scanning = False
        self._scan_btn.configure(state="normal", text="Scanner")
        self._status.configure(text=f"Erreur : {error}", text_color=get_color("error"))
        self.app.toasts.show(f"Échec du scan : {error}", kind="error")

    def _on_table_selection(self, _selected: list[dict[str, Any]]) -> None:
        self._update_count()

    def _update_count(self) -> None:
        if not hasattr(self, "_table"):
            self._count_label.configure(text="Aucune sélection")
            self._next_btn.configure(state="disabled")
            return
        selected = self._table.get_selected()
        n = len(selected)
        if n == 0:
            self._count_label.configure(text=f"{fmt_int(len(self._rows))} images détectées")
            self._next_btn.configure(state="disabled")
        else:
            self._count_label.configure(text=f"{fmt_int(n)} sélectionnée(s) sur {fmt_int(len(self._rows))}")
            self._next_btn.configure(state="normal")
            self.app.app_state.set("selected_paths", [r["_path"] for r in selected])

    def _goto_analyze(self) -> None:
        self.app.router.navigate_to("analyze")
