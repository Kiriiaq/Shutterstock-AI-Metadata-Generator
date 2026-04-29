"""WorkspaceView — single-screen dense atelier.

Replaces the multi-step workflow (sources → analyze → editor) with one
simultaneous view. Every option lives next to its tools, every advanced
system state is visible without a click. Detail pages (settings, audit,
ai_control, validate, upload) remain accessible via the SystemPanel
quick-action buttons or the sidebar — but the atelier alone covers
the production loop end-to-end.

Layout (2 columns, left ≈ 2/3, right ≈ 1/3):
    LEFT  : Sources panel  (stack)  + Editor panel (stack) + Analyze panel
    RIGHT : SystemPanel (live status, stats, audit tail, quick actions)
"""

from __future__ import annotations

import csv
import logging
import threading
from pathlib import Path
from tkinter import filedialog
from typing import TYPE_CHECKING, Any

import customtkinter as ctk

from app.components.data_table import Column, DataTable
from app.components.system_panel import SystemPanel
from app.config.theme import (
    RADIUS_MD,
    SPACE_MD,
    SPACE_SM,
    SPACE_XS,
    get_color,
    get_font,
)
from app.utils.formatters import fmt_int, fmt_size
from app.views.base_view import BaseView

if TYPE_CHECKING:
    from app.app import App

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class WorkspaceView(BaseView):
    view_id = "home"

    def __init__(self, master: ctk.CTkFrame, *, app: "App") -> None:
        super().__init__(master)
        self.app = app
        self._scanned: list[dict[str, Any]] = []
        self._current_path: Path | None = None
        self._processing = False
        self._iptc_fields: dict[str, ctk.CTkEntry | ctk.CTkTextbox] = {}
        self._build()

    # ------------------------------------------------------------------

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1, minsize=380)
        self.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(self, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(SPACE_MD, SPACE_SM), pady=SPACE_MD)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=2)
        left.grid_rowconfigure(1, weight=2)
        left.grid_rowconfigure(2, weight=2)

        self._build_sources_panel(left, row=0)
        self._build_editor_panel(left, row=1)
        self._build_analyze_panel(left, row=2)

        self.system_panel = SystemPanel(self, app=self.app)
        self.system_panel.grid(row=0, column=1, sticky="nsew", padx=(SPACE_SM, SPACE_MD), pady=SPACE_MD)

    # ------------------------------------------------------------------
    # Sources panel

    def _build_sources_panel(self, parent: ctk.CTkFrame, row: int) -> None:
        section = self._panel(parent, row, "SOURCES & TRI")
        section.grid_rowconfigure(2, weight=1)

        bar = ctk.CTkFrame(section, fg_color="transparent")
        bar.grid(row=1, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))
        bar.grid_columnconfigure(0, weight=1)

        self._folder_entry = ctk.CTkEntry(bar, font=get_font("body"), placeholder_text="Dossier source…")
        self._folder_entry.grid(row=0, column=0, sticky="ew", padx=(0, SPACE_XS))
        ctk.CTkButton(bar, text="…", width=36, command=self._browse).grid(row=0, column=1, padx=2)
        self._scan_btn = ctk.CTkButton(
            bar,
            text="Scanner",
            width=90,
            fg_color=get_color("accent"),
            hover_color=get_color("accent_hover"),
            text_color=get_color("accent_fg"),
            command=self._scan,
        )
        self._scan_btn.grid(row=0, column=2, padx=(2, 0))

        self._recursive_var = ctk.BooleanVar(value=True)
        opts = ctk.CTkFrame(section, fg_color="transparent")
        opts.grid(row=2, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))
        ctk.CTkCheckBox(opts, text="Récursif", variable=self._recursive_var, font=get_font("body")).pack(side="left")
        self._sources_status = ctk.CTkLabel(
            opts, text="Aucun scan", font=get_font("small"), text_color=get_color("fg_muted")
        )
        self._sources_status.pack(side="right")

        self._sources_table = DataTable(
            section,
            columns=[
                Column(id="name", label="Fichier", width=240),
                Column(id="size", label="Taille", width=70, anchor="e"),
                Column(id="dim", label="Dim.", width=80, anchor="center"),
                Column(id="meta", label="Méta", width=50, anchor="center"),
            ],
            select_mode="extended",
        )
        self._sources_table.grid(row=3, column=0, sticky="nsew", padx=SPACE_SM, pady=(0, SPACE_SM))
        self._sources_table.on_select(self._on_sources_select)
        self._sources_table.on_activate(self._on_sources_activate)
        section.grid_rowconfigure(3, weight=1)

    def _browse(self) -> None:
        path = filedialog.askdirectory(title="Choisir un dossier")
        if path:
            self._folder_entry.delete(0, "end")
            self._folder_entry.insert(0, path)
            self._scan()

    def _scan(self) -> None:
        folder = self._folder_entry.get().strip()
        if not folder or not Path(folder).is_dir():
            self.app.toasts.show("Dossier introuvable.", kind="error")
            return
        self._scan_btn.configure(state="disabled", text="Scan…")
        self._sources_status.configure(text="Recherche…", text_color=get_color("warning"))
        threading.Thread(target=self._scan_worker, args=(Path(folder),), daemon=True).start()

    def _scan_worker(self, folder: Path) -> None:
        try:
            from PIL import Image

            from src.modules.workers.worker_pool import collect_image_files

            files = collect_image_files(folder, recursive=self._recursive_var.get(), extensions=list(SUPPORTED_EXTS))
            api = self.app.api
            reader = api.metadata_reader if api else None
            rows: list[dict[str, Any]] = []
            for f in files:
                row: dict[str, Any] = {
                    "_path": f,
                    "name": f.name,
                    "size": fmt_size(f.stat().st_size),
                    "dim": "—",
                    "meta": "—",
                }
                try:
                    with Image.open(f) as im:
                        row["dim"] = f"{im.width}×{im.height}"
                except Exception:
                    pass
                if reader is not None:
                    try:
                        row["meta"] = "Oui" if reader.get_quick_info(f) else "Non"
                    except Exception:
                        row["meta"] = "?"
                rows.append(row)
            self.after(0, lambda r=rows, fld=folder: self._on_scan_complete(r, fld))
        except Exception as e:
            logger.exception("Scan failed")
            self.after(0, lambda err=str(e): self._on_scan_failed(err))

    def _on_scan_complete(self, rows: list[dict[str, Any]], folder: Path) -> None:
        self._scan_btn.configure(state="normal", text="Scanner")
        self._scanned = rows
        self._sources_table.set_rows(rows)
        self._sources_status.configure(
            text=f"{fmt_int(len(rows))} images dans {folder.name}",
            text_color=get_color("success") if rows else get_color("warning"),
        )
        self.app.app_state.set("source_folder", folder)
        self.app.app_state.set("scanned_images", [r["_path"] for r in rows])
        self._update_selection_summary()

    def _on_scan_failed(self, err: str) -> None:
        self._scan_btn.configure(state="normal", text="Scanner")
        self._sources_status.configure(text=f"Erreur : {err}", text_color=get_color("error"))

    def _on_sources_select(self, selected: list[dict[str, Any]]) -> None:
        self._update_selection_summary()
        self.app.app_state.set("selected_paths", [r["_path"] for r in selected])

    def _on_sources_activate(self, row: dict[str, Any]) -> None:
        path = row.get("_path")
        if isinstance(path, Path):
            self._select_for_edit(path)

    def _update_selection_summary(self) -> None:
        n_total = len(self._scanned)
        n_sel = len(self._sources_table.get_selected())
        if n_total == 0:
            self._analyze_summary.configure(text="0 image sélectionnée")
        else:
            self._analyze_summary.configure(text=f"{fmt_int(n_sel)} / {fmt_int(n_total)} images sélectionnées")

    # ------------------------------------------------------------------
    # Editor panel

    def _build_editor_panel(self, parent: ctk.CTkFrame, row: int) -> None:
        section = self._panel(parent, row, "APERÇU & ÉDITION IPTC")
        section.grid_columnconfigure(1, weight=1)
        section.grid_rowconfigure(1, weight=1)

        # Left: file label + thumbnail placeholder
        left = ctk.CTkFrame(section, fg_color="transparent")
        left.grid(row=1, column=0, sticky="nw", padx=SPACE_SM, pady=(0, SPACE_SM))
        self._editor_path_label = ctk.CTkLabel(
            left,
            text="(double-cliquez sur une image dans Sources)",
            font=get_font("small"),
            text_color=get_color("fg_muted"),
            anchor="w",
            justify="left",
            wraplength=180,
        )
        self._editor_path_label.pack(anchor="w")

        # Right: form rows
        form = ctk.CTkFrame(section, fg_color="transparent")
        form.grid(row=1, column=1, sticky="nsew", padx=SPACE_SM, pady=(0, SPACE_SM))
        form.grid_columnconfigure(1, weight=1)

        for r, (key, label) in enumerate(
            [
                ("headline", "Titre"),
                ("caption", "Description"),
                ("keywords", "Mots-clés"),
                ("byline", "Auteur"),
                ("copyright_notice", "Copyright"),
            ]
        ):
            ctk.CTkLabel(
                form, text=label, font=get_font("small"), text_color=get_color("fg_muted"), width=80, anchor="w"
            ).grid(row=r, column=0, sticky="w", padx=(0, SPACE_XS), pady=1)
            entry = ctk.CTkEntry(form, font=get_font("body"), height=24)
            entry.grid(row=r, column=1, sticky="ew", pady=1)
            self._iptc_fields[key] = entry

        actions = ctk.CTkFrame(section, fg_color="transparent")
        actions.grid(row=2, column=0, columnspan=2, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_SM))
        ctk.CTkButton(actions, text="Lire", width=60, height=26, command=self._editor_read).pack(
            side="left", padx=(0, SPACE_XS)
        )
        ctk.CTkButton(
            actions,
            text="Écrire",
            width=80,
            height=26,
            fg_color=get_color("accent"),
            hover_color=get_color("accent_hover"),
            text_color=get_color("accent_fg"),
            command=self._editor_write,
        ).pack(side="left", padx=SPACE_XS)
        ctk.CTkButton(actions, text="Effacer", width=70, height=26, command=self._editor_clear).pack(
            side="left", padx=SPACE_XS
        )
        self._editor_status = ctk.CTkLabel(actions, text="", font=get_font("small"), text_color=get_color("fg_muted"))
        self._editor_status.pack(side="right")

    def _select_for_edit(self, path: Path) -> None:
        self._current_path = path
        self._editor_path_label.configure(text=path.name, text_color=get_color("fg"))
        self._editor_read()

    def _editor_read(self) -> None:
        if self._current_path is None:
            self.app.toasts.show("Sélectionnez d'abord un fichier.", kind="warning")
            return
        api = self.app.api
        if api is None or api.metadata_reader is None:
            self._editor_status.configure(text="ExifTool absent", text_color=get_color("warning"))
            return
        try:
            metadata = api.read_metadata(self._current_path)
        except Exception:
            logger.exception("read_metadata failed")
            self._editor_status.configure(text="Lecture échouée", text_color=get_color("error"))
            return
        if metadata is None:
            self._editor_clear()
            return
        iptc = metadata.iptc
        self._set_field("headline", iptc.headline or iptc.object_name or "")
        self._set_field("caption", iptc.caption or "")
        self._set_field("keywords", ", ".join(iptc.keywords or []))
        self._set_field("byline", iptc.byline or "")
        self._set_field("copyright_notice", iptc.copyright_notice or "")
        self._editor_status.configure(text="Lu", text_color=get_color("success"))

    def _editor_write(self) -> None:
        if self._current_path is None:
            self.app.toasts.show("Sélectionnez d'abord un fichier.", kind="warning")
            return
        api = self.app.api
        if api is None or api.metadata_writer is None:
            self._editor_status.configure(text="ExifTool absent", text_color=get_color("warning"))
            return
        from src.modules.models.metadata_models import IPTCFields

        kw = [k.strip() for k in self._get_field("keywords").split(",") if k.strip()]
        iptc = IPTCFields(
            headline=self._get_field("headline") or None,
            caption=self._get_field("caption") or None,
            keywords=kw,
            byline=self._get_field("byline") or None,
            copyright_notice=self._get_field("copyright_notice") or None,
        )
        try:
            ok = api.write_metadata(self._current_path, iptc=iptc)
        except Exception:
            logger.exception("write_metadata failed")
            self._editor_status.configure(text="Écriture échouée", text_color=get_color("error"))
            return
        if ok:
            self._editor_status.configure(text="Écrit", text_color=get_color("success"))
            self.app.toasts.show(f"Métadonnées écrites : {self._current_path.name}", kind="success")
        else:
            self._editor_status.configure(text="Échec", text_color=get_color("error"))

    def _editor_clear(self) -> None:
        for key in self._iptc_fields:
            self._set_field(key, "")
        self._editor_status.configure(text="Effacé", text_color=get_color("fg_muted"))

    def _set_field(self, key: str, value: str) -> None:
        widget = self._iptc_fields[key]
        if isinstance(widget, ctk.CTkEntry):
            widget.delete(0, "end")
            widget.insert(0, value)

    def _get_field(self, key: str) -> str:
        widget = self._iptc_fields[key]
        if isinstance(widget, ctk.CTkEntry):
            return widget.get()
        return ""

    # ------------------------------------------------------------------
    # Analyze panel

    def _build_analyze_panel(self, parent: ctk.CTkFrame, row: int) -> None:
        section = self._panel(parent, row, "ANALYSE IA")
        section.grid_rowconfigure(3, weight=1)
        section.grid_columnconfigure(0, weight=1)

        opts = ctk.CTkFrame(section, fg_color="transparent")
        opts.grid(row=1, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))

        self._skip_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(opts, text="Ignorer si méta", variable=self._skip_var, font=get_font("body")).pack(side="left")
        self._write_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(opts, text="Écrire les résultats", variable=self._write_var, font=get_font("body")).pack(
            side="left", padx=SPACE_MD
        )
        self._analyze_summary = ctk.CTkLabel(
            opts, text="0 image sélectionnée", font=get_font("small"), text_color=get_color("fg_muted")
        )
        self._analyze_summary.pack(side="right")

        controls = ctk.CTkFrame(section, fg_color="transparent")
        controls.grid(row=2, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))
        controls.grid_columnconfigure(2, weight=1)

        self._start_btn = ctk.CTkButton(
            controls,
            text="Démarrer",
            width=110,
            height=28,
            fg_color=get_color("accent"),
            hover_color=get_color("accent_hover"),
            text_color=get_color("accent_fg"),
            font=get_font("body_strong"),
            command=self._analyze_start,
        )
        self._start_btn.grid(row=0, column=0, padx=(0, SPACE_XS))
        self._stop_btn = ctk.CTkButton(
            controls,
            text="Arrêter",
            width=80,
            height=28,
            fg_color=get_color("error"),
            text_color="#FFFFFF",
            state="disabled",
            command=self._analyze_stop,
        )
        self._stop_btn.grid(row=0, column=1, padx=SPACE_XS)
        self._analyze_progress = ctk.CTkProgressBar(controls)
        self._analyze_progress.set(0)
        self._analyze_progress.grid(row=0, column=2, sticky="ew", padx=SPACE_MD)
        self._analyze_status = ctk.CTkLabel(
            controls, text="Prêt", font=get_font("small"), text_color=get_color("fg_muted")
        )
        self._analyze_status.grid(row=0, column=3, padx=(SPACE_XS, 0), sticky="e")

        self._analyze_results = ctk.CTkTextbox(
            section,
            font=get_font("code"),
            fg_color=get_color("bg"),
            text_color=get_color("fg"),
            border_color=get_color("border"),
            border_width=1,
            corner_radius=RADIUS_MD,
        )
        self._analyze_results.grid(row=3, column=0, sticky="nsew", padx=SPACE_SM, pady=(0, SPACE_SM))
        self._analyze_results.insert("1.0", "Les résultats apparaîtront ici en temps réel.\n")
        self._analyze_results.configure(state="disabled")

    def _analyze_start(self) -> None:
        api = self.app.api
        if api is None:
            self.app.toasts.show("Backend indisponible.", kind="error")
            return
        selected = list(self.app.app_state.get("selected_paths") or [])
        if not selected:
            self.app.toasts.show("Aucune image sélectionnée.", kind="warning")
            return
        self._processing = True
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._analyze_progress.set(0)
        self._set_analyze_results("Initialisation…\n")
        threading.Thread(target=self._analyze_worker, args=(api, selected), daemon=True).start()

    def _analyze_worker(self, api: Any, selected: list[Path]) -> None:
        try:

            def on_progress(done: int, total: int, current: str) -> None:
                self.after(0, lambda: self._analyze_on_progress(done, total, current))

            def on_result(res: Any) -> None:
                self.after(0, lambda r=res: self._analyze_on_result(r))

            result = api.analyze_batch_ai(
                selected,
                skip_if_has_metadata=self._skip_var.get(),
                write_metadata=self._write_var.get(),
                on_progress=on_progress,
                on_result=on_result,
            )
            self.after(0, lambda r=result: self._analyze_on_complete(r))
        except Exception as e:
            logger.exception("Analyze worker failed")
            self.after(0, lambda err=str(e): self._analyze_on_failed(err))

    def _analyze_stop(self) -> None:
        api = self.app.api
        analyzer = getattr(api, "vision_analyzer", None) if api else None
        cancel = getattr(analyzer, "cancel", None)
        if callable(cancel):
            cancel()
        self._analyze_status.configure(text="Arrêt…", text_color=get_color("warning"))

    def _analyze_on_progress(self, done: int, total: int, current: str) -> None:
        if total > 0:
            self._analyze_progress.set(done / total)
        self._analyze_status.configure(
            text=f"{fmt_int(done)} / {fmt_int(total)} — {Path(current).name if current else ''}",
            text_color=get_color("fg"),
        )

    def _analyze_on_result(self, res: Any) -> None:
        if isinstance(res, dict):
            ok = bool(res.get("success", True))
            path = res.get("file_path", "")
        else:
            ok = bool(getattr(res, "success", True))
            path = getattr(res, "file_path", "")
        symbol = "✓" if ok else "✗"
        self._append_analyze_results(f"{symbol} {Path(path).name}\n")

    def _analyze_on_complete(self, result: dict[str, Any]) -> None:
        self._processing = False
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._analyze_progress.set(1)
        completed = result.get("completed", 0)
        failed = result.get("failed", 0)
        skipped = result.get("skipped", 0)
        self._append_analyze_results(
            f"\n— Terminé : {fmt_int(completed)} succès · {fmt_int(failed)} échecs · {fmt_int(skipped)} ignorés.\n"
        )
        self._analyze_status.configure(text="Terminé", text_color=get_color("success"))
        self.app.toasts.show(f"Analyse terminée — {fmt_int(completed)} succès.", kind="success")

    def _analyze_on_failed(self, err: str) -> None:
        self._processing = False
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._analyze_status.configure(text="Erreur", text_color=get_color("error"))
        self._append_analyze_results(f"\nERREUR : {err}\n")

    def _set_analyze_results(self, text: str) -> None:
        self._analyze_results.configure(state="normal")
        self._analyze_results.delete("1.0", "end")
        self._analyze_results.insert("1.0", text)
        self._analyze_results.configure(state="disabled")

    def _append_analyze_results(self, text: str) -> None:
        self._analyze_results.configure(state="normal")
        self._analyze_results.insert("end", text)
        self._analyze_results.see("end")
        self._analyze_results.configure(state="disabled")

    # ------------------------------------------------------------------
    # Helpers

    def _panel(self, parent: ctk.CTkFrame, row: int, title: str) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(
            parent,
            fg_color=get_color("bg_elevated"),
            border_color=get_color("border"),
            border_width=1,
            corner_radius=RADIUS_MD,
        )
        frame.grid(row=row, column=0, sticky="nsew", pady=(0, SPACE_SM))
        frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            frame,
            text=title,
            font=get_font("small"),
            text_color=get_color("fg_subtle"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=SPACE_SM, pady=(SPACE_SM, SPACE_XS))
        return frame

    # ------------------------------------------------------------------
    # Lifecycle

    def on_enter(self, **_kwargs: Any) -> None:
        self.system_panel.start_auto_refresh()

    def on_leave(self) -> None:
        self.system_panel.stop_auto_refresh()


# Re-export csv to silence an unused-import lint when typing is later enabled.
_ = csv
