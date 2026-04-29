"""WorkspaceView — single-screen Atelier with 8 tool panels.

Design rule: per tool, one panel with its indicators and its actions
visible at all times. No duplicated entry-point. Deep editing surfaces
(full settings form, full audit table, etc.) open as modals from the
corresponding panel's "Détail…" button — exactly one path per tool.

Layout (1400×900 fits comfortably; resizes well):
    LEFT col (≈ 60 %)               RIGHT col (≈ 40 %)
    ─────────────────────           ─────────────────────
    SOURCES & TRI         (big)     MODÈLE IA            (compact)
    ÉDITION IPTC          (med)     VALIDATION           (compact)
    ANALYSE IA            (med)     HISTORIQUE           (compact)
                                    PARAMÈTRES           (compact)
                                    TÉLÉVERSEMENT        (compact stub)
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import filedialog
from typing import TYPE_CHECKING, Any

import customtkinter as ctk

from app.components.data_table import Column, DataTable
from app.config.theme import (
    RADIUS_MD,
    SPACE_LG,
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
REFRESH_INTERVAL_MS = 5000
HISTORY_TAIL = 5


class WorkspaceView(BaseView):
    view_id = "home"

    def __init__(self, master: ctk.CTkFrame, *, app: "App") -> None:
        super().__init__(master)
        self.app = app
        # State
        self._scanned: list[dict[str, Any]] = []
        self._current_path: Path | None = None
        self._processing = False
        self._refresh_after_id: str | None = None
        # Widgets
        self._iptc_fields: dict[str, ctk.CTkEntry] = {}
        self._build()

    # ------------------------------------------------------------------
    # Layout

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2, minsize=420)
        self.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(self, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(SPACE_MD, SPACE_SM), pady=SPACE_MD)
        left.grid_columnconfigure(0, weight=1)
        for r, w in enumerate([3, 2, 3]):
            left.grid_rowconfigure(r, weight=w)

        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(SPACE_SM, SPACE_MD), pady=SPACE_MD)
        right.grid_columnconfigure(0, weight=1)
        for r in range(5):
            right.grid_rowconfigure(r, weight=1)

        self._build_sources_panel(left, row=0)
        self._build_editor_panel(left, row=1)
        self._build_analyze_panel(left, row=2)

        self._build_model_panel(right, row=0)
        self._build_validate_panel(right, row=1)
        self._build_history_panel(right, row=2)
        self._build_settings_panel(right, row=3)
        self._build_upload_panel(right, row=4)

    # ==================================================================
    # LEFT COLUMN — production loop
    # ==================================================================

    # ----- Panel: Sources & tri ---------------------------------------

    def _build_sources_panel(self, parent: ctk.CTkFrame, row: int) -> None:
        section = self._panel(parent, row, "SOURCES & TRI")
        section.grid_rowconfigure(3, weight=1)

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

        opts = ctk.CTkFrame(section, fg_color="transparent")
        opts.grid(row=2, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))
        self._recursive_var = ctk.BooleanVar(value=True)
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
            text=f"{fmt_int(len(rows))} images · {folder.name}",
            text_color=get_color("success") if rows else get_color("warning"),
        )
        self.app.app_state.set("source_folder", folder)
        self.app.app_state.set("scanned_images", [r["_path"] for r in rows])
        self._update_selection_summary()
        self._validate_summary.configure(text=f"{fmt_int(len(rows))} images · non validées")

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
        self._analyze_summary.configure(
            text=f"{fmt_int(n_sel)} / {fmt_int(n_total)} sélectionnée(s)" if n_total else "Aucune image"
        )

    # ----- Panel: Édition IPTC ----------------------------------------

    def _build_editor_panel(self, parent: ctk.CTkFrame, row: int) -> None:
        section = self._panel(parent, row, "ÉDITION IPTC")
        section.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(section, fg_color="transparent")
        head.grid(row=1, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))
        head.grid_columnconfigure(0, weight=1)
        self._editor_path_label = ctk.CTkLabel(
            head,
            text="(double-cliquez sur une image dans Sources)",
            font=get_font("small"),
            text_color=get_color("fg_muted"),
            anchor="w",
        )
        self._editor_path_label.grid(row=0, column=0, sticky="ew")

        form = ctk.CTkFrame(section, fg_color="transparent")
        form.grid(row=2, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))
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
        actions.grid(row=3, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_SM))
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
        self._iptc_set("headline", iptc.headline or iptc.object_name or "")
        self._iptc_set("caption", iptc.caption or "")
        self._iptc_set("keywords", ", ".join(iptc.keywords or []))
        self._iptc_set("byline", iptc.byline or "")
        self._iptc_set("copyright_notice", iptc.copyright_notice or "")
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

        kw = [k.strip() for k in self._iptc_get("keywords").split(",") if k.strip()]
        iptc = IPTCFields(
            headline=self._iptc_get("headline") or None,
            caption=self._iptc_get("caption") or None,
            keywords=kw,
            byline=self._iptc_get("byline") or None,
            copyright_notice=self._iptc_get("copyright_notice") or None,
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
            self._iptc_set(key, "")
        self._editor_status.configure(text="Effacé", text_color=get_color("fg_muted"))

    def _iptc_set(self, key: str, value: str) -> None:
        widget = self._iptc_fields[key]
        widget.delete(0, "end")
        widget.insert(0, value)

    def _iptc_get(self, key: str) -> str:
        return self._iptc_fields[key].get()

    # ----- Panel: Analyse IA ------------------------------------------

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
            opts, text="Aucune image", font=get_font("small"), text_color=get_color("fg_muted")
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
            text_color=get_color("error_fg"),
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

    # ==================================================================
    # RIGHT COLUMN — system & control
    # ==================================================================

    # ----- Panel: Modèle IA -------------------------------------------

    def _build_model_panel(self, parent: ctk.CTkFrame, row: int) -> None:
        section = self._panel(parent, row, "MODÈLE IA")
        section.grid_columnconfigure(0, weight=1)

        body = ctk.CTkFrame(section, fg_color="transparent")
        body.grid(row=1, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_SM))
        body.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(body, text="Statut :", font=get_font("small"), text_color=get_color("fg_muted"), width=70).grid(
            row=0, column=0, sticky="w", pady=1
        )
        self._model_status_dot = ctk.CTkLabel(
            body, text="●", font=get_font("body_strong"), text_color=get_color("fg_subtle"), width=14
        )
        self._model_status_dot.grid(row=0, column=1, sticky="w")
        self._model_status_text = ctk.CTkLabel(
            body, text="—", font=get_font("body"), text_color=get_color("fg"), anchor="w"
        )
        self._model_status_text.grid(row=0, column=2, sticky="ew", padx=(SPACE_XS, 0))

        ctk.CTkLabel(body, text="URL :", font=get_font("small"), text_color=get_color("fg_muted"), width=70).grid(
            row=1, column=0, sticky="w", pady=1
        )
        self._model_url_label = ctk.CTkLabel(
            body, text="—", font=get_font("code"), text_color=get_color("fg"), anchor="w"
        )
        self._model_url_label.grid(row=1, column=1, columnspan=2, sticky="ew", pady=1)

        ctk.CTkLabel(body, text="Modèle :", font=get_font("small"), text_color=get_color("fg_muted"), width=70).grid(
            row=2, column=0, sticky="w", pady=1
        )
        self._model_name_label = ctk.CTkLabel(
            body, text="—", font=get_font("body_strong"), text_color=get_color("fg"), anchor="w"
        )
        self._model_name_label.grid(row=2, column=1, columnspan=2, sticky="ew", pady=1)

        actions = ctk.CTkFrame(section, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_SM))
        ctk.CTkButton(actions, text="Tester", width=90, height=26, command=self._model_test).pack(
            side="left", padx=(0, SPACE_XS)
        )
        ctk.CTkButton(
            actions,
            text="Configurer…",
            width=120,
            height=26,
            command=lambda: self.app.open_in_modal("ai_control"),
        ).pack(side="left", padx=SPACE_XS)
        self._model_test_msg = ctk.CTkLabel(actions, text="", font=get_font("small"), text_color=get_color("fg_muted"))
        self._model_test_msg.pack(side="right")

    def _model_test(self) -> None:
        api = self.app.api
        if api is None:
            self._model_test_msg.configure(text="Backend absent", text_color=get_color("warning"))
            return
        self._model_test_msg.configure(text="Test…", text_color=get_color("warning"))
        threading.Thread(target=self._model_test_worker, args=(api,), daemon=True).start()

    def _model_test_worker(self, api: Any) -> None:
        try:
            if not hasattr(api, "ollama_client"):
                api.init_ai()
            result = api.ollama_client.test_connection()
        except Exception as e:
            logger.exception("model test failed")
            self.after(
                0,
                lambda err=str(e): self._model_test_msg.configure(
                    text=f"Échec : {err[:30]}", text_color=get_color("error")
                ),
            )
            return
        if result.get("success"):
            ms = result.get("response_time_ms", 0)
            self.after(
                0, lambda m=ms: self._model_test_msg.configure(text=f"OK · {m} ms", text_color=get_color("success"))
            )
        else:
            self.after(
                0,
                lambda r=result: self._model_test_msg.configure(
                    text=f"Échec : {r.get('message', '')[:30]}", text_color=get_color("error")
                ),
            )

    # ----- Panel: Validation ------------------------------------------

    def _build_validate_panel(self, parent: ctk.CTkFrame, row: int) -> None:
        section = self._panel(parent, row, "VALIDATION")
        section.grid_columnconfigure(0, weight=1)

        self._validate_summary = ctk.CTkLabel(
            section,
            text="Aucun scan",
            font=get_font("body"),
            text_color=get_color("fg"),
            anchor="w",
        )
        self._validate_summary.grid(row=1, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))

        self._validate_detail = ctk.CTkLabel(
            section,
            text="—",
            font=get_font("small"),
            text_color=get_color("fg_muted"),
            anchor="w",
            wraplength=380,
            justify="left",
        )
        self._validate_detail.grid(row=2, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))

        actions = ctk.CTkFrame(section, fg_color="transparent")
        actions.grid(row=3, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_SM))
        ctk.CTkButton(
            actions,
            text="Lancer",
            width=90,
            height=26,
            fg_color=get_color("accent"),
            hover_color=get_color("accent_hover"),
            text_color=get_color("accent_fg"),
            command=self._validate_run,
        ).pack(side="left", padx=(0, SPACE_XS))
        ctk.CTkButton(
            actions,
            text="Détail…",
            width=90,
            height=26,
            command=lambda: self.app.open_in_modal("validate"),
        ).pack(side="left", padx=SPACE_XS)

    def _validate_run(self) -> None:
        api = self.app.api
        if api is None:
            self._validate_summary.configure(text="Backend indisponible", text_color=get_color("warning"))
            return
        files = list(self.app.app_state.get("scanned_images") or [])
        if not files:
            self._validate_summary.configure(text="Scannez d'abord un dossier", text_color=get_color("warning"))
            return
        self._validate_summary.configure(
            text=f"Validation de {fmt_int(len(files))} images…", text_color=get_color("warning")
        )
        threading.Thread(target=self._validate_worker, args=(api, files), daemon=True).start()

    def _validate_worker(self, api: Any, files: list[Path]) -> None:
        ok = ko = 0
        first_issue = ""
        for f in files:
            try:
                res = api.validate_image(f)
                if getattr(res, "is_valid", False):
                    ok += 1
                else:
                    ko += 1
                    if not first_issue:
                        errs = list(getattr(res, "errors", []))
                        first_issue = f"{f.name} : {errs[0] if errs else 'invalide'}"
            except Exception:
                ko += 1
        self.after(0, lambda: self._validate_done(ok, ko, first_issue))

    def _validate_done(self, ok: int, ko: int, first_issue: str) -> None:
        total = ok + ko
        if ko == 0 and total > 0:
            self._validate_summary.configure(
                text=f"{fmt_int(total)} images · toutes conformes ✓", text_color=get_color("success")
            )
        else:
            self._validate_summary.configure(
                text=f"{fmt_int(total)} images · {fmt_int(ok)} OK · {fmt_int(ko)} à corriger",
                text_color=get_color("warning") if ko else get_color("success"),
            )
        self._validate_detail.configure(text=first_issue or "Aucune anomalie")

    # ----- Panel: Historique ------------------------------------------

    def _build_history_panel(self, parent: ctk.CTkFrame, row: int) -> None:
        section = self._panel(parent, row, "HISTORIQUE")
        section.grid_columnconfigure(0, weight=1)
        section.grid_rowconfigure(2, weight=1)

        self._history_summary = ctk.CTkLabel(
            section, text="—", font=get_font("body"), text_color=get_color("fg"), anchor="w"
        )
        self._history_summary.grid(row=1, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))

        tail = ctk.CTkFrame(section, fg_color=get_color("bg"), corner_radius=RADIUS_MD)
        tail.grid(row=2, column=0, sticky="nsew", padx=SPACE_SM, pady=(0, SPACE_XS))
        tail.grid_columnconfigure(0, weight=1)
        self._history_lines: list[ctk.CTkLabel] = []
        for i in range(HISTORY_TAIL):
            label = ctk.CTkLabel(tail, text="", font=get_font("code"), text_color=get_color("fg_muted"), anchor="w")
            label.grid(row=i, column=0, sticky="ew", padx=SPACE_SM, pady=0)
            self._history_lines.append(label)

        actions = ctk.CTkFrame(section, fg_color="transparent")
        actions.grid(row=3, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_SM))
        ctk.CTkButton(
            actions,
            text="Tout voir…",
            width=100,
            height=26,
            command=lambda: self.app.open_in_modal("audit"),
        ).pack(side="left", padx=(0, SPACE_XS))
        ctk.CTkButton(
            actions,
            text="Exporter…",
            width=100,
            height=26,
            command=self._history_export,
        ).pack(side="left", padx=SPACE_XS)

    def _history_export(self) -> None:
        api = self.app.api
        if api is None:
            return
        path = filedialog.asksaveasfilename(
            title="Exporter le journal",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("CSV", "*.csv")],
        )
        if not path:
            return
        out = Path(path)
        try:
            count = api.database.export_audit_log(out, format="csv" if out.suffix.lower() == ".csv" else "json")
            self.app.toasts.show(f"{fmt_int(count)} entrée(s) exportée(s).", kind="success")
        except Exception as e:
            logger.exception("export failed")
            self.app.toasts.show(f"Échec : {e}", kind="error")

    # ----- Panel: Paramètres ------------------------------------------

    def _build_settings_panel(self, parent: ctk.CTkFrame, row: int) -> None:
        section = self._panel(parent, row, "PARAMÈTRES")
        section.grid_columnconfigure(0, weight=1)

        body = ctk.CTkFrame(section, fg_color="transparent")
        body.grid(row=1, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))
        body.grid_columnconfigure((0, 1), weight=1)
        self._settings_chips: dict[str, ctk.CTkLabel] = {}
        items = [
            ("workers", "Workers"),
            ("batch", "Batch"),
            ("model", "Modèle"),
            ("backup", "Backup _orig"),
            ("write_iptc", "IPTC"),
            ("write_xmp", "XMP"),
        ]
        for i, (key, label) in enumerate(items):
            row_f = ctk.CTkFrame(body, fg_color="transparent")
            row_f.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=1)
            ctk.CTkLabel(row_f, text=label, font=get_font("small"), text_color=get_color("fg_muted"), anchor="w").pack(
                side="left", padx=(0, SPACE_XS)
            )
            value = ctk.CTkLabel(row_f, text="—", font=get_font("body_strong"), text_color=get_color("fg"), anchor="w")
            value.pack(side="left")
            self._settings_chips[key] = value

        actions = ctk.CTkFrame(section, fg_color="transparent")
        actions.grid(row=2, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_SM))
        ctk.CTkButton(
            actions,
            text="Modifier…",
            width=120,
            height=26,
            command=lambda: self.app.open_in_modal("settings"),
        ).pack(side="left")

    # ----- Panel: Téléversement ---------------------------------------

    def _build_upload_panel(self, parent: ctk.CTkFrame, row: int) -> None:
        section = self._panel(parent, row, "TÉLÉVERSEMENT FTPS")
        section.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            section,
            text="⚠ Non implémenté — utilisez un client FTPS externe.",
            font=get_font("small"),
            text_color=get_color("warning"),
            anchor="w",
            wraplength=380,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))

        self._upload_host_label = ctk.CTkLabel(
            section, text="Host : —", font=get_font("code"), text_color=get_color("fg_muted"), anchor="w"
        )
        self._upload_host_label.grid(row=2, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_XS))

        actions = ctk.CTkFrame(section, fg_color="transparent")
        actions.grid(row=3, column=0, sticky="ew", padx=SPACE_SM, pady=(0, SPACE_SM))
        ctk.CTkButton(
            actions,
            text="Détail…",
            width=120,
            height=26,
            command=lambda: self.app.open_in_modal("upload"),
        ).pack(side="left")

    # ==================================================================
    # Auto-refresh of the right-column live indicators
    # ==================================================================

    def on_enter(self, **_kwargs: Any) -> None:
        self._refresh()

    def on_leave(self) -> None:
        if self._refresh_after_id is not None:
            try:
                self.after_cancel(self._refresh_after_id)
            except Exception:
                pass
            self._refresh_after_id = None

    def _refresh(self) -> None:
        self._refresh_settings_chips()
        self._refresh_upload_host()
        self._refresh_dynamic_async()
        self._refresh_after_id = self.after(REFRESH_INTERVAL_MS, self._refresh)

    def _refresh_settings_chips(self) -> None:
        api = self.app.api
        defaults = {
            "workers": 4,
            "batch": 50,
            "model": "—",
            "backup": True,
            "write_iptc": True,
            "write_xmp": True,
        }
        getter = (lambda k, d: api.get_setting(k, d)) if api else (lambda _k, d: d)
        self._settings_chips["workers"].configure(text=str(int(getter("max_workers", defaults["workers"]))))
        self._settings_chips["batch"].configure(text=str(int(getter("batch_size", defaults["batch"]))))
        self._settings_chips["model"].configure(text=str(getter("ollama_model", defaults["model"]))[:18])
        self._settings_chips["backup"].configure(
            text="Oui" if bool(getter("create_backup", defaults["backup"])) else "Non"
        )
        self._settings_chips["write_iptc"].configure(
            text="Oui" if bool(getter("write_iptc", defaults["write_iptc"])) else "Non"
        )
        self._settings_chips["write_xmp"].configure(
            text="Oui" if bool(getter("write_xmp", defaults["write_xmp"])) else "Non"
        )

    def _refresh_upload_host(self) -> None:
        api = self.app.api
        host = api.get_setting("ftps_host", "—") if api else "—"
        self._upload_host_label.configure(text=f"Host : {host}")

    def _refresh_dynamic_async(self) -> None:
        api = self.app.api
        if api is None:
            self._set_model_status("muted", "Backend absent", "—", "—")
            self._set_history_summary(0, 0, [])
            return
        threading.Thread(target=self._refresh_dynamic_worker, args=(api,), daemon=True).start()

    def _refresh_dynamic_worker(self, api: Any) -> None:
        try:
            ai_status = api.check_ai_status() if hasattr(api, "check_ai_status") else {}
        except Exception:
            logger.exception("ai status failed")
            ai_status = {"available": False, "message": "erreur"}
        try:
            since = datetime.now() - timedelta(hours=24)
            logs = api.database.get_audit_logs(start_date=since, limit=HISTORY_TAIL)
            since_24h = datetime.now() - timedelta(hours=24)
            all_24h = api.database.get_audit_logs(start_date=since_24h, limit=10_000)
            n_ops = len(all_24h)
            n_err = sum(1 for log in all_24h if not log.success)
        except Exception:
            logger.exception("audit fetch failed")
            logs = []
            n_ops = n_err = 0

        url = api.get_setting("ollama_url", "—")
        if ai_status.get("available"):
            kind = "success"
            status_text = f"En ligne · {ai_status.get('version', '')}".strip()
            current_model = ai_status.get("current_model") or "(aucun chargé)"
        elif ai_status.get("status") == "not_initialized":
            kind = "muted"
            status_text = "Non initialisé"
            current_model = "—"
        else:
            kind = "warning"
            status_text = ai_status.get("message", "Hors ligne") or "Hors ligne"
            current_model = "—"

        self.after(0, lambda: self._set_model_status(kind, status_text, url, current_model))
        self.after(0, lambda lg=logs, no=n_ops, ne=n_err: self._set_history_summary(no, ne, lg))

    def _set_model_status(self, kind: str, status_text: str, url: str, model: str) -> None:
        color = {
            "success": get_color("success"),
            "warning": get_color("warning"),
            "error": get_color("error"),
            "muted": get_color("fg_muted"),
        }[kind]
        self._model_status_dot.configure(text_color=color)
        self._model_status_text.configure(text=status_text, text_color=color)
        self._model_url_label.configure(text=url)
        self._model_name_label.configure(text=model)

    def _set_history_summary(self, n_ops: int, n_err: int, logs: list[Any]) -> None:
        text = f"{fmt_int(n_ops)} opérations / 24 h · {fmt_int(n_err)} erreur(s)"
        self._history_summary.configure(
            text=text,
            text_color=get_color("warning") if n_err else get_color("fg"),
        )
        for i, label in enumerate(self._history_lines):
            if i < len(logs):
                log = logs[i]
                ts = log.timestamp.strftime("%H:%M:%S")
                action = log.action_type.value
                fname = Path(log.file_path).name if log.file_path else "—"
                if len(fname) > 22:
                    fname = fname[:19] + "…"
                ok = "✓" if log.success else "✗"
                color = get_color("success") if log.success else get_color("error")
                label.configure(text=f"{ts}  {ok} {action:<14} {fname}", text_color=color)
            else:
                label.configure(text="", text_color=get_color("fg_muted"))

    # ==================================================================
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


_ = SPACE_LG  # keep imported constant for future use
