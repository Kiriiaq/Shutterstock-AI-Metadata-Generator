"""Batch export view — compact dashboard, FTP-aware.

Single modal that consumes ``app_state["selected_paths"]`` and lets
the user:

- Choose target platform : Adobe / Shutterstock / both (radio, inline).
- Toggle « écrire IPTC dans le fichier » (off by default — fichier
  intact si décoché).
- Toggle « enrichir avec IA » (off by default — économe).
- Pick output folder + optional FTP push (with credentials form).
- Watch the per-file status update live (1 row per file, compact).
- Resume after errors — each row carries its own status badge.

The whole layout is built in ONE column with three "bands":
    1. Top band : options (radios + checkboxes) on a single row.
    2. Middle band : destination (folder + FTP toggle + credentials).
    3. Main band : compact file table with live status + log textbox.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from tkinter import filedialog
from typing import TYPE_CHECKING, Any, List, Optional

import customtkinter as ctk

from app.components.data_table import Column, DataTable
from app.components.empty_state import EmptyState
from app.config.theme import (
    RADIUS_MD,
    RADIUS_SM,
    SPACE_MD,
    SPACE_SM,
    SPACE_XL,
    SPACE_XS,
    get_font,
    palette_pair,
)
from app.utils.formatters import fmt_int, fmt_size
from app.views.base_view import BaseView, _modal_header

if TYPE_CHECKING:
    from app.app import App

logger = logging.getLogger(__name__)


STATUS_GLYPH = {
    "pending": "⏸",
    "analyzing": "⏳",
    "writing_iptc": "✎",
    "done": "✅",
    "failed": "❌",
}
STATUS_COLOR = {
    "pending": "fg_muted",
    "analyzing": "accent",
    "writing_iptc": "warning",
    "done": "success",
    "failed": "error",
}


class ExportBatchView(BaseView):
    view_id = "export_batch"

    def __init__(self, master: ctk.CTkFrame, *, app: "App") -> None:
        super().__init__(master)
        self.app = app
        self._running = False
        self._files: List[Path] = []
        self._status_by_path: dict[str, str] = {}
        self._build()

    # ------------------------------------------------------------------

    def _get_target_paths(self) -> List[Path]:
        raw = self.app.app_state.get("selected_paths") or []
        explicit = self.app.app_state.get("expert_report_paths") or []
        paths = explicit or raw
        return [Path(p) for p in paths if p]

    def _build(self) -> None:
        if self.app.api is None:
            EmptyState(
                self,
                icon="📤",
                title="Export indisponible",
                subtitle="Le backend n'est pas chargé.",
            ).grid(row=0, column=0, sticky="nsew")
            return

        self._files = self._get_target_paths()
        if not self._files:
            EmptyState(
                self,
                icon="📤",
                title="Aucun fichier sélectionné",
                subtitle=(
                    "Sélectionnez des images dans Sources, "
                    "puis ouvrez « Exporter… »."
                ),
            ).grid(row=0, column=0, sticky="nsew")
            return

        wrapper = ctk.CTkFrame(self, fg_color="transparent")
        wrapper.grid(row=0, column=0, sticky="nsew", padx=SPACE_XL, pady=SPACE_XL)
        wrapper.grid_columnconfigure(0, weight=1)
        wrapper.grid_rowconfigure(3, weight=1)  # table grows

        _modal_header(wrapper, icon="📤", title="Exporter le lot", row=0)

        self._build_options_band(wrapper, row=1)
        self._build_ollama_band(wrapper, row=2)
        self._build_destination_band(wrapper, row=3)
        self._build_file_table(wrapper, row=4)
        self._build_actions(wrapper, row=5)
        # Grow rule: row 4 (the file table) is the only one that
        # stretches when the modal is resized. The bands above stay
        # tight, actions at the bottom stay tight too.
        wrapper.grid_rowconfigure(4, weight=1)

    # ------------------------------------------------------------------
    # Options band — single row : plateforme + IPTC + IA + résumé sélection
    # ------------------------------------------------------------------

    def _build_options_band(self, parent: ctk.CTkFrame, row: int) -> None:
        band = ctk.CTkFrame(
            parent,
            fg_color=palette_pair("bg_elevated"),
            corner_radius=RADIUS_MD,
            border_color=palette_pair("border"),
            border_width=1,
        )
        band.grid(row=row, column=0, sticky="ew", pady=(0, SPACE_SM))
        band.grid_columnconfigure(99, weight=1)

        # Plateforme — radios inline. All free now: the only paywall is
        # the export run itself (quota enforced at Start). Default to
        # « both » since dual-platform CSV is no longer gated.
        ctk.CTkLabel(
            band, text="Plateforme :", font=get_font("small"),
            text_color=palette_pair("fg_muted"),
        ).grid(row=0, column=0, sticky="w", padx=(SPACE_MD, SPACE_SM), pady=SPACE_SM)

        self._platform_var = ctk.StringVar(value="both")
        platform_options = [
            ("adobe", "Adobe Stock"),
            ("shutterstock", "Shutterstock"),
            ("both", "Les deux"),
        ]
        for col, (val, label) in enumerate(platform_options, start=1):
            ctk.CTkRadioButton(
                band, text=label, variable=self._platform_var, value=val,
                font=get_font("body"),
            ).grid(row=0, column=col, sticky="w", padx=SPACE_XS, pady=SPACE_SM)

        # Séparateur visuel
        ctk.CTkLabel(band, text="│", text_color=palette_pair("border")).grid(
            row=0, column=10, padx=SPACE_SM
        )

        # IPTC + IA — checkboxes inline (both free)
        self._write_iptc_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            band, text="Écrire IPTC dans le fichier",
            variable=self._write_iptc_var, font=get_font("body"),
        ).grid(row=0, column=11, sticky="w", padx=SPACE_SM, pady=SPACE_SM)

        self._use_ai_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            band, text="Enrichir avec IA (Ollama)",
            variable=self._use_ai_var, font=get_font("body"),
            command=self._on_ai_toggle,
        ).grid(row=0, column=12, sticky="w", padx=SPACE_SM, pady=SPACE_SM)

        # Résumé sélection — toujours à droite
        total_size = sum((p.stat().st_size if p.exists() else 0) for p in self._files)
        ctk.CTkLabel(
            band,
            text=f"{fmt_int(len(self._files))} fichier(s) · {fmt_size(total_size)}",
            font=get_font("small"),
            text_color=palette_pair("fg_muted"),
        ).grid(row=0, column=99, sticky="e", padx=SPACE_MD)

        # Export-quota banner (Community only). The data export is the
        # single paid feature: COMMUNITY_EXPORT_QUOTA free runs, then
        # the 10 € lifetime key. Hidden entirely for licensed users.
        self._quota_banner = None
        if not self._export_unlocked():
            self._quota_banner = ctk.CTkLabel(
                band, text="", font=get_font("small"),
                text_color=palette_pair("warning"), anchor="w", justify="left",
            )
            self._quota_banner.grid(
                row=1, column=0, columnspan=100, sticky="ew",
                padx=SPACE_MD, pady=(0, SPACE_SM),
            )
            self._refresh_quota_banner()

    # ------------------------------------------------------------------
    # Export quota — the single paywall is the data export itself
    # ------------------------------------------------------------------

    def _export_unlocked(self) -> bool:
        """True iff the licence removes the export quota (unlimited)."""
        api = self.app.api
        lic = getattr(api, "license", None)
        return bool(lic and lic.has_feature("data_export"))

    def _refresh_quota_banner(self) -> None:
        """Update the Community export-quota banner (no-op if licensed)."""
        banner = getattr(self, "_quota_banner", None)
        if banner is None:
            return
        from src.modules.licensing import COMMUNITY_EXPORT_QUOTA

        remaining = self.app.api.export_quota_remaining()
        if remaining <= 0:
            banner.configure(
                text=(
                    f"🛑 Export gratuit épuisé ({COMMUNITY_EXPORT_QUOTA}/"
                    f"{COMMUNITY_EXPORT_QUOTA}). Passez Pro (10 € à vie) pour "
                    f"un export illimité — Réglages → Licence."
                ),
                text_color=palette_pair("error"),
            )
        else:
            banner.configure(
                text=(
                    f"🎁 Édition Community · {remaining}/{COMMUNITY_EXPORT_QUOTA} "
                    f"export(s) gratuit(s) restant(s). Pro (10 € à vie) = illimité."
                ),
                text_color=palette_pair("warning"),
            )

    # ------------------------------------------------------------------
    # Ollama band — revealed when « Enrichir avec IA » is checked
    # ------------------------------------------------------------------

    def _build_ollama_band(self, parent: ctk.CTkFrame, row: int) -> None:
        """Compact single-row band: test + dropdown + load + status chip.

        The whole band is hidden by default and revealed by
        :meth:`_on_ai_toggle` when the user checks « Enrichir avec
        IA ». Layout (all on one row, left-to-right):

            [Modèle ▼]  [🔌 Tester]  [⬇ Charger]   ● llama3.2-vision (Chargé)

        Status chip colours :
            grey   = inconnu / non testé
            yellow = en cours (test ou chargement)
            green  = modèle chargé
            red    = serveur HS / chargement KO
        """
        self._ollama_band = ctk.CTkFrame(
            parent,
            fg_color=palette_pair("bg_elevated"),
            corner_radius=RADIUS_MD,
            border_color=palette_pair("border"),
            border_width=1,
        )
        self._ollama_band.grid(row=row, column=0, sticky="ew", pady=(0, SPACE_SM))
        self._ollama_band.grid_columnconfigure(99, weight=1)
        # Hidden by default — only show when IA checkbox is on.
        self._ollama_band.grid_remove()

        ctk.CTkLabel(
            self._ollama_band, text="🧠 Modèle :", font=get_font("small"),
            text_color=palette_pair("fg_muted"),
        ).grid(row=0, column=0, sticky="w", padx=(SPACE_MD, SPACE_SM), pady=SPACE_SM)

        # Pre-fill dropdown with the persisted model (if any) — list
        # is refreshed when « Tester » is clicked.
        api = self.app.api
        saved = api.get_setting("ollama_model", "") if api else ""
        initial_values = [saved] if saved else ["(non testé)"]
        self._model_var = ctk.StringVar(value=initial_values[0])
        self._model_dropdown = ctk.CTkOptionMenu(
            self._ollama_band,
            values=initial_values,
            variable=self._model_var,
            width=240, height=26,
            font=get_font("body"),
        )
        self._model_dropdown.grid(row=0, column=1, sticky="w", padx=SPACE_XS, pady=SPACE_SM)

        ctk.CTkButton(
            self._ollama_band, text="🔌 Tester", width=90, height=26,
            font=get_font("small"),
            fg_color=palette_pair("bg_hover"),
            hover_color=palette_pair("bg_active"),
            text_color=palette_pair("fg"),
            border_width=1, border_color=palette_pair("border"),
            command=self._test_ollama,
        ).grid(row=0, column=2, padx=SPACE_XS, pady=SPACE_SM)

        ctk.CTkButton(
            self._ollama_band, text="⬇ Charger", width=100, height=26,
            font=get_font("small"),
            fg_color=palette_pair("accent"),
            hover_color=palette_pair("accent_hover"),
            text_color=palette_pair("accent_fg"),
            command=self._load_model,
        ).grid(row=0, column=3, padx=SPACE_XS, pady=SPACE_SM)

        # Status chip — coloured dot + label. Updated by the workers.
        self._ollama_status = ctk.CTkLabel(
            self._ollama_band,
            text="● Inconnu",
            font=get_font("small"),
            text_color=palette_pair("fg_muted"),
            anchor="w",
        )
        self._ollama_status.grid(row=0, column=99, sticky="e", padx=SPACE_MD, pady=SPACE_SM)

    def _on_ai_toggle(self) -> None:
        """Reveal/hide the Ollama band based on the IA checkbox."""
        if self._use_ai_var.get():
            self._ollama_band.grid()
            # First reveal → auto-probe the server so the dropdown is
            # populated without forcing the user to click « Tester ».
            if not getattr(self, "_ollama_auto_probed", False):
                self._ollama_auto_probed = True
                self._test_ollama()
        else:
            self._ollama_band.grid_remove()

    # ----- Test connection + populate model list --------------------

    def _test_ollama(self) -> None:
        """Probe Ollama, refresh the dropdown with vision models."""
        self._set_ollama_status("Test…", "warning")
        api = self.app.api

        def worker():
            try:
                status = api.check_ai_status()
                models = api.list_vision_models(refresh=True)
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda e=str(exc): self._on_test_done(False, e, []))
                return
            ok = bool(status.get("available"))
            msg = status.get("message", "")
            self.after(0, lambda: self._on_test_done(ok, msg, models))

        threading.Thread(target=worker, daemon=True).start()

    def _on_test_done(self, ok: bool, msg: str, models: List[str]) -> None:
        if not ok:
            self._set_ollama_status(f"● Ollama HS : {msg}", "error")
            return
        if not models:
            self._set_ollama_status("● Connecté · aucun modèle vision installé", "warning")
            return
        # Repopulate dropdown
        current = self._model_var.get()
        keep = current if current in models else models[0]
        self._model_dropdown.configure(values=models)
        self._model_var.set(keep)
        loaded = self.app.api.get_current_model()
        if loaded and loaded == keep:
            self._set_ollama_status(f"● {loaded} (chargé)", "success")
        else:
            self._set_ollama_status(
                f"● Connecté · {len(models)} modèle(s) · « Charger » pour activer",
                "fg_muted",
            )

    # ----- Preload --------------------------------------------------

    def _load_model(self) -> None:
        name = self._model_var.get().strip()
        if not name or name == "(non testé)":
            self._set_ollama_status("● Cliquez « Tester » d'abord", "warning")
            return
        self._set_ollama_status(f"● Chargement de {name}…", "warning")
        api = self.app.api

        def worker():
            try:
                ok, msg = api.preload_model(name)
            except Exception as exc:  # noqa: BLE001
                ok, msg = False, str(exc)
            self.after(0, lambda: self._on_load_done(name, ok, msg))

        threading.Thread(target=worker, daemon=True).start()

    def _on_load_done(self, name: str, ok: bool, msg: str) -> None:
        if ok:
            self._set_ollama_status(f"● {name} (chargé)", "success")
            self.app.toasts.show(msg, kind="success")
        else:
            self._set_ollama_status(f"● Échec : {msg[:60]}", "error")
            self.app.toasts.show(f"Chargement IA KO : {msg}", kind="error")

    def _set_ollama_status(self, text: str, color_key: str) -> None:
        self._ollama_status.configure(
            text=text,
            text_color=palette_pair(color_key),
        )

    # ------------------------------------------------------------------
    # Destination band — dossier + FTP toggle + champs si actif
    # ------------------------------------------------------------------

    def _build_destination_band(self, parent: ctk.CTkFrame, row: int) -> None:
        band = ctk.CTkFrame(
            parent,
            fg_color=palette_pair("bg_elevated"),
            corner_radius=RADIUS_MD,
            border_color=palette_pair("border"),
            border_width=1,
        )
        band.grid(row=row, column=0, sticky="ew", pady=(0, SPACE_SM))
        band.grid_columnconfigure(1, weight=1)

        # Ligne 1 — dossier de sortie + bouton parcourir
        ctk.CTkLabel(
            band, text="Dossier d'export :", font=get_font("small"),
            text_color=palette_pair("fg_muted"),
        ).grid(row=0, column=0, sticky="w", padx=(SPACE_MD, SPACE_SM), pady=SPACE_SM)

        # Défaut : dossier de l'image (peut être édité)
        default_out = str(self._files[0].parent if self._files else Path.home())
        self._out_var = ctk.StringVar(value=default_out)
        ctk.CTkEntry(
            band, textvariable=self._out_var, font=get_font("body"), height=28,
        ).grid(row=0, column=1, sticky="ew", pady=SPACE_SM)

        ctk.CTkButton(
            band, text="…", width=36, height=28,
            command=self._browse_out,
        ).grid(row=0, column=2, padx=SPACE_XS, pady=SPACE_SM)

        # Basename
        ctk.CTkLabel(
            band, text="Nom de base :", font=get_font("small"),
            text_color=palette_pair("fg_muted"),
        ).grid(row=0, column=3, sticky="w", padx=(SPACE_MD, SPACE_SM), pady=SPACE_SM)
        self._basename_var = ctk.StringVar(
            value=f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        ctk.CTkEntry(
            band, textvariable=self._basename_var, font=get_font("body"),
            height=28, width=200,
        ).grid(row=0, column=4, sticky="w", pady=SPACE_SM, padx=(0, SPACE_MD))

        # Ligne 2 — FTP : toggle + bandeau credentials (revealed)
        self._ftp_enabled_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            band, text="Pousser en FTP après export",
            variable=self._ftp_enabled_var, font=get_font("body"),
            command=self._on_ftp_toggle,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=(SPACE_MD, 0), pady=(0, SPACE_SM))

        # Hidden by default — revealed when checkbox is ON.
        self._ftp_frame = ctk.CTkFrame(band, fg_color="transparent")
        self._ftp_frame.grid(row=2, column=0, columnspan=5, sticky="ew",
                             padx=SPACE_MD, pady=(0, SPACE_SM))
        self._ftp_frame.grid_columnconfigure(1, weight=1)
        self._ftp_frame.grid_columnconfigure(3, weight=1)
        self._ftp_frame.grid_remove()

        # FTP fields on a single row each
        self._ftp_host = self._labeled_entry(self._ftp_frame, "Hôte", 0, 0, width=240)
        self._ftp_user = self._labeled_entry(self._ftp_frame, "Utilisateur", 0, 2, width=200)
        self._ftp_password = self._labeled_entry(
            self._ftp_frame, "Mot de passe", 1, 0, width=240, show="*"
        )
        self._ftp_remote = self._labeled_entry(self._ftp_frame, "Dossier distant", 1, 2, width=200)

        # FTP options row : TLS + test button
        opts_row = ctk.CTkFrame(self._ftp_frame, fg_color="transparent")
        opts_row.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(SPACE_XS, 0))
        self._ftp_tls_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            opts_row, text="FTPS (TLS)", variable=self._ftp_tls_var,
            font=get_font("small"),
        ).pack(side="left")
        ctk.CTkButton(
            opts_row, text="Tester la connexion", height=24,
            font=get_font("small"),
            fg_color=palette_pair("bg_hover"),
            hover_color=palette_pair("bg_active"),
            text_color=palette_pair("fg"),
            border_width=1, border_color=palette_pair("border"),
            command=self._test_ftp,
        ).pack(side="left", padx=SPACE_SM)
        self._ftp_test_status = ctk.CTkLabel(
            opts_row, text="", font=get_font("small"),
            text_color=palette_pair("fg_muted"),
        )
        self._ftp_test_status.pack(side="left", padx=SPACE_SM)

        # Pre-fill from settings if present (so users don't retype each session)
        settings = self.app.api.get_all_settings() if self.app.api else {}
        self._ftp_host.insert(0, settings.get("ftp_host", ""))
        self._ftp_user.insert(0, settings.get("ftp_user", ""))
        self._ftp_remote.insert(0, settings.get("ftp_remote_dir", "/"))
        # NB: password intentionally NOT persisted by default.

    def _labeled_entry(
        self,
        parent: ctk.CTkFrame,
        label: str,
        row: int,
        col: int,
        *,
        width: int = 180,
        show: str = "",
    ) -> ctk.CTkEntry:
        ctk.CTkLabel(
            parent, text=label, font=get_font("small"),
            text_color=palette_pair("fg_muted"), anchor="w",
        ).grid(row=row, column=col, sticky="w", padx=(0, SPACE_XS), pady=SPACE_XS)
        entry = ctk.CTkEntry(parent, font=get_font("body"), width=width, height=24, show=show)
        entry.grid(row=row, column=col + 1, sticky="ew", pady=SPACE_XS, padx=(0, SPACE_MD))
        return entry

    def _on_ftp_toggle(self) -> None:
        if self._ftp_enabled_var.get():
            self._ftp_frame.grid()
        else:
            self._ftp_frame.grid_remove()

    # ------------------------------------------------------------------
    # File table — 1 row per file, live status
    # ------------------------------------------------------------------

    def _build_file_table(self, parent: ctk.CTkFrame, row: int) -> None:
        wrap = ctk.CTkFrame(
            parent,
            fg_color=palette_pair("bg_elevated"),
            corner_radius=RADIUS_MD,
            border_color=palette_pair("border"),
            border_width=1,
        )
        wrap.grid(row=row, column=0, sticky="nsew", pady=(0, SPACE_SM))
        wrap.grid_columnconfigure(0, weight=1)
        wrap.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            wrap, text="Fichiers du lot", font=get_font("small"),
            text_color=palette_pair("fg_subtle"), anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=SPACE_MD, pady=(SPACE_SM, 0))

        self._table = DataTable(
            wrap,
            columns=[
                Column(id="name", label="Fichier", width=300),
                Column(id="size", label="Taille", width=80, anchor="e"),
                Column(id="status", label="Statut", width=110, anchor="center"),
                Column(id="info", label="Info", width=200, anchor="w"),
            ],
            select_mode="browse",
        )
        self._table.grid(row=1, column=0, sticky="nsew", padx=SPACE_MD, pady=SPACE_SM)
        self._populate_initial_table()

    def _populate_initial_table(self) -> None:
        rows = []
        for p in self._files:
            size = p.stat().st_size if p.exists() else 0
            rows.append({
                "name": p.name,
                "size": fmt_size(size),
                "status": f"{STATUS_GLYPH['pending']} en attente",
                "info": "",
                "_path": str(p),
            })
        self._table.set_rows(rows)

    def _update_row(self, path: Path, status: str, info: str = "") -> None:
        """Mutate the row matching *path* in place. Called from main thread.

        We rebuild the whole table from ``self._files`` + the
        ``_status_by_path`` map rather than trying to mutate the
        underlying DataTable in place — the table's row API doesn't
        expose a stable identity per row, so a full rebuild is
        simpler and indistinguishable visually.
        """
        self._status_by_path[str(path)] = status
        rebuilt = []
        for p in self._files:
            s = self._status_by_path.get(str(p), "pending")
            size = p.stat().st_size if p.exists() else 0
            badge = f"{STATUS_GLYPH.get(s, '?')} {s}"
            rebuilt.append({
                "name": p.name,
                "size": fmt_size(size),
                "status": badge,
                "info": info if str(p) == str(path) else "",
                "_path": str(p),
            })
        self._table.set_rows(rebuilt)

    # ------------------------------------------------------------------
    # Actions row + log
    # ------------------------------------------------------------------

    def _build_actions(self, parent: ctk.CTkFrame, row: int) -> None:
        actions = ctk.CTkFrame(parent, fg_color="transparent")
        actions.grid(row=row, column=0, sticky="ew", pady=(SPACE_SM, 0))
        actions.grid_columnconfigure(1, weight=1)

        self._start_btn = ctk.CTkButton(
            actions, text="▶ Lancer l'export",
            fg_color=palette_pair("accent"),
            hover_color=palette_pair("accent_hover"),
            text_color=palette_pair("accent_fg"),
            font=get_font("body_strong"),
            height=32, width=180,
            command=self._start,
        )
        self._start_btn.grid(row=0, column=0, padx=(0, SPACE_SM))

        self._progress = ctk.CTkProgressBar(
            actions, height=10, corner_radius=RADIUS_SM,
            progress_color=palette_pair("accent"),
            fg_color=palette_pair("bg_hover"),
            border_color=palette_pair("border"),
            border_width=1,
        )
        self._progress.set(0)
        self._progress.grid(row=0, column=1, sticky="ew", padx=SPACE_SM)

        self._status_label = ctk.CTkLabel(
            actions, text="En attente", font=get_font("small"),
            text_color=palette_pair("fg_muted"), width=160, anchor="e",
        )
        self._status_label.grid(row=0, column=2, sticky="e")

    # ------------------------------------------------------------------
    # Browsers / FTP test
    # ------------------------------------------------------------------

    def _browse_out(self) -> None:
        path = filedialog.askdirectory(title="Dossier d'export CSV",
                                       initialdir=self._out_var.get() or str(Path.home()))
        if path:
            self._out_var.set(path)

    def _test_ftp(self) -> None:
        cfg = self._build_ftp_config()
        if cfg is None:
            self._ftp_test_status.configure(
                text="Renseignez hôte, utilisateur, mot de passe.",
                text_color=palette_pair("warning"),
            )
            return
        self._ftp_test_status.configure(text="Test en cours…",
                                        text_color=palette_pair("fg_muted"))

        def worker():
            try:
                ok, msg = self.app.api.test_ftp_connection(cfg)
            except Exception as exc:  # noqa: BLE001
                ok, msg = False, str(exc)
            self.after(0, lambda: self._on_ftp_test_done(ok, msg))

        threading.Thread(target=worker, daemon=True).start()

    def _on_ftp_test_done(self, ok: bool, msg: str) -> None:
        self._ftp_test_status.configure(
            text=msg[:80],
            text_color=palette_pair("success" if ok else "error"),
        )

    def _build_ftp_config(self) -> Optional[Any]:
        host = self._ftp_host.get().strip()
        user = self._ftp_user.get().strip()
        password = self._ftp_password.get()
        if not (host and user and password):
            return None
        from src.modules.export.ftp_uploader import FtpConfig
        return FtpConfig(
            host=host,
            user=user,
            password=password,
            remote_dir=self._ftp_remote.get().strip() or "/",
            use_tls=bool(self._ftp_tls_var.get()),
        )

    # ------------------------------------------------------------------
    # Start / worker
    # ------------------------------------------------------------------

    def _start(self) -> None:
        if self._running:
            return

        api = self.app.api

        # --- Paywall : the data export itself (Community quota) ------
        # Platform choice, AI enrichment and batch size are all free.
        # The only gated thing is running the export: COMMUNITY_EXPORT_
        # QUOTA free runs, then the 10 € lifetime key.
        if not self._export_unlocked() and api.export_quota_remaining() <= 0:
            self.app.toasts.show(
                "Export gratuit épuisé. Passez Pro (10 € à vie) pour un "
                "export illimité — Réglages → Licence.",
                kind="warning",
                timeout_ms=7000,
            )
            self._status_label.configure(
                text="⛔ Quota épuisé — Pro requis",
                text_color=palette_pair("warning"),
            )
            self._refresh_quota_banner()
            return

        out_dir = Path(self._out_var.get().strip() or self._files[0].parent)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.app.toasts.show(f"Dossier inaccessible : {exc}", kind="error")
            return

        ftp_cfg = self._build_ftp_config() if self._ftp_enabled_var.get() else None
        if self._ftp_enabled_var.get() and ftp_cfg is None:
            self.app.toasts.show("FTP : credentials incomplets.", kind="warning")
            return

        # Persist non-secret FTP settings for next session
        try:
            self.app.api.set_setting("ftp_host", self._ftp_host.get().strip())
            self.app.api.set_setting("ftp_user", self._ftp_user.get().strip())
            self.app.api.set_setting("ftp_remote_dir", self._ftp_remote.get().strip())
        except Exception:  # noqa: BLE001
            logger.debug("FTP settings persist failed", exc_info=True)

        self._running = True
        self._start_btn.configure(state="disabled", text="Export en cours…")
        self._progress.set(0)
        self._status_label.configure(
            text=f"0 / {len(self._files)}",
            text_color=palette_pair("fg_muted"),
        )

        threading.Thread(
            target=self._worker,
            args=(out_dir, ftp_cfg),
            daemon=True,
        ).start()

    def _worker(self, out_dir: Path, ftp_cfg: Optional[Any]) -> None:
        api = self.app.api
        platform = self._platform_var.get()
        write_iptc = bool(self._write_iptc_var.get())
        use_ai = bool(self._use_ai_var.get())
        basename = self._basename_var.get().strip() or "metadata"
        total = len(self._files)
        done = 0

        def on_progress(fp: Any) -> None:
            nonlocal done
            self.after(0, lambda: self._update_row(fp.path, fp.status.value,
                                                   info=fp.error or ""))
            if fp.status.value in {"done", "failed"}:
                done += 1
                self.after(0, lambda d=done: self._tick(d, total))

        try:
            result = api.export_batch(
                self._files,
                out_dir,
                platform=platform,
                write_iptc=write_iptc,
                use_ai=use_ai,
                basename=basename,
                ftp_config=ftp_cfg,
                on_progress=on_progress,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("export_batch crashed")
            self.after(0, lambda e=str(exc): self._on_finished_error(e))
            return

        self.after(0, lambda r=result: self._on_finished(r))

    def _tick(self, done: int, total: int) -> None:
        self._progress.set(done / total if total else 0)
        self._status_label.configure(text=f"{done} / {total}")

    def _on_finished(self, result: Any) -> None:
        self._running = False
        self._start_btn.configure(state="normal", text="▶ Relancer")
        ok = result.success_count
        ko = result.failure_count
        total = ok + ko
        if result.errors:
            self._status_label.configure(
                text=f"{ok} / {total} OK · {ko} erreur(s)",
                text_color=palette_pair("warning"),
            )
            self.app.toasts.show(
                f"Export terminé avec {ko} erreur(s). Voir détails.",
                kind="warning",
            )
        else:
            self._status_label.configure(
                text=f"{ok} / {total} OK",
                text_color=palette_pair("success"),
            )
            csv_count = len(result.csv_paths)
            ftp_msg = ""
            if result.ftp_result and result.ftp_result.is_complete_success:
                ftp_msg = f" + {result.ftp_result.success_count} fichier(s) FTP"
            self.app.toasts.show(
                f"Export OK : {csv_count} CSV produit(s){ftp_msg}.",
                kind="success",
            )

        # Debit one free export run (Community only) when output was
        # actually produced, then refresh the banner so the next run
        # shows the updated count or the upsell.
        if not self._export_unlocked() and getattr(result, "csv_paths", None):
            self.app.api.consume_export_quota()
            self._refresh_quota_banner()

    def _on_finished_error(self, msg: str) -> None:
        self._running = False
        self._start_btn.configure(state="normal", text="▶ Réessayer")
        self._status_label.configure(text="Erreur", text_color=palette_pair("error"))
        self.app.toasts.show(f"Échec : {msg}", kind="error")
