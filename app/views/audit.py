"""Audit view — searchable log of every backend operation."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import filedialog
from typing import TYPE_CHECKING

import customtkinter as ctk

from app.components.data_table import Column, DataTable
from app.components.empty_state import EmptyState
from app.config.theme import (
    RADIUS_MD,
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    SPACE_XL,
    get_font,
    palette_pair,
)
from app.utils.formatters import fmt_datetime, fmt_duration_ms
from app.views.base_view import BaseView, _modal_header

if TYPE_CHECKING:
    from app.app import App

logger = logging.getLogger(__name__)


class AuditView(BaseView):
    view_id = "audit"

    PERIODS: dict[str, int] = {
        "Tout l'historique": 0,
        "Aujourd'hui": 1,
        "7 derniers jours": 7,
        "30 derniers jours": 30,
    }

    def __init__(self, master: ctk.CTkFrame, *, app: "App") -> None:
        super().__init__(master)
        self.app = app
        self._build()

    # ------------------------------------------------------------------

    def _build(self) -> None:
        if self.app.api is None:
            EmptyState(
                self,
                icon="📜",
                title="Historique indisponible",
                subtitle="Le backend n'est pas chargé.",
            ).grid(row=0, column=0, sticky="nsew")
            return

        wrapper = ctk.CTkFrame(self, fg_color="transparent")
        wrapper.grid(row=0, column=0, sticky="nsew", padx=SPACE_XL, pady=SPACE_XL)
        wrapper.grid_columnconfigure(0, weight=1)
        wrapper.grid_rowconfigure(2, weight=1)

        # Header: icon + h1 title, anchored top-left.
        _modal_header(wrapper, icon="🕐", title="Historique", row=0)

        self._build_filters(wrapper, row=1)
        self._build_table(wrapper, row=2)
        self.after(50, self.on_enter)

    def _build_filters(self, parent: ctk.CTkFrame, row: int) -> None:
        bar = ctk.CTkFrame(parent, fg_color=palette_pair("bg_elevated"), corner_radius=RADIUS_MD)
        bar.grid(row=row, column=0, sticky="ew", pady=(0, SPACE_MD))

        ctk.CTkLabel(bar, text="Période :", font=get_font("body"), text_color=palette_pair("fg")).pack(
            side="left", padx=(SPACE_LG, SPACE_SM), pady=SPACE_MD
        )
        self._period_var = ctk.StringVar(value="7 derniers jours")
        ctk.CTkOptionMenu(
            bar,
            values=list(self.PERIODS.keys()),
            variable=self._period_var,
            command=lambda _v: self._reload(),
            width=200,
        ).pack(side="left", padx=(0, SPACE_LG), pady=SPACE_MD)

        ctk.CTkLabel(bar, text="Statut :", font=get_font("body"), text_color=palette_pair("fg")).pack(
            side="left", padx=(0, SPACE_SM), pady=SPACE_MD
        )
        self._status_var = ctk.StringVar(value="Tous")
        ctk.CTkOptionMenu(
            bar,
            values=["Tous", "Succès", "Échec"],
            variable=self._status_var,
            command=lambda _v: self._reload(),
            width=120,
        ).pack(side="left", padx=(0, SPACE_LG), pady=SPACE_MD)

        ctk.CTkButton(bar, text="Actualiser", width=110, command=self._reload).pack(
            side="left", padx=SPACE_SM, pady=SPACE_MD
        )
        ctk.CTkButton(bar, text="Exporter…", width=110, command=self._export).pack(
            side="left", padx=SPACE_SM, pady=SPACE_MD
        )

    def _build_table(self, parent: ctk.CTkFrame, row: int) -> None:
        self._table = DataTable(
            parent,
            columns=[
                Column(id="timestamp", label="Date / heure", width=160),
                Column(id="action_type", label="Action", width=160),
                Column(id="file", label="Fichier", width=320),
                Column(id="status", label="Statut", width=80, anchor="center"),
                Column(id="duration", label="Durée", width=80, anchor="e"),
            ],
            select_mode="browse",
        )
        self._table.grid(row=row, column=0, sticky="nsew")
        self._table.on_activate(self._show_details)

    # ------------------------------------------------------------------

    def on_enter(self, **_kwargs) -> None:
        self._reload()

    def _reload(self) -> None:
        api = self.app.api
        if api is None:
            return
        threading.Thread(target=self._reload_worker, args=(api,), daemon=True).start()

    def _reload_worker(self, api) -> None:
        try:
            days = self.PERIODS.get(self._period_var.get(), 0)
            end = datetime.now()
            start = end - timedelta(days=days) if days else None
            logs = api.database.get_audit_logs(start_date=start, end_date=end, limit=500)
            wanted_status = self._status_var.get()
            if wanted_status == "Succès":
                logs = [log for log in logs if log.success]
            elif wanted_status == "Échec":
                logs = [log for log in logs if not log.success]
            self.after(0, lambda lg=logs: self._render_logs(lg))
        except Exception as e:
            logger.exception("Audit reload failed")
            self.after(0, lambda err=str(e): self.app.toasts.show(f"Échec : {err}", kind="error"))

    def _render_logs(self, logs) -> None:
        rows = []
        for log in logs:
            rows.append(
                {
                    "timestamp": fmt_datetime(log.timestamp),
                    "action_type": log.action_type.value,
                    "file": Path(log.file_path).name if log.file_path else "—",
                    "status": "✓" if log.success else "✗",
                    "duration": fmt_duration_ms(log.duration_ms) if log.duration_ms else "—",
                    "_log": log,
                }
            )
        self._table.set_rows(rows)

    def _show_details(self, row: dict) -> None:
        log = row.get("_log")
        if log is None:
            return

        def builder(parent: ctk.CTkFrame) -> None:
            tb = ctk.CTkTextbox(
                parent,
                font=get_font("code"),
                fg_color=palette_pair("bg"),
                text_color=palette_pair("fg"),
                border_width=0,
            )
            tb.grid(row=0, column=0, sticky="nsew")
            details = (
                f"Date    : {fmt_datetime(log.timestamp)}\n"
                f"Action  : {log.action_type.value}\n"
                f"Fichier : {log.file_path or '—'}\n"
                f"Statut  : {'Succès' if log.success else 'Échec'}\n"
                f"Durée   : {fmt_duration_ms(log.duration_ms) if log.duration_ms else '—'}\n"
                f"Batch   : {log.batch_id or '—'}\n"
            )
            if log.error_message:
                details += f"\nErreur :\n{log.error_message}\n"
            if log.details:
                details += f"\nDétails :\n{log.details}\n"
            tb.insert("1.0", details)
            tb.configure(state="disabled")

        self.app.show_details("Détails de l'opération", builder)

    def _export(self) -> None:
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
            self.app.toasts.show(f"{count} entrée(s) exportée(s).", kind="success")
        except Exception as e:
            logger.exception("Audit export failed")
            self.app.toasts.show(f"Échec : {e}", kind="error")
