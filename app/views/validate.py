"""Validate view — pre-upload checklist on the scanned set."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import customtkinter as ctk

from app.components.data_table import Column, DataTable
from app.components.empty_state import EmptyState
from app.config.theme import (
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    SPACE_XL,
    get_color,
    get_font,
)
from app.utils.formatters import fmt_int
from app.views.base_view import BaseView

if TYPE_CHECKING:
    from app.app import App

logger = logging.getLogger(__name__)


class ValidateView(BaseView):
    view_id = "validate"

    def __init__(self, master: ctk.CTkFrame, *, app: "App") -> None:
        super().__init__(master)
        self.app = app
        self._build()

    # ------------------------------------------------------------------

    def _build(self) -> None:
        api = self.app.api
        if api is None:
            EmptyState(
                self,
                icon="✓",
                title="Validation indisponible",
                subtitle="Le backend n'est pas chargé.",
            ).grid(row=0, column=0, sticky="nsew")
            return

        files = list(self.app.app_state.get("scanned_images") or [])
        if not files:
            EmptyState(
                self,
                icon="✓",
                title="Aucune image à valider",
                subtitle="Scannez d'abord un dossier dans Sources et tri.",
                action_label="Aller à Sources",
                on_action=lambda: self.app.router.navigate_to("sources"),
            ).grid(row=0, column=0, sticky="nsew")
            return

        wrapper = ctk.CTkFrame(self, fg_color="transparent")
        wrapper.grid(row=0, column=0, sticky="nsew", padx=SPACE_XL, pady=SPACE_XL)
        wrapper.grid_columnconfigure(0, weight=1)
        wrapper.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(wrapper, text="Validation", font=get_font("h1"), text_color=get_color("fg")).grid(
            row=0, column=0, sticky="w", pady=(0, SPACE_LG)
        )

        self._summary = ctk.CTkLabel(
            wrapper, text="Lancement…", font=get_font("body"), text_color=get_color("fg_muted")
        )
        self._summary.grid(row=1, column=0, sticky="w", pady=(0, SPACE_MD))

        self._table = DataTable(
            wrapper,
            columns=[
                Column(id="file", label="Fichier", width=280),
                Column(id="score", label="Score", width=80, anchor="e", sort_key=lambda v: int(v or 0)),
                Column(id="status", label="Statut", width=80, anchor="center"),
                Column(id="errors", label="Anomalies", width=80, anchor="e", sort_key=lambda v: int(v or 0)),
                Column(id="warnings", label="Alertes", width=80, anchor="e", sort_key=lambda v: int(v or 0)),
                Column(id="first_issue", label="Premier point bloquant", width=320),
            ],
            select_mode="browse",
        )
        self._table.grid(row=2, column=0, sticky="nsew")
        self._table.on_activate(self._show_details)

        threading.Thread(target=self._validate_worker, args=(api, files), daemon=True).start()

    def _validate_worker(self, api, files: list[Path]) -> None:
        results = []
        try:
            for f in files:
                try:
                    res = api.validate_image(f)
                    results.append((f, res))
                except Exception:
                    logger.exception("validate_image failed for %s", f)
        except Exception:
            logger.exception("Validate worker crashed")
        self.after(0, lambda r=results: self._render(r))

    def _render(self, results: list[tuple[Path, Any]]) -> None:
        rows = []
        ok = 0
        for path, res in results:
            errs = list(getattr(res, "errors", []))
            warns = list(getattr(res, "warnings", []))
            score = int(getattr(res, "completeness_score", 0))
            valid = bool(getattr(res, "is_valid", False))
            if valid:
                ok += 1
            rows.append(
                {
                    "file": path.name,
                    "score": score,
                    "status": "✓" if valid else "✗",
                    "errors": len(errs),
                    "warnings": len(warns),
                    "first_issue": (errs + warns)[0] if (errs or warns) else "—",
                    "_errors": errs,
                    "_warnings": warns,
                    "_path": path,
                }
            )
        self._table.set_rows(rows)
        total = len(results)
        ko = total - ok
        self._summary.configure(
            text=f"{fmt_int(total)} image(s) — {fmt_int(ok)} conformes, {fmt_int(ko)} à corriger.",
            text_color=get_color("success") if ko == 0 else get_color("warning"),
        )

    def _show_details(self, row: dict[str, Any]) -> None:
        errs = row.get("_errors", [])
        warns = row.get("_warnings", [])
        path = row.get("_path")

        def builder(parent: ctk.CTkFrame) -> None:
            ctk.CTkLabel(
                parent,
                text=str(path),
                font=get_font("body_strong"),
                text_color=get_color("fg"),
                wraplength=280,
                justify="left",
            ).pack(fill="x", pady=(0, SPACE_SM))
            if errs:
                ctk.CTkLabel(
                    parent, text="Anomalies", font=get_font("body_strong"), text_color=get_color("error")
                ).pack(anchor="w")
                for e in errs:
                    ctk.CTkLabel(
                        parent,
                        text=f"• {e}",
                        font=get_font("body"),
                        text_color=get_color("fg"),
                        wraplength=280,
                        justify="left",
                    ).pack(anchor="w")
            if warns:
                ctk.CTkLabel(
                    parent, text="Alertes", font=get_font("body_strong"), text_color=get_color("warning")
                ).pack(anchor="w", pady=(SPACE_SM, 0))
                for w in warns:
                    ctk.CTkLabel(
                        parent,
                        text=f"• {w}",
                        font=get_font("body"),
                        text_color=get_color("fg"),
                        wraplength=280,
                        justify="left",
                    ).pack(anchor="w")

        self.app.show_details("Détails de validation", builder)
