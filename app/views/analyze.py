"""Analyze view — batch AI processing with live progress."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

import customtkinter as ctk

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
from app.utils.formatters import fmt_int
from app.views.base_view import BaseView

if TYPE_CHECKING:
    from app.app import App

logger = logging.getLogger(__name__)


class AnalyzeView(BaseView):
    view_id = "analyze"

    def __init__(self, master: ctk.CTkFrame, *, app: "App") -> None:
        super().__init__(master)
        self.app = app
        self._processing = False
        self._build()

    # ------------------------------------------------------------------

    def _build(self) -> None:
        selected = self.app.app_state.get("selected_paths") or []
        if not selected:
            EmptyState(
                self,
                icon="🧠",
                title="Aucune image à analyser",
                subtitle="Retournez à Sources et tri pour sélectionner des images.",
                action_label="Aller à Sources",
                on_action=lambda: self.app.router.navigate_to("sources"),
            ).grid(row=0, column=0, sticky="nsew")
            return

        wrapper = ctk.CTkFrame(self, fg_color="transparent")
        wrapper.grid(row=0, column=0, sticky="nsew", padx=SPACE_XL, pady=SPACE_XL)
        wrapper.grid_columnconfigure(0, weight=1)
        wrapper.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(
            wrapper,
            text="Analyse IA",
            font=get_font("h1"),
            text_color=get_color("fg"),
        ).grid(row=0, column=0, sticky="w", pady=(0, SPACE_LG))

        self._build_options(wrapper, row=1, selected_count=len(selected))
        self._build_controls(wrapper, row=2)
        self._build_results(wrapper, row=3)

    def _build_options(self, parent: ctk.CTkFrame, row: int, selected_count: int) -> None:
        bar = ctk.CTkFrame(parent, fg_color=get_color("bg_elevated"), corner_radius=RADIUS_MD)
        bar.grid(row=row, column=0, sticky="ew", pady=(0, SPACE_MD))

        ctk.CTkLabel(
            bar,
            text=f"{fmt_int(selected_count)} image(s) sélectionnée(s)",
            font=get_font("body_strong"),
            text_color=get_color("fg"),
        ).pack(side="left", padx=SPACE_LG, pady=SPACE_MD)

        self._skip_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(bar, text="Ignorer si métadonnées présentes", variable=self._skip_var).pack(
            side="left", padx=SPACE_MD, pady=SPACE_MD
        )
        self._write_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(bar, text="Écrire les résultats dans les fichiers", variable=self._write_var).pack(
            side="left", padx=SPACE_MD, pady=SPACE_MD
        )

    def _build_controls(self, parent: ctk.CTkFrame, row: int) -> None:
        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.grid(row=row, column=0, sticky="ew", pady=(0, SPACE_MD))
        bar.grid_columnconfigure(2, weight=1)

        self._start_btn = ctk.CTkButton(
            bar,
            text="Démarrer l'analyse",
            fg_color=get_color("accent"),
            hover_color=get_color("accent_hover"),
            text_color=get_color("accent_fg"),
            font=get_font("body_strong"),
            command=self._start,
        )
        self._start_btn.grid(row=0, column=0, padx=(0, SPACE_SM))

        self._stop_btn = ctk.CTkButton(
            bar,
            text="Arrêter",
            fg_color=get_color("error"),
            text_color="#FFFFFF",
            state="disabled",
            command=self._stop,
        )
        self._stop_btn.grid(row=0, column=1, padx=SPACE_SM)

        self._progress = ctk.CTkProgressBar(bar)
        self._progress.set(0)
        self._progress.grid(row=0, column=2, sticky="ew", padx=SPACE_LG)

        self._status = ctk.CTkLabel(bar, text="Prêt", font=get_font("small"), text_color=get_color("fg_muted"))
        self._status.grid(row=0, column=3, sticky="e")

    def _build_results(self, parent: ctk.CTkFrame, row: int) -> None:
        self._results = ctk.CTkTextbox(
            parent,
            font=get_font("code"),
            fg_color=get_color("bg"),
            text_color=get_color("fg"),
            border_color=get_color("border"),
            border_width=1,
            corner_radius=RADIUS_MD,
        )
        self._results.grid(row=row, column=0, sticky="nsew")
        self._results.insert("1.0", "Les résultats apparaîtront ici en temps réel.\n")
        self._results.configure(state="disabled")

    # ------------------------------------------------------------------

    def _start(self) -> None:
        api = self.app.api
        if api is None:
            self.app.toasts.show("Backend indisponible.", kind="error")
            return
        selected = list(self.app.app_state.get("selected_paths") or [])
        if not selected:
            self.app.toasts.show("Aucune sélection.", kind="warning")
            return

        self._processing = True
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._progress.set(0)
        self._set_results("Initialisation de l'analyse…\n")

        threading.Thread(target=self._worker, args=(api, selected), daemon=True).start()

    def _worker(self, api: Any, selected: list[Any]) -> None:
        try:

            def on_progress(done: int, total: int, current: str) -> None:
                self.after(0, lambda: self._on_progress(done, total, current))

            def on_result(res: dict[str, Any]) -> None:
                self.after(0, lambda r=res: self._on_result(r))

            result = api.analyze_batch_ai(
                selected,
                skip_if_has_metadata=self._skip_var.get(),
                write_metadata=self._write_var.get(),
                on_progress=on_progress,
                on_result=on_result,
            )
            self.after(0, lambda r=result: self._on_complete(r))
        except Exception as e:
            logger.exception("Analyze worker failed")
            self.after(0, lambda err=str(e): self._on_failed(err))

    def _stop(self) -> None:
        api = self.app.api
        analyzer = getattr(api, "vision_analyzer", None) if api else None
        cancel = getattr(analyzer, "cancel", None)
        if callable(cancel):
            cancel()
        self._status.configure(text="Arrêt en cours…", text_color=get_color("warning"))

    def _on_progress(self, done: int, total: int, current: str) -> None:
        if total > 0:
            self._progress.set(done / total)
        self._status.configure(text=f"{fmt_int(done)} / {fmt_int(total)} — {current}", text_color=get_color("fg"))

    def _on_result(self, res: dict[str, Any]) -> None:
        ok = res.get("success") if isinstance(res, dict) else getattr(res, "success", True)
        symbol = "✓" if ok else "✗"
        path = res.get("file_path", "") if isinstance(res, dict) else getattr(res, "file_path", "")
        self._append_results(f"{symbol} {path}\n")

    def _on_complete(self, result: dict[str, Any]) -> None:
        self._processing = False
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._progress.set(1)
        completed = result.get("completed", 0)
        failed = result.get("failed", 0)
        skipped = result.get("skipped", 0)
        rate = result.get("success_rate", 0)
        summary = (
            f"\n{'-' * 40}\n"
            f"Terminé : {fmt_int(completed)} succès, {fmt_int(failed)} échecs, "
            f"{fmt_int(skipped)} ignorés ({rate:.1f}%).\n"
        )
        self._append_results(summary)
        self._status.configure(text="Analyse terminée", text_color=get_color("success"))
        self.app.toasts.show(f"Analyse terminée : {fmt_int(completed)} succès.", kind="success")

    def _on_failed(self, err: str) -> None:
        self._processing = False
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._status.configure(text="Erreur", text_color=get_color("error"))
        self._append_results(f"\nERREUR : {err}\n")
        self.app.toasts.show(f"Échec de l'analyse : {err}", kind="error")

    def _set_results(self, text: str) -> None:
        self._results.configure(state="normal")
        self._results.delete("1.0", "end")
        self._results.insert("1.0", text)
        self._results.configure(state="disabled")

    def _append_results(self, text: str) -> None:
        self._results.configure(state="normal")
        self._results.insert("end", text)
        self._results.see("end")
        self._results.configure(state="disabled")
