"""Expert microstock report view — multi-platform, AI-optional.

Opened as a modal from the IPTC editor panel via ``app.open_in_modal
("expert_report")``. Consumes ``app_state["expert_report_path"]`` (the
file the user clicked on) and optionally ``app_state["expert_report_paths"]``
(the full selection — used for the double CSV export).

Layout (single scrollable column for compatibility with shrinking
windows):

    [Header]
    [SCORES — 4 horizontal jauges]
    [TITRES Adobe / Shutterstock]
    [DESCRIPTION]
    [KEYWORDS — 10 prioritaires en accent, le reste en muted]
    [CATÉGORIES Adobe + Shutterstock]
    [RISQUES DE REJET — table]
    [AMÉLIORATIONS — liste à puces]
    [USAGES MARKETING — chips]
    [ACTIONS — Régénérer / Export CSV / Fermer]
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from tkinter import filedialog
from typing import TYPE_CHECKING, Any, List, Optional

import customtkinter as ctk

from app.components.empty_state import EmptyState
from app.config.theme import (
    RADIUS_MD,
    RADIUS_SM,
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    SPACE_XL,
    SPACE_XS,
    get_font,
    palette_pair,
)
from app.views.base_view import BaseView, _modal_header

if TYPE_CHECKING:
    from app.app import App

logger = logging.getLogger(__name__)


SEVERITY_COLOR = {
    "blocker": "error",
    "warning": "warning",
    "info": "fg_muted",
}


class ExpertReportView(BaseView):
    """Modal view showing the full expert microstock report."""

    view_id = "expert_report"

    def __init__(self, master: ctk.CTkFrame, *, app: "App") -> None:
        super().__init__(master)
        self.app = app
        self._report = None
        self._use_ai = ctk.BooleanVar(value=False)
        # Track whether we've already debited the Community quota for
        # this modal opening so a « Régénérer » click doesn't burn a
        # second teaser slot on the same image.
        self._quota_consumed_for_this_session = False
        self._build()

    # ------------------------------------------------------------------
    # Pro/Community helpers

    def _is_pro(self) -> bool:
        """True iff the active licence unlocks the expert report."""
        api = self.app.api
        lic = getattr(api, "license", None)
        return bool(lic and lic.has_feature("expert_report"))

    def _is_ai_pro(self) -> bool:
        """True iff the active licence unlocks AI enrichment."""
        api = self.app.api
        lic = getattr(api, "license", None)
        return bool(lic and lic.has_feature("ai_enrichment"))

    # ------------------------------------------------------------------

    def _build(self) -> None:
        if self.app.api is None:
            EmptyState(
                self,
                icon="🧾",
                title="Rapport indisponible",
                subtitle="Le backend n'est pas chargé.",
            ).grid(row=0, column=0, sticky="nsew")
            return

        path = self._get_target_path()
        if path is None:
            EmptyState(
                self,
                icon="🧾",
                title="Aucune image sélectionnée",
                subtitle=(
                    "Sélectionnez d'abord une image dans Sources, "
                    "puis utilisez le bouton « Rapport expert » de l'éditeur IPTC."
                ),
            ).grid(row=0, column=0, sticky="nsew")
            return

        # Community quota check — show the upsell screen before
        # consuming any compute when the teaser slots are depleted.
        if not self._is_pro():
            remaining = self.app.api.expert_report_quota_remaining()
            if remaining == 0:
                self._build_upsell_screen()
                return

        wrapper = ctk.CTkScrollableFrame(self, fg_color="transparent")
        wrapper.grid(row=0, column=0, sticky="nsew", padx=SPACE_XL, pady=SPACE_XL)
        wrapper.grid_columnconfigure(0, weight=1)

        _modal_header(wrapper, icon="🧾", title="Rapport expert microstock", row=0)

        # Subtitle = path
        self._path_label = ctk.CTkLabel(
            wrapper,
            text=str(path.name),
            font=get_font("body_strong"),
            text_color=palette_pair("fg"),
            anchor="w",
        )
        self._path_label.grid(row=1, column=0, sticky="ew", pady=(0, SPACE_SM))

        self._mode_label = ctk.CTkLabel(
            wrapper,
            text="Analyse en cours…",
            font=get_font("small"),
            text_color=palette_pair("fg_muted"),
            anchor="w",
        )
        self._mode_label.grid(row=2, column=0, sticky="ew", pady=(0, SPACE_MD))

        # Community quota banner — visible only when the user is not
        # Pro and still has free slots left. Sits between the mode
        # label and the report body so the user sees the cost upfront.
        if not self._is_pro():
            from src.modules.licensing import COMMUNITY_EXPERT_REPORT_QUOTA
            remaining = self.app.api.expert_report_quota_remaining()
            self._quota_banner = ctk.CTkLabel(
                wrapper,
                text=(
                    f"🎁 Aperçu Community · il reste {remaining}/"
                    f"{COMMUNITY_EXPERT_REPORT_QUOTA} rapport(s) gratuit(s). "
                    f"Passez Pro pour un usage illimité."
                ),
                font=get_font("small"),
                text_color=palette_pair("warning"),
                anchor="w",
                justify="left",
            )
            self._quota_banner.grid(
                row=3, column=0, sticky="ew", pady=(0, SPACE_SM)
            )

        # Body container — sections are rendered into here by ``_render``
        # once the worker thread finishes. The header above stays
        # visible during the spinner state.
        self._body = ctk.CTkFrame(wrapper, fg_color="transparent")
        self._body.grid(row=4, column=0, sticky="nsew")
        self._body.grid_columnconfigure(0, weight=1)

        # Actions row at the bottom — visible during loading too so the
        # user can cancel by closing.
        actions = ctk.CTkFrame(wrapper, fg_color="transparent")
        actions.grid(row=5, column=0, sticky="ew", pady=(SPACE_LG, 0))
        actions.grid_columnconfigure(2, weight=1)

        ai_label = "Enrichir avec IA (Ollama)"
        ai_state = "normal"
        if not self._is_ai_pro():
            ai_label = "🔒 Enrichir avec IA — Pro"
            ai_state = "disabled"
            # Force-uncheck in case the user toggled it before the
            # licence was revoked.
            self._use_ai.set(False)

        ctk.CTkCheckBox(
            actions,
            text=ai_label,
            variable=self._use_ai,
            font=get_font("small"),
            state=ai_state,
        ).grid(row=0, column=0, sticky="w")

        self._refresh_btn = ctk.CTkButton(
            actions,
            text="Régénérer",
            width=110,
            height=28,
            fg_color=palette_pair("bg_hover"),
            hover_color=palette_pair("bg_active"),
            text_color=palette_pair("fg"),
            border_width=1,
            border_color=palette_pair("border"),
            command=self._regenerate,
        )
        self._refresh_btn.grid(row=0, column=1, padx=SPACE_SM)

        self._export_btn = ctk.CTkButton(
            actions,
            text="Exporter CSV (Adobe + Shutterstock)…",
            height=28,
            fg_color=palette_pair("accent"),
            hover_color=palette_pair("accent_hover"),
            text_color=palette_pair("accent_fg"),
            font=get_font("body_strong"),
            command=self._export_csv,
            state="disabled",
        )
        self._export_btn.grid(row=0, column=3, sticky="e")

        self._build_loading_placeholder()
        self._target_path = path
        self._launch_build()

    # ------------------------------------------------------------------
    # Upsell — Community quota exhausted

    def _build_upsell_screen(self) -> None:
        """Render the Pro pitch when the Community quota is depleted.

        Replaces the normal modal body. Kept intentionally short and
        action-oriented — the user already knows what they get, the
        question is whether to pay. One primary CTA (Acheter Pro),
        one secondary (Activer une clé existante), and a discreet
        Annuler that closes the modal.
        """
        from src.modules.licensing import COMMUNITY_EXPERT_REPORT_QUOTA

        wrapper = ctk.CTkScrollableFrame(self, fg_color="transparent")
        wrapper.grid(row=0, column=0, sticky="nsew", padx=SPACE_XL, pady=SPACE_XL)
        wrapper.grid_columnconfigure(0, weight=1)

        _modal_header(wrapper, icon="🔒", title="Passez en édition Pro", row=0)

        ctk.CTkLabel(
            wrapper,
            text=(
                f"Vous avez utilisé vos {COMMUNITY_EXPERT_REPORT_QUOTA} "
                f"aperçus gratuits du rapport expert."
            ),
            font=get_font("body_strong"),
            text_color=palette_pair("fg"),
            anchor="w",
            justify="left",
            wraplength=620,
        ).grid(row=1, column=0, sticky="ew", pady=(0, SPACE_MD))

        # Card listing the Pro benefits — short, direct, no bullet padding.
        card = ctk.CTkFrame(
            wrapper,
            fg_color=palette_pair("bg_elevated"),
            corner_radius=RADIUS_MD,
            border_width=1,
            border_color=palette_pair("border"),
        )
        card.grid(row=2, column=0, sticky="ew", pady=(0, SPACE_LG))
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            card,
            text="Édition Pro — Évaluation qualité illimitée",
            font=get_font("body_strong"),
            text_color=palette_pair("accent"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=SPACE_MD, pady=(SPACE_MD, SPACE_XS))

        benefits = [
            "✓ Rapport expert illimité — 4 scores, risques, améliorations",
            "✓ Export double CSV Adobe Stock + Shutterstock en un clic",
            "✓ Enrichissement IA via Ollama local (LLaMA Vision, LLaVA…)",
            "✓ Anti-stuffing automatique des mots-clés (marques, padding)",
            "✓ Batch illimité (> 50 images par run)",
            "✓ Marketing uses, profils acheteurs et tendances par image",
        ]
        for i, line in enumerate(benefits, start=1):
            ctk.CTkLabel(
                card, text=line, font=get_font("body"),
                text_color=palette_pair("fg"),
                anchor="w",
            ).grid(row=i, column=0, sticky="ew", padx=SPACE_MD, pady=1)

        ctk.CTkLabel(
            card,
            text="29 €/an  ·  79 € à vie  ·  paiement Gumroad",
            font=get_font("small"),
            text_color=palette_pair("fg_muted"),
            anchor="w",
        ).grid(row=99, column=0, sticky="ew", padx=SPACE_MD, pady=(SPACE_SM, SPACE_MD))

        actions = ctk.CTkFrame(wrapper, fg_color="transparent")
        actions.grid(row=3, column=0, sticky="ew", pady=(0, SPACE_SM))
        actions.grid_columnconfigure(2, weight=1)

        ctk.CTkButton(
            actions,
            text="Acheter Pro →",
            height=32,
            fg_color=palette_pair("accent"),
            hover_color=palette_pair("accent_hover"),
            text_color=palette_pair("accent_fg"),
            font=get_font("body_strong"),
            command=self._open_gumroad,
        ).grid(row=0, column=0, sticky="w", padx=(0, SPACE_SM))

        ctk.CTkButton(
            actions,
            text="J'ai déjà une clé…",
            height=32,
            fg_color=palette_pair("bg_hover"),
            hover_color=palette_pair("bg_active"),
            text_color=palette_pair("fg"),
            border_width=1,
            border_color=palette_pair("border"),
            command=self._open_settings_for_license,
        ).grid(row=0, column=1, sticky="w")

        ctk.CTkLabel(
            wrapper,
            text=(
                "Astuce : la version Community reste utile pour scanner "
                "vos dossiers, éditer les IPTC manuellement et exporter "
                "un CSV simple par plateforme."
            ),
            font=get_font("small"),
            text_color=palette_pair("fg_muted"),
            anchor="w",
            justify="left",
            wraplength=620,
        ).grid(row=4, column=0, sticky="ew", pady=(SPACE_MD, 0))

    def _open_gumroad(self) -> None:
        """Open the Gumroad listing in the user's browser."""
        import webbrowser
        # Placeholder — replace with the real URL once the product is live.
        url = "https://gumroad.com/l/shutterstockanalyzer-pro"
        try:
            webbrowser.open(url, new=2)
        except Exception:  # noqa: BLE001
            logger.exception("Could not open Gumroad URL")
            self.app.toasts.show(f"Ouvrez manuellement : {url}", kind="info")

    def _open_settings_for_license(self) -> None:
        """Switch to the Settings modal so the user can paste a key."""
        try:
            self.app.open_in_modal("settings")
        except Exception:  # noqa: BLE001
            logger.exception("Could not open settings modal")

    # ------------------------------------------------------------------
    # Helpers

    def _get_target_path(self) -> Optional[Path]:
        """Resolve which file we're reporting on.

        Order of fallback: explicit ``expert_report_path`` (set when the
        user clicked the panel button), then the first of
        ``selected_paths``, then None (empty state).
        """
        state = self.app.app_state
        raw = state.get("expert_report_path")
        if raw:
            return Path(raw)
        selected = state.get("selected_paths") or []
        if selected:
            return Path(selected[0])
        return None

    def _build_loading_placeholder(self) -> None:
        ctk.CTkLabel(
            self._body,
            text="⏳  Calcul du rapport…",
            font=get_font("body"),
            text_color=palette_pair("fg_muted"),
        ).grid(row=0, column=0, sticky="w", pady=SPACE_LG)

    def _launch_build(self) -> None:
        api = self.app.api
        path = self._target_path
        use_ai = bool(self._use_ai.get())
        self._refresh_btn.configure(state="disabled")
        self._export_btn.configure(state="disabled")
        self._mode_label.configure(
            text=("Analyse en cours… (mode IA)" if use_ai else "Analyse en cours… (mode rapide, sans IA)"),
            text_color=palette_pair("fg_muted"),
        )
        threading.Thread(
            target=self._build_worker,
            args=(api, path, use_ai),
            daemon=True,
        ).start()

    def _build_worker(self, api: Any, path: Path, use_ai: bool) -> None:
        try:
            report = api.build_expert_report(path, use_ai=use_ai)
        except Exception as exc:  # noqa: BLE001
            logger.exception("build_expert_report failed")
            self.after(0, lambda e=str(exc): self._on_failure(e))
            return
        self.after(0, lambda r=report: self._on_success(r))

    def _on_success(self, report: Any) -> None:
        self._report = report
        self._refresh_btn.configure(state="normal")
        # Community users can't trigger the dual-CSV export from this
        # modal — it's a Pro feature now. Keep the button visible but
        # disabled, with a 🔒 hint so the upsell is obvious.
        if self._is_dual_csv_pro():
            self._export_btn.configure(state="normal")
        else:
            self._export_btn.configure(
                state="disabled",
                text="🔒 Exporter CSV double — Pro",
            )
        mode_label = {
            "heuristic": "Rapide (sans IA)",
            "ai": "Enrichi IA",
            "hybrid": "Rapide + IA",
        }.get(getattr(report, "source", "heuristic"), "Rapide")
        self._mode_label.configure(
            text=f"Mode : {mode_label}",
            text_color=palette_pair("fg_muted"),
        )
        self._render(report)

        # Debit one Community slot AFTER the report renders so a
        # failed build doesn't burn a teaser. Re-generating the same
        # image (via « Régénérer ») within the same modal opening
        # doesn't re-debit — that would feel punitive.
        if not self._is_pro() and not self._quota_consumed_for_this_session:
            try:
                remaining = self.app.api.consume_expert_report_quota()
            except Exception:  # noqa: BLE001
                logger.exception("consume_expert_report_quota failed")
                remaining = -1
            self._quota_consumed_for_this_session = True
            self._update_quota_banner(remaining)

    def _is_dual_csv_pro(self) -> bool:
        """True iff the active licence unlocks the dual CSV export."""
        api = self.app.api
        lic = getattr(api, "license", None)
        return bool(lic and lic.has_feature("dual_csv_export"))

    def _update_quota_banner(self, remaining: int) -> None:
        """Refresh the quota label after consumption."""
        banner = getattr(self, "_quota_banner", None)
        if banner is None:
            return
        from src.modules.licensing import COMMUNITY_EXPERT_REPORT_QUOTA
        if remaining <= 0:
            banner.configure(
                text=(
                    f"🛑 Dernier aperçu Community consommé "
                    f"({COMMUNITY_EXPERT_REPORT_QUOTA}/"
                    f"{COMMUNITY_EXPERT_REPORT_QUOTA}). "
                    f"Passez Pro pour un usage illimité."
                ),
                text_color=palette_pair("error"),
            )
        else:
            banner.configure(
                text=(
                    f"🎁 Aperçu Community · il reste {remaining}/"
                    f"{COMMUNITY_EXPERT_REPORT_QUOTA} rapport(s) gratuit(s)."
                ),
            )

    def _on_failure(self, error: str) -> None:
        self._refresh_btn.configure(state="normal")
        self._mode_label.configure(
            text=f"Échec : {error}",
            text_color=palette_pair("error"),
        )
        for w in self._body.winfo_children():
            w.destroy()
        ctk.CTkLabel(
            self._body,
            text=f"Impossible de construire le rapport.\n{error}",
            font=get_font("body"),
            text_color=palette_pair("error"),
            justify="left",
            wraplength=600,
        ).grid(row=0, column=0, sticky="w", pady=SPACE_LG)

    def _regenerate(self) -> None:
        for w in self._body.winfo_children():
            w.destroy()
        self._build_loading_placeholder()
        self._launch_build()

    def _export_csv(self) -> None:
        if self._report is None:
            return
        # Pro gate — dual CSV is part of the Pro bundle.
        if not self._is_dual_csv_pro():
            self.app.toasts.show(
                "L'export double CSV (Adobe + Shutterstock) est réservé à "
                "l'édition Pro. Activez votre licence dans Paramètres → Licence.",
                kind="warning",
                timeout_ms=6000,
            )
            return
        # Multi-file: prefer the workspace selection if any, else the
        # single current report.
        selected = self.app.app_state.get("selected_paths") or []
        reports = [self._report]
        if selected and len(selected) > 1:
            # Build the missing reports in background; for simplicity
            # we do them sequentially inline — the user is already
            # waiting for the file dialog.
            extras: List[Any] = []
            for p in selected:
                p = Path(p)
                if p == self._target_path:
                    continue
                try:
                    extras.append(self.app.api.build_expert_report(p, use_ai=False))
                except Exception:  # noqa: BLE001
                    logger.warning("build_expert_report skipped for %s", p, exc_info=True)
            reports = [self._report, *extras]

        out_dir = filedialog.askdirectory(title="Dossier d'export CSV")
        if not out_dir:
            return
        try:
            result = self.app.api.export_double_csv(reports, Path(out_dir), basename="metadata")
        except Exception as exc:  # noqa: BLE001
            logger.exception("export_double_csv failed")
            self.app.toasts.show(f"Échec export : {exc}", kind="error")
            return
        self.app.toasts.show(
            f"{result.row_count} ligne(s) écrite(s) — Adobe + Shutterstock.",
            kind="success",
        )

    # ------------------------------------------------------------------
    # Rendering

    def _render(self, report: Any) -> None:
        for w in self._body.winfo_children():
            w.destroy()

        row = 0
        row = self._render_scores(report, row)
        row = self._render_titles(report, row)
        row = self._render_description(report, row)
        row = self._render_keywords(report, row)
        row = self._render_categories(report, row)
        row = self._render_warnings(report, row)
        row = self._render_risks(report, row)
        row = self._render_improvements(report, row)
        row = self._render_marketing(report, row)
        row = self._render_buyers_trends(report, row)

    # ----- section primitives -----

    def _section_header(self, parent: ctk.CTkFrame, *, row: int, title: str) -> None:
        ctk.CTkLabel(
            parent,
            text=title,
            font=get_font("body_strong"),
            text_color=palette_pair("fg_subtle"),
            anchor="w",
        ).grid(row=row, column=0, sticky="ew", pady=(SPACE_LG, SPACE_XS))

    def _card(self, *, row: int) -> ctk.CTkFrame:
        card = ctk.CTkFrame(
            self._body,
            fg_color=palette_pair("bg_elevated"),
            corner_radius=RADIUS_MD,
            border_width=1,
            border_color=palette_pair("border"),
        )
        card.grid(row=row, column=0, sticky="ew", pady=(0, SPACE_SM))
        card.grid_columnconfigure(0, weight=1)
        return card

    # ----- 1. Scores -----

    def _render_scores(self, report: Any, row: int) -> int:
        self._section_header(self._body, row=row, title="1 · SCORE GLOBAL")
        row += 1
        card = self._card(row=row)
        scores = report.scores

        gauges = [
            ("Potentiel commercial", scores.commercial, "success"),
            ("Qualité technique", scores.technical, "success"),
            ("Potentiel SEO", scores.seo, "accent"),
            ("Risque de rejet", scores.rejection_risk, "error"),
        ]
        for i, (label, value, color_key) in enumerate(gauges):
            self._gauge(card, label=label, value=value, color_key=color_key, row=i)
        return row + 1

    def _gauge(
        self,
        parent: ctk.CTkFrame,
        *,
        label: str,
        value: int,
        color_key: str,
        row: int,
    ) -> None:
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.grid(row=row, column=0, sticky="ew", padx=SPACE_MD, pady=(SPACE_XS, SPACE_XS))
        wrap.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            wrap,
            text=label,
            font=get_font("body"),
            text_color=palette_pair("fg"),
            anchor="w",
            width=180,
        ).grid(row=0, column=0, sticky="w")

        bar = ctk.CTkProgressBar(
            wrap,
            height=12,
            corner_radius=RADIUS_SM,
            progress_color=palette_pair(color_key),
            fg_color=palette_pair("bg_hover"),
            border_color=palette_pair("border"),
            border_width=1,
        )
        bar.set(max(0.0, min(1.0, value / 10.0)))
        bar.grid(row=0, column=1, sticky="ew", padx=SPACE_SM)

        ctk.CTkLabel(
            wrap,
            text=f"{value} / 10",
            font=get_font("body_strong"),
            text_color=palette_pair("fg"),
            width=60,
            anchor="e",
        ).grid(row=0, column=2, sticky="e")

    # ----- 2. Titles -----

    def _render_titles(self, report: Any, row: int) -> int:
        self._section_header(self._body, row=row, title="2 · TITRES SEO")
        row += 1
        card = self._card(row=row)
        self._labeled_text(card, label="Adobe Stock", text=report.title_adobe or "—", row=0)
        self._labeled_text(card, label="Shutterstock", text=report.title_shutterstock or "—", row=1)
        return row + 1

    def _labeled_text(self, parent: ctk.CTkFrame, *, label: str, text: str, row: int) -> None:
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.grid(row=row, column=0, sticky="ew", padx=SPACE_MD, pady=SPACE_XS)
        wrap.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            wrap,
            text=label,
            font=get_font("small"),
            text_color=palette_pair("fg_muted"),
            width=120,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            wrap,
            text=text,
            font=get_font("body"),
            text_color=palette_pair("fg"),
            anchor="w",
            justify="left",
            wraplength=520,
        ).grid(row=0, column=1, sticky="ew")

    # ----- 3. Description -----

    def _render_description(self, report: Any, row: int) -> int:
        self._section_header(self._body, row=row, title="3 · DESCRIPTION COMMERCIALE")
        row += 1
        card = self._card(row=row)
        text = report.description or "—"
        ctk.CTkLabel(
            card,
            text=text,
            font=get_font("body"),
            text_color=palette_pair("fg"),
            anchor="w",
            justify="left",
            wraplength=650,
        ).grid(row=0, column=0, sticky="ew", padx=SPACE_MD, pady=SPACE_SM)
        return row + 1

    # ----- 4. Keywords -----

    def _render_keywords(self, report: Any, row: int) -> int:
        keywords = list(report.keywords or [])
        self._section_header(
            self._body,
            row=row,
            title=f"4 · KEYWORDS ({len(keywords)}/50) — 10 prioritaires en accent",
        )
        row += 1
        card = self._card(row=row)

        if not keywords:
            ctk.CTkLabel(
                card,
                text="Aucun mot-clé.",
                font=get_font("body"),
                text_color=palette_pair("fg_muted"),
            ).grid(row=0, column=0, sticky="w", padx=SPACE_MD, pady=SPACE_SM)
            return row + 1

        # Use a wrapping frame for chips. Tkinter doesn't have native
        # flow layout — we lay chips in rows of N, breaking on width.
        # Coarse approach: 5 chips per row in priority order, then
        # 6 per row for the rest. Good enough for a 1100px modal.
        chips_wrap = ctk.CTkFrame(card, fg_color="transparent")
        chips_wrap.grid(row=0, column=0, sticky="ew", padx=SPACE_MD, pady=SPACE_SM)

        for i, kw in enumerate(keywords):
            is_priority = i < 10
            chip = ctk.CTkLabel(
                chips_wrap,
                text=kw,
                font=get_font("small"),
                text_color=palette_pair("accent_fg" if is_priority else "fg"),
                fg_color=palette_pair("accent" if is_priority else "bg_hover"),
                corner_radius=RADIUS_SM,
                padx=8,
                pady=2,
            )
            # Manual grid layout: 5 columns priority, 6 columns rest.
            cols = 5 if is_priority else 6
            r = i // cols
            c = i % cols
            chip.grid(row=r, column=c, padx=2, pady=2, sticky="w")
        return row + 1

    # ----- 5. Categories -----

    def _render_categories(self, report: Any, row: int) -> int:
        self._section_header(self._body, row=row, title="5 · CATÉGORIES RECOMMANDÉES")
        row += 1
        card = self._card(row=row)

        self._labeled_text(
            card,
            label="Adobe — principale",
            text=report.category_adobe_primary or "—",
            row=0,
        )
        self._labeled_text(
            card,
            label="Adobe — secondaire",
            text=report.category_adobe_secondary or "—",
            row=1,
        )
        self._labeled_text(
            card,
            label="Shutterstock",
            text=", ".join(report.categories_shutterstock) or "—",
            row=2,
        )
        return row + 1

    # ----- 5b. Platform warnings -----

    def _render_warnings(self, report: Any, row: int) -> int:
        adobe = list(report.adobe_warnings or [])
        sh = list(report.shutterstock_warnings or [])
        if not adobe and not sh:
            return row

        self._section_header(self._body, row=row, title="⚠  ALERTES PLATEFORMES (informatives)")
        row += 1
        card = self._card(row=row)
        sub_row = 0
        for w in adobe + sh:
            ctk.CTkLabel(
                card,
                text=f"• {w}",
                font=get_font("small"),
                text_color=palette_pair("warning"),
                anchor="w",
                justify="left",
                wraplength=650,
            ).grid(row=sub_row, column=0, sticky="w", padx=SPACE_MD, pady=1)
            sub_row += 1
        return row + 1

    # ----- 6. Rejection risks -----

    def _render_risks(self, report: Any, row: int) -> int:
        risks = list(report.rejection_risks or [])
        self._section_header(
            self._body,
            row=row,
            title=f"6 · RISQUES DE REJET ({len(risks)})",
        )
        row += 1
        card = self._card(row=row)
        if not risks:
            ctk.CTkLabel(
                card,
                text="Aucun risque détecté côté pipeline. Le reviewer plateforme tranche.",
                font=get_font("body"),
                text_color=palette_pair("success"),
                anchor="w",
                wraplength=650,
            ).grid(row=0, column=0, sticky="w", padx=SPACE_MD, pady=SPACE_SM)
            return row + 1

        for idx, risk in enumerate(risks):
            color = SEVERITY_COLOR.get(risk.severity, "fg_muted")
            block = ctk.CTkFrame(card, fg_color="transparent")
            block.grid(row=idx, column=0, sticky="ew", padx=SPACE_MD, pady=SPACE_XS)
            block.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(
                block,
                text=f"[{risk.severity.upper()}]",
                font=get_font("small"),
                text_color=palette_pair(color),
                width=90,
                anchor="w",
            ).grid(row=0, column=0, sticky="w")
            ctk.CTkLabel(
                block,
                text=risk.issue,
                font=get_font("body_strong"),
                text_color=palette_pair("fg"),
                anchor="w",
                justify="left",
                wraplength=540,
            ).grid(row=0, column=1, sticky="w")
            if risk.cause:
                ctk.CTkLabel(
                    block,
                    text=f"Cause : {risk.cause}",
                    font=get_font("small"),
                    text_color=palette_pair("fg_muted"),
                    anchor="w",
                    justify="left",
                    wraplength=540,
                ).grid(row=1, column=1, sticky="w")
            if risk.fix:
                ctk.CTkLabel(
                    block,
                    text=f"Correction : {risk.fix}",
                    font=get_font("small"),
                    text_color=palette_pair("fg"),
                    anchor="w",
                    justify="left",
                    wraplength=540,
                ).grid(row=2, column=1, sticky="w")
        return row + 1

    # ----- 7. Improvements -----

    def _render_improvements(self, report: Any, row: int) -> int:
        improvements = list(report.improvements or [])
        if not improvements:
            return row
        self._section_header(self._body, row=row, title="7 · AMÉLIORATIONS RECOMMANDÉES")
        row += 1
        card = self._card(row=row)
        for idx, item in enumerate(improvements):
            ctk.CTkLabel(
                card,
                text=f"• {item}",
                font=get_font("body"),
                text_color=palette_pair("fg"),
                anchor="w",
                justify="left",
                wraplength=650,
            ).grid(row=idx, column=0, sticky="w", padx=SPACE_MD, pady=1)
        return row + 1

    # ----- 8. Marketing -----

    def _render_marketing(self, report: Any, row: int) -> int:
        uses = list(report.marketing_uses or [])
        if not uses:
            return row
        self._section_header(self._body, row=row, title="8 · USAGES MARKETING POSSIBLES")
        row += 1
        card = self._card(row=row)
        chips_wrap = ctk.CTkFrame(card, fg_color="transparent")
        chips_wrap.grid(row=0, column=0, sticky="ew", padx=SPACE_MD, pady=SPACE_SM)
        for i, use in enumerate(uses):
            chip = ctk.CTkLabel(
                chips_wrap,
                text=use,
                font=get_font("small"),
                text_color=palette_pair("fg"),
                fg_color=palette_pair("bg_hover"),
                corner_radius=RADIUS_SM,
                padx=8,
                pady=2,
            )
            chip.grid(row=i // 4, column=i % 4, padx=2, pady=2, sticky="w")
        return row + 1

    # ----- Buyer profiles + trends -----

    def _render_buyers_trends(self, report: Any, row: int) -> int:
        buyers = list(report.buyer_profiles or [])
        trends = list(report.trends or [])
        if not buyers and not trends:
            return row
        self._section_header(self._body, row=row, title="ACHETEURS CIBLES & TENDANCES")
        row += 1
        card = self._card(row=row)
        if buyers:
            self._labeled_text(card, label="Profils acheteurs", text=", ".join(buyers), row=0)
        if trends:
            self._labeled_text(card, label="Tendances visuelles", text=", ".join(trends), row=1)
        return row + 1
