"""Home / dashboard view — entry point with stats and quick start."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import customtkinter as ctk

from app.config.theme import (
    RADIUS_LG,
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


class HomeView(BaseView):
    view_id = "home"

    def __init__(self, master: ctk.CTkFrame, *, app: "App") -> None:
        super().__init__(master)
        self.app = app
        self._stat_labels: dict[str, ctk.CTkLabel] = {}
        self._build()

    def _build(self) -> None:
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.grid(row=0, column=0, sticky="nsew", padx=SPACE_XL, pady=SPACE_XL)
        scroll.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            scroll,
            text="Tableau de bord",
            font=get_font("h1"),
            text_color=get_color("fg"),
        ).grid(row=0, column=0, sticky="w", pady=(0, SPACE_SM))
        ctk.CTkLabel(
            scroll,
            text="Aperçu de l'activité et accès rapide aux étapes de production.",
            font=get_font("body"),
            text_color=get_color("fg_muted"),
        ).grid(row=1, column=0, sticky="w", pady=(0, SPACE_LG))

        self._build_status_row(scroll, row=2)
        self._build_stats_grid(scroll, row=3)
        self._build_quick_actions(scroll, row=4)

    def _build_status_row(self, parent: ctk.CTkFrame, row: int) -> None:
        bar = ctk.CTkFrame(parent, fg_color=get_color("bg_elevated"), corner_radius=RADIUS_LG)
        bar.grid(row=row, column=0, sticky="ew", pady=(0, SPACE_LG))

        api = self.app.api
        ai_ok = bool(api) and api.exiftool_available  # rough heuristic; refined in on_enter
        exif_ok = bool(api) and api.exiftool_available
        self._add_status_chip(
            bar, "ExifTool", "OK" if exif_ok else "Absent", get_color("success") if exif_ok else get_color("warning")
        )
        self._add_status_chip(
            bar,
            "Backend",
            "Disponible" if api else "Indisponible",
            get_color("success") if api else get_color("warning"),
        )
        self._add_status_chip(bar, "IA", "À vérifier", get_color("fg_muted"))
        self._ai_chip = bar.winfo_children()[-1]

        _ = ai_ok

    def _add_status_chip(self, parent: ctk.CTkFrame, label: str, value: str, color: str) -> None:
        chip = ctk.CTkFrame(parent, fg_color="transparent")
        chip.pack(side="left", padx=SPACE_LG, pady=SPACE_MD)
        ctk.CTkLabel(chip, text=label, font=get_font("small"), text_color=get_color("fg_muted")).pack(anchor="w")
        ctk.CTkLabel(chip, text=value, font=get_font("body_strong"), text_color=color).pack(anchor="w")

    def _build_stats_grid(self, parent: ctk.CTkFrame, row: int) -> None:
        grid = ctk.CTkFrame(parent, fg_color="transparent")
        grid.grid(row=row, column=0, sticky="ew", pady=(0, SPACE_LG))
        for i in range(4):
            grid.grid_columnconfigure(i, weight=1, uniform="stat")

        for col, (key, label) in enumerate(
            [
                ("total_processed", "Images traitées"),
                ("with_metadata", "Avec métadonnées"),
                ("with_ai_analysis", "Analysées par IA"),
                ("recent_errors", "Erreurs (24 h)"),
            ]
        ):
            self._stat_card(grid, key, label, col)

    def _stat_card(self, parent: ctk.CTkFrame, key: str, label: str, col: int) -> None:
        card = ctk.CTkFrame(
            parent,
            fg_color=get_color("bg_elevated"),
            border_color=get_color("border"),
            border_width=1,
            corner_radius=RADIUS_LG,
        )
        card.grid(row=0, column=col, sticky="ew", padx=SPACE_SM)
        ctk.CTkLabel(card, text=label, font=get_font("small"), text_color=get_color("fg_muted"), anchor="w").pack(
            fill="x", padx=SPACE_LG, pady=(SPACE_MD, 0)
        )
        value_label = ctk.CTkLabel(card, text="—", font=get_font("h1"), text_color=get_color("fg"), anchor="w")
        value_label.pack(fill="x", padx=SPACE_LG, pady=(0, SPACE_MD))
        self._stat_labels[key] = value_label

    def _build_quick_actions(self, parent: ctk.CTkFrame, row: int) -> None:
        actions = ctk.CTkFrame(parent, fg_color="transparent")
        actions.grid(row=row, column=0, sticky="ew")
        for i in range(2):
            actions.grid_columnconfigure(i, weight=1, uniform="qa")

        for col, (label, target) in enumerate(
            [
                ("Scanner un dossier d'images", "sources"),
                ("Configurer le modèle IA", "ai_control"),
            ]
        ):
            ctk.CTkButton(
                actions,
                text=label,
                font=get_font("body_strong"),
                fg_color=get_color("accent"),
                hover_color=get_color("accent_hover"),
                text_color=get_color("accent_fg"),
                corner_radius=RADIUS_MD,
                height=48,
                command=lambda t=target: self.app.router.navigate_to(t),
            ).grid(row=0, column=col, sticky="ew", padx=SPACE_SM)

    def on_enter(self, **_kwargs: Any) -> None:
        api = self.app.api
        if api is None:
            for label in self._stat_labels.values():
                label.configure(text="—")
            return
        try:
            stats = api.get_statistics()
        except Exception:
            logger.exception("Could not load home stats")
            return
        for key, label in self._stat_labels.items():
            label.configure(text=fmt_int(int(stats.get(key, 0))))
