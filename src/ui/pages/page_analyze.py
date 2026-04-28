"""
Page Analyse — Lancement et suivi de l'analyse IA.
"""

import customtkinter as ctk
from typing import Callable
from ...core.params import ShutterstockParams


COLORS = {"text_muted": ("#64748B", "#94A3B8"), "border": ("#E2E8F0", "#334155"),
           "success": ("#10B981", "#34D399"), "warning": ("#F59E0B", "#FBBF24"),
           "error": ("#EF4444", "#F87171")}


class PageAnalyze(ctk.CTkScrollableFrame):
    """Page d'analyse des images."""

    PAGE_KEY = "analyse"

    def __init__(self, parent, params: ShutterstockParams, on_change: Callable,
                 analyzer=None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.params = params
        self._on_change = on_change
        self.analyzer = analyzer
        self._is_running = False
        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(self, text="Analyse", font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(self, text="Lancez l'analyse IA de vos images",
                     font=ctk.CTkFont(size=11), text_color=COLORS["text_muted"]).pack(anchor="w", pady=(0, 20))

        # Bouton lancer
        self.analyze_btn = ctk.CTkButton(
            self, text="▶  Lancer l'analyse",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=50, corner_radius=10,
            command=self._start_analysis
        )
        self.analyze_btn.pack(fill="x", pady=(0, 15))

        # Progression
        prog_card = ctk.CTkFrame(self, corner_radius=12, border_width=1, border_color=COLORS["border"])
        prog_card.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(prog_card, text="📊  Progression",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=16, pady=(16, 8))

        self.progress_bar = ctk.CTkProgressBar(prog_card, height=12, corner_radius=6)
        self.progress_bar.pack(fill="x", padx=16, pady=(0, 8))
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(prog_card, text="En attente",
                                           font=ctk.CTkFont(size=11), text_color=COLORS["text_muted"])
        self.progress_label.pack(anchor="w", padx=16, pady=(0, 16))

        # Statistiques
        stats_card = ctk.CTkFrame(self, corner_radius=12, border_width=1, border_color=COLORS["border"])
        stats_card.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(stats_card, text="📈  Statistiques",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=16, pady=(16, 8))

        stats_frame = ctk.CTkFrame(stats_card, fg_color="transparent")
        stats_frame.pack(fill="x", padx=16, pady=(0, 16))

        self.stats = {}
        for key, label, color in [("total", "Total", "#3498db"), ("success", "Succès", "#10B981"),
                                   ("failed", "Échecs", "#EF4444"), ("invalid", "Invalides", "#F59E0B")]:
            col = ctk.CTkFrame(stats_frame, fg_color="transparent")
            col.pack(side="left", padx=15, expand=True)
            val = ctk.CTkLabel(col, text="0", font=ctk.CTkFont(size=24, weight="bold"), text_color=color)
            val.pack()
            ctk.CTkLabel(col, text=label, font=ctk.CTkFont(size=9), text_color=COLORS["text_muted"]).pack()
            self.stats[key] = val

    def _start_analysis(self):
        if self._is_running:
            return
        self._is_running = True
        self.analyze_btn.configure(state="disabled", text="Analyse en cours...")
        self.progress_label.configure(text="Démarrage de l'analyse...")
        # L'analyse réelle serait lancée ici via self.analyzer

    def update_progress(self, total, success, failed, invalid, message=""):
        self.stats["total"].configure(text=str(total))
        self.stats["success"].configure(text=str(success))
        self.stats["failed"].configure(text=str(failed))
        self.stats["invalid"].configure(text=str(invalid))
        if total > 0:
            self.progress_bar.set(success / total)
        if message:
            self.progress_label.configure(text=message)

    def analysis_complete(self):
        self._is_running = False
        self.analyze_btn.configure(state="normal", text="▶  Lancer l'analyse")
        self.progress_label.configure(text="✔ Analyse terminée !")

    def refresh(self, params: ShutterstockParams):
        self.params = params
