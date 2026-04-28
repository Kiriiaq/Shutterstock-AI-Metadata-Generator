"""
Page Source — Sélection du dossier et options de traitement.
"""

import customtkinter as ctk
from tkinter import filedialog
from pathlib import Path
from typing import Callable
from ...core.params import ShutterstockParams, PARAMS_META


COLORS = {"text_muted": ("#64748B", "#94A3B8"), "border": ("#E2E8F0", "#334155"),
           "success": ("#10B981", "#34D399"), "warning": ("#F59E0B", "#FBBF24"),
           "error": ("#EF4444", "#F87171")}


class PageSource(ctk.CTkScrollableFrame):
    """Page de sélection source."""

    PAGE_KEY = "source"

    def __init__(self, parent, params: ShutterstockParams, on_change: Callable, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.params = params
        self._on_change = on_change

        self.folder_var = ctk.StringVar(value=params.source_folder)
        self.prefilter_var = ctk.BooleanVar(value=params.prefilter_enabled)
        self.resume_var = ctk.BooleanVar(value=params.resume_mode)
        self.skip_var = ctk.BooleanVar(value=params.skip_analyzed)

        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(self, text="Source", font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(self, text="Sélectionnez le dossier contenant vos photos à analyser",
                     font=ctk.CTkFont(size=11), text_color=COLORS["text_muted"]).pack(anchor="w", pady=(0, 20))

        # Dossier
        card = ctk.CTkFrame(self, corner_radius=12, border_width=1, border_color=COLORS["border"])
        card.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(card, text="📂  Dossier source",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=16, pady=(16, 8))

        entry_frame = ctk.CTkFrame(card, fg_color="transparent")
        entry_frame.pack(fill="x", padx=16, pady=(0, 8))

        self.folder_entry = ctk.CTkEntry(entry_frame, textvariable=self.folder_var,
                                         placeholder_text="Ex: C:/Mes_Photos", height=34)
        self.folder_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(entry_frame, text="Parcourir", width=90, height=34,
                      command=self._browse).pack(side="right")

        self.info_label = ctk.CTkLabel(card, text="", font=ctk.CTkFont(size=10),
                                       text_color=COLORS["text_muted"])
        self.info_label.pack(anchor="w", padx=16, pady=(0, 16))
        self._update_info()

        # Options
        opts_card = ctk.CTkFrame(self, corner_radius=12, border_width=1, border_color=COLORS["border"])
        opts_card.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(opts_card, text="⚡  Options de traitement",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=16, pady=(16, 8))

        opts = ctk.CTkFrame(opts_card, fg_color="transparent")
        opts.pack(fill="x", padx=16, pady=(0, 16))

        ctk.CTkCheckBox(opts, text="Pré-filtrer (résolution, taille, format)", variable=self.prefilter_var,
                        command=lambda: self._on_change("prefilter_enabled", self.prefilter_var.get())
                        ).pack(fill="x", pady=3)
        ctk.CTkCheckBox(opts, text="Reprendre un traitement interrompu", variable=self.resume_var,
                        command=lambda: self._on_change("resume_mode", self.resume_var.get())
                        ).pack(fill="x", pady=3)
        ctk.CTkCheckBox(opts, text="Ignorer les photos déjà analysées", variable=self.skip_var,
                        command=lambda: self._on_change("skip_analyzed", self.skip_var.get())
                        ).pack(fill="x", pady=3)

        # Bind folder change
        self.folder_var.trace_add("write", lambda *_: self._on_folder_change())

    def _browse(self):
        folder = filedialog.askdirectory(title="Sélectionner le dossier source")
        if folder:
            self.folder_var.set(folder)

    def _on_folder_change(self):
        self._on_change("source_folder", self.folder_var.get())
        self._update_info()

    def _update_info(self):
        folder = self.folder_var.get()
        if not folder:
            self.info_label.configure(text="Aucun dossier sélectionné", text_color=COLORS["text_muted"])
            return
        path = Path(folder)
        if not path.exists():
            self.info_label.configure(text="⚠ Ce dossier n'existe pas", text_color=COLORS["error"])
            return
        # Compter images
        exts = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
        images = [f for f in path.iterdir() if f.suffix.lower() in exts]
        self.info_label.configure(
            text=f"✔ {len(images)} image(s) trouvée(s)",
            text_color=COLORS["success"]
        )

    def refresh(self, params: ShutterstockParams):
        self.params = params
        self.folder_var.set(params.source_folder)
        self.prefilter_var.set(params.prefilter_enabled)
        self.resume_var.set(params.resume_mode)
        self.skip_var.set(params.skip_analyzed)
        self._update_info()
