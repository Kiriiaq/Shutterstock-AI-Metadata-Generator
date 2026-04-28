"""
Fenêtre paramètres avancés Shutterstock Analyzer.
"""

import customtkinter as ctk
from typing import Callable
from ...core.params import ShutterstockParams


COLORS = {"text_muted": ("#64748B", "#94A3B8"), "border": ("#E2E8F0", "#334155"),
           "warning_bg": ("#FEF3C7", "#422006"), "warning_text": ("#92400E", "#FCD34D")}


class AdvancedSettingsWindow(ctk.CTkToplevel):
    """Fenêtre de paramètres avancés."""

    def __init__(self, parent, params: ShutterstockParams, on_change: Callable):
        super().__init__(parent)
        self.title("Paramètres avancés")
        self.geometry("520x480")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.params = params
        self._on_change = on_change
        self._build_ui()

    def _build_ui(self):
        # Warning banner
        warn = ctk.CTkFrame(self, fg_color=COLORS["warning_bg"], corner_radius=8)
        warn.pack(fill="x", padx=16, pady=(16, 10))
        ctk.CTkLabel(warn, text="⚠ Ces paramètres sont destinés aux utilisateurs expérimentés.",
                     font=ctk.CTkFont(size=10), text_color=COLORS["warning_text"]).pack(padx=12, pady=8)

        # Tabview
        self.tabs = ctk.CTkTabview(self, corner_radius=10)
        self.tabs.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        self._build_performance_tab()
        self._build_prefilter_tab()
        self._build_ftps_tab()
        self._build_debug_tab()

        # Footer
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=16, pady=(0, 16))

        ctk.CTkButton(footer, text="Réinitialiser tout", width=120, height=32,
                      fg_color="transparent", border_width=1,
                      command=self._reset_all).pack(side="left")
        ctk.CTkButton(footer, text="Fermer", width=100, height=32,
                      command=self.destroy).pack(side="right")

    def _build_performance_tab(self):
        tab = self.tabs.add("Performance")

        # GPU Layers
        ctk.CTkLabel(tab, text="Couches GPU", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", pady=(10, 2))
        ctk.CTkLabel(tab, text="Nombre de couches du modèle chargées sur le GPU (0 = CPU seul)",
                     font=ctk.CTkFont(size=9), text_color=COLORS["text_muted"]).pack(anchor="w", pady=(0, 5))

        self.gpu_var = ctk.IntVar(value=self.params.gpu_layers)
        gpu_frame = ctk.CTkFrame(tab, fg_color="transparent")
        gpu_frame.pack(fill="x", pady=(0, 12))
        self.gpu_slider = ctk.CTkSlider(gpu_frame, from_=0, to=100, number_of_steps=100,
                                         variable=self.gpu_var,
                                         command=lambda v: self._update("gpu_layers", int(v)))
        self.gpu_slider.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.gpu_label = ctk.CTkLabel(gpu_frame, textvariable=self.gpu_var, width=35,
                                       font=ctk.CTkFont(size=11))
        self.gpu_label.pack(side="right")

        # Cooldown
        ctk.CTkLabel(tab, text="Délai entre requêtes (s)", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", pady=(5, 2))
        ctk.CTkLabel(tab, text="Pause entre chaque analyse pour éviter la surcharge",
                     font=ctk.CTkFont(size=9), text_color=COLORS["text_muted"]).pack(anchor="w", pady=(0, 5))

        self.cooldown_var = ctk.DoubleVar(value=self.params.cooldown)
        cd_frame = ctk.CTkFrame(tab, fg_color="transparent")
        cd_frame.pack(fill="x", pady=(0, 12))
        ctk.CTkSlider(cd_frame, from_=0.5, to=10.0, number_of_steps=19,
                      variable=self.cooldown_var,
                      command=lambda v: self._update("cooldown", round(float(v), 1))
                      ).pack(side="left", fill="x", expand=True, padx=(0, 10))
        ctk.CTkLabel(cd_frame, textvariable=self.cooldown_var, width=35,
                     font=ctk.CTkFont(size=11)).pack(side="right")

        # Workers
        ctk.CTkLabel(tab, text="Workers parallèles", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", pady=(5, 2))
        ctk.CTkLabel(tab, text="Nombre de traitements simultanés",
                     font=ctk.CTkFont(size=9), text_color=COLORS["text_muted"]).pack(anchor="w", pady=(0, 5))

        self.workers_var = ctk.IntVar(value=self.params.workers)
        w_frame = ctk.CTkFrame(tab, fg_color="transparent")
        w_frame.pack(fill="x", pady=(0, 5))
        ctk.CTkSlider(w_frame, from_=1, to=8, number_of_steps=7,
                      variable=self.workers_var,
                      command=lambda v: self._update("workers", int(v))
                      ).pack(side="left", fill="x", expand=True, padx=(0, 10))
        ctk.CTkLabel(w_frame, textvariable=self.workers_var, width=35,
                     font=ctk.CTkFont(size=11)).pack(side="right")

    def _build_prefilter_tab(self):
        tab = self.tabs.add("Préfiltrage")

        # Min megapixels
        ctk.CTkLabel(tab, text="Résolution minimum (MP)", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", pady=(10, 2))
        ctk.CTkLabel(tab, text="Shutterstock exige minimum 4 mégapixels",
                     font=ctk.CTkFont(size=9), text_color=COLORS["text_muted"]).pack(anchor="w", pady=(0, 5))

        self.mp_var = ctk.DoubleVar(value=self.params.min_megapixels)
        mp_frame = ctk.CTkFrame(tab, fg_color="transparent")
        mp_frame.pack(fill="x", pady=(0, 12))
        ctk.CTkSlider(mp_frame, from_=1.0, to=20.0, number_of_steps=38,
                      variable=self.mp_var,
                      command=lambda v: self._update("min_megapixels", round(float(v), 1))
                      ).pack(side="left", fill="x", expand=True, padx=(0, 10))
        ctk.CTkLabel(mp_frame, textvariable=self.mp_var, width=40,
                     font=ctk.CTkFont(size=11)).pack(side="right")

        # Max file size
        ctk.CTkLabel(tab, text="Taille max fichier (MB)", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", pady=(5, 2))
        ctk.CTkLabel(tab, text="Fichiers dépassant cette taille seront exclus",
                     font=ctk.CTkFont(size=9), text_color=COLORS["text_muted"]).pack(anchor="w", pady=(0, 5))

        self.size_var = ctk.DoubleVar(value=self.params.max_file_size_mb)
        sz_frame = ctk.CTkFrame(tab, fg_color="transparent")
        sz_frame.pack(fill="x", pady=(0, 12))
        ctk.CTkSlider(sz_frame, from_=5.0, to=200.0, number_of_steps=39,
                      variable=self.size_var,
                      command=lambda v: self._update("max_file_size_mb", round(float(v), 1))
                      ).pack(side="left", fill="x", expand=True, padx=(0, 10))
        ctk.CTkLabel(sz_frame, textvariable=self.size_var, width=45,
                     font=ctk.CTkFont(size=11)).pack(side="right")

        # Fix orientation
        self.orient_var = ctk.BooleanVar(value=self.params.fix_orientation)
        ctk.CTkCheckBox(tab, text="Corriger l'orientation EXIF automatiquement",
                        variable=self.orient_var,
                        command=lambda: self._update("fix_orientation", self.orient_var.get())
                        ).pack(anchor="w", pady=(10, 0))

    def _build_ftps_tab(self):
        tab = self.tabs.add("FTPS")

        ctk.CTkLabel(tab, text="Configuration FTPS", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", pady=(10, 2))
        ctk.CTkLabel(tab, text="Identifiants de connexion au serveur Shutterstock",
                     font=ctk.CTkFont(size=9), text_color=COLORS["text_muted"]).pack(anchor="w", pady=(0, 15))

        ctk.CTkLabel(tab, text="Nom d'utilisateur:", font=ctk.CTkFont(size=11)).pack(anchor="w", pady=(0, 2))
        self.ftps_user_entry = ctk.CTkEntry(tab, height=32, placeholder_text="Identifiant Shutterstock")
        self.ftps_user_entry.pack(fill="x", pady=(0, 10))
        if self.params.ftps_username:
            self.ftps_user_entry.insert(0, self.params.ftps_username)

        ctk.CTkLabel(tab, text="Mot de passe:", font=ctk.CTkFont(size=11)).pack(anchor="w", pady=(0, 2))
        self.ftps_pass_entry = ctk.CTkEntry(tab, show="•", height=32, placeholder_text="Mot de passe FTPS")
        self.ftps_pass_entry.pack(fill="x", pady=(0, 15))
        if self.params.ftps_password:
            self.ftps_pass_entry.insert(0, self.params.ftps_password)

        # Info serveur
        info = ctk.CTkFrame(tab, fg_color=("#F1F5F9", "#1E293B"), corner_radius=8)
        info.pack(fill="x")
        ctk.CTkLabel(info, text="Serveur: ftps.shutterstock.com\nPort: 21 | Protocole: FTPS (TLS implicite)",
                     font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"], justify="left").pack(padx=12, pady=8)

        # Bind changes
        self.ftps_user_entry.bind("<FocusOut>", lambda e: self._update("ftps_username", self.ftps_user_entry.get()))
        self.ftps_pass_entry.bind("<FocusOut>", lambda e: self._update("ftps_password", self.ftps_pass_entry.get()))

    def _build_debug_tab(self):
        tab = self.tabs.add("Debug")

        ctk.CTkLabel(tab, text="Mode debug", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", pady=(10, 2))
        ctk.CTkLabel(tab, text="Active les logs détaillés et sorties de diagnostic",
                     font=ctk.CTkFont(size=9), text_color=COLORS["text_muted"]).pack(anchor="w", pady=(0, 10))

        self.debug_var = ctk.BooleanVar(value=self.params.debug_mode)
        ctk.CTkCheckBox(tab, text="Activer le mode debug",
                        variable=self.debug_var,
                        command=lambda: self._update("debug_mode", self.debug_var.get())
                        ).pack(anchor="w", pady=(5, 0))

        ctk.CTkLabel(tab, text="Le mode debug génère des fichiers de log supplémentaires\n"
                               "dans le dossier source et affiche les requêtes/réponses Ollama.",
                     font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"],
                     justify="left").pack(anchor="w", pady=(15, 0))

    def _update(self, field: str, value):
        setattr(self.params, field, value)
        self._on_change(field, value)

    def _reset_all(self):
        defaults = ShutterstockParams()
        # Performance
        self.gpu_var.set(defaults.gpu_layers)
        self.cooldown_var.set(defaults.cooldown)
        self.workers_var.set(defaults.workers)
        # Prefilter
        self.mp_var.set(defaults.min_megapixels)
        self.size_var.set(defaults.max_file_size_mb)
        self.orient_var.set(defaults.fix_orientation)
        # FTPS
        self.ftps_user_entry.delete(0, "end")
        self.ftps_pass_entry.delete(0, "end")
        # Debug
        self.debug_var.set(defaults.debug_mode)

        for field in ["gpu_layers", "cooldown", "workers", "min_megapixels",
                      "max_file_size_mb", "fix_orientation", "ftps_username",
                      "ftps_password", "debug_mode"]:
            self._update(field, getattr(defaults, field))
