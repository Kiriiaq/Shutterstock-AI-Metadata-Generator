"""
Page Modèle IA — Configuration et gestion d'Ollama.
"""

import customtkinter as ctk
import threading
from typing import Callable
from ...core.params import ShutterstockParams, PARAMS_META


COLORS = {"text_muted": ("#64748B", "#94A3B8"), "border": ("#E2E8F0", "#334155"),
           "success": ("#10B981", "#34D399"), "warning": ("#F59E0B", "#FBBF24"),
           "error": ("#EF4444", "#F87171"), "primary": ("#3B82F6", "#60A5FA")}


class PageModel(ctk.CTkScrollableFrame):
    """Page de configuration du modèle IA."""

    PAGE_KEY = "modele"

    def __init__(self, parent, params: ShutterstockParams, on_change: Callable,
                 ollama_manager=None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.params = params
        self._on_change = on_change
        self.ollama_manager = ollama_manager

        self.model_var = ctk.StringVar(value=params.model_name)

        self._build_ui()
        self.after(1000, self._refresh_status)

    def _build_ui(self):
        ctk.CTkLabel(self, text="Modèle IA", font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(self, text="Configurez le modèle Ollama Vision pour l'analyse d'images",
                     font=ctk.CTkFont(size=11), text_color=COLORS["text_muted"]).pack(anchor="w", pady=(0, 20))

        # Statut Ollama
        status_card = ctk.CTkFrame(self, corner_radius=12, border_width=1, border_color=COLORS["border"])
        status_card.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(status_card, text="🤖  Statut Ollama",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=16, pady=(16, 8))

        self.status_label = ctk.CTkLabel(status_card, text="Vérification...",
                                         font=ctk.CTkFont(size=12), text_color=COLORS["warning"])
        self.status_label.pack(anchor="w", padx=16, pady=(0, 8))

        # Boutons de contrôle
        ctrl_frame = ctk.CTkFrame(status_card, fg_color="transparent")
        ctrl_frame.pack(fill="x", padx=16, pady=(0, 16))

        self.start_btn = ctk.CTkButton(ctrl_frame, text="Démarrer", width=90, height=30,
                                       fg_color=("#10B981", "#059669"), command=self._start_ollama)
        self.start_btn.pack(side="left", padx=(0, 5))

        self.stop_btn = ctk.CTkButton(ctrl_frame, text="Arrêter", width=80, height=30,
                                      fg_color=("#EF4444", "#DC2626"), command=self._stop_ollama)
        self.stop_btn.pack(side="left", padx=(0, 5))

        ctk.CTkButton(ctrl_frame, text="↻ Actualiser", width=90, height=30,
                      fg_color="transparent", border_width=1,
                      command=self._refresh_status).pack(side="left")

        # Sélection modèle
        model_card = ctk.CTkFrame(self, corner_radius=12, border_width=1, border_color=COLORS["border"])
        model_card.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(model_card, text="📋  Modèle actif",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=16, pady=(16, 8))

        model_frame = ctk.CTkFrame(model_card, fg_color="transparent")
        model_frame.pack(fill="x", padx=16, pady=(0, 8))

        self.model_combo = ctk.CTkComboBox(
            model_frame, variable=self.model_var,
            values=PARAMS_META["model_name"].choices,
            width=220, height=32,
            command=lambda v: self._on_change("model_name", v)
        )
        self.model_combo.pack(side="left", padx=(0, 10))

        ctk.CTkButton(model_frame, text="Charger", width=80, height=32,
                      fg_color=COLORS["primary"], command=self._load_model).pack(side="left")

        self.loaded_label = ctk.CTkLabel(model_card, text="Aucun modèle chargé",
                                         font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"])
        self.loaded_label.pack(anchor="w", padx=16, pady=(0, 8))

        # GPU info
        self.gpu_label = ctk.CTkLabel(model_card, text="",
                                      font=ctk.CTkFont(size=10), text_color=COLORS["primary"])
        self.gpu_label.pack(anchor="w", padx=16, pady=(0, 16))

        # Téléchargement
        dl_card = ctk.CTkFrame(self, corner_radius=12, border_width=1, border_color=COLORS["border"])
        dl_card.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(dl_card, text="⬇  Télécharger un modèle",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=16, pady=(16, 8))

        dl_frame = ctk.CTkFrame(dl_card, fg_color="transparent")
        dl_frame.pack(fill="x", padx=16, pady=(0, 16))

        self.dl_combo = ctk.CTkComboBox(dl_frame, values=PARAMS_META["model_name"].choices,
                                        width=200, height=30)
        self.dl_combo.set("llama3.2-vision:11b")
        self.dl_combo.pack(side="left", padx=(0, 10))

        ctk.CTkButton(dl_frame, text="Télécharger", width=100, height=30,
                      fg_color=("#F59E0B", "#D97706"), command=self._download_model).pack(side="left")

    def _refresh_status(self):
        """Actualise le statut d'Ollama."""
        if self.ollama_manager:
            try:
                status = self.ollama_manager.check_status()
                if status.value == "prêt":
                    self.status_label.configure(text="● Ollama en cours d'exécution", text_color=COLORS["success"])
                elif status.value == "non_démarré":
                    self.status_label.configure(text="○ Ollama arrêté", text_color=COLORS["error"])
                else:
                    self.status_label.configure(text=f"⚠ {status.value}", text_color=COLORS["warning"])

                gpu_text = self.ollama_manager.get_gpu_status_string()
                self.gpu_label.configure(text=gpu_text)
            except Exception:
                self.status_label.configure(text="⚠ Impossible de vérifier Ollama", text_color=COLORS["warning"])
        else:
            self.status_label.configure(text="ⓘ Gestionnaire Ollama non disponible", text_color=COLORS["text_muted"])

    def _start_ollama(self):
        if self.ollama_manager:
            threading.Thread(target=self.ollama_manager.start_server, daemon=True).start()
            self.after(3000, self._refresh_status)

    def _stop_ollama(self):
        if self.ollama_manager:
            threading.Thread(target=self.ollama_manager.stop_server, daemon=True).start()
            self.after(2000, self._refresh_status)

    def _load_model(self):
        if self.ollama_manager:
            model = self.model_var.get()
            self.loaded_label.configure(text=f"Chargement de {model}...")
            threading.Thread(target=lambda: self._do_load(model), daemon=True).start()

    def _do_load(self, model):
        try:
            if self.ollama_manager:
                self.ollama_manager.load_model(model)
            self.after(0, lambda: self.loaded_label.configure(
                text=f"✔ {model} chargé", text_color=COLORS["success"]))
        except Exception as e:
            self.after(0, lambda: self.loaded_label.configure(
                text=f"✖ Erreur: {e}", text_color=COLORS["error"]))

    def _download_model(self):
        model = self.dl_combo.get()
        if self.ollama_manager:
            threading.Thread(target=lambda: self.ollama_manager.pull_model(model), daemon=True).start()

    def refresh(self, params: ShutterstockParams):
        self.params = params
        self.model_var.set(params.model_name)
