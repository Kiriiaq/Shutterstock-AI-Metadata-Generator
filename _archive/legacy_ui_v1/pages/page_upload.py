"""
Page Upload FTPS — Envoi vers Shutterstock.
"""

import customtkinter as ctk
from typing import Callable
from ...core.params import ShutterstockParams


COLORS = {"text_muted": ("#64748B", "#94A3B8"), "border": ("#E2E8F0", "#334155"),
           "success": ("#10B981", "#34D399"), "error": ("#EF4444", "#F87171")}


class PageUpload(ctk.CTkScrollableFrame):
    """Page d'upload FTPS."""

    PAGE_KEY = "upload"

    def __init__(self, parent, params: ShutterstockParams, on_change: Callable, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.params = params
        self._on_change = on_change

        self.username_var = ctk.StringVar(value=params.ftps_username)
        self.password_var = ctk.StringVar(value=params.ftps_password)

        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(self, text="Upload FTPS", font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(self, text="Envoyez vos lots vers Shutterstock via FTPS sécurisé",
                     font=ctk.CTkFont(size=11), text_color=COLORS["text_muted"]).pack(anchor="w", pady=(0, 20))

        # Identifiants
        auth_card = ctk.CTkFrame(self, corner_radius=12, border_width=1, border_color=COLORS["border"])
        auth_card.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(auth_card, text="🔐  Identifiants FTPS",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=16, pady=(16, 8))

        auth_content = ctk.CTkFrame(auth_card, fg_color="transparent")
        auth_content.pack(fill="x", padx=16, pady=(0, 16))

        ctk.CTkLabel(auth_content, text="Nom d'utilisateur:", font=ctk.CTkFont(size=11)).pack(anchor="w")
        ctk.CTkEntry(auth_content, textvariable=self.username_var, height=32,
                     placeholder_text="Votre identifiant Shutterstock").pack(fill="x", pady=(2, 8))

        ctk.CTkLabel(auth_content, text="Mot de passe:", font=ctk.CTkFont(size=11)).pack(anchor="w")
        ctk.CTkEntry(auth_content, textvariable=self.password_var, show="•", height=32,
                     placeholder_text="Mot de passe FTPS").pack(fill="x", pady=(2, 0))

        # Info
        info = ctk.CTkFrame(auth_card, fg_color=("#F1F5F9", "#1E293B"), corner_radius=8)
        info.pack(fill="x", padx=16, pady=(8, 16))
        ctk.CTkLabel(info, text="ⓘ Serveur: ftps.shutterstock.com:21 (TLS implicite)",
                     font=ctk.CTkFont(size=10), text_color=COLORS["text_muted"]).pack(padx=10, pady=6)

        # Sélection du lot
        lot_card = ctk.CTkFrame(self, corner_radius=12, border_width=1, border_color=COLORS["border"])
        lot_card.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(lot_card, text="📦  Lot à envoyer",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=16, pady=(16, 8))

        self.lot_label = ctk.CTkLabel(lot_card, text="Aucun lot prêt à l'envoi",
                                      font=ctk.CTkFont(size=11), text_color=COLORS["text_muted"])
        self.lot_label.pack(anchor="w", padx=16, pady=(0, 16))

        # Bouton upload
        self.upload_btn = ctk.CTkButton(
            self, text="📤  Envoyer vers Shutterstock",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=45, corner_radius=10
        )
        self.upload_btn.pack(fill="x", pady=(0, 15))

        # Progression
        self.upload_progress = ctk.CTkProgressBar(self, height=8, corner_radius=4)
        self.upload_progress.pack(fill="x", pady=(0, 5))
        self.upload_progress.set(0)

        self.upload_status = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=10),
                                          text_color=COLORS["text_muted"])
        self.upload_status.pack(anchor="w")

    def refresh(self, params: ShutterstockParams):
        self.params = params
        self.username_var.set(params.ftps_username)
        self.password_var.set(params.ftps_password)
