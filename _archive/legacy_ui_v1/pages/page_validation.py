"""
Page Validation — Checklist de vérification avant upload.
"""

import customtkinter as ctk
from pathlib import Path
from typing import Callable, List, Tuple
from ...core.params import ShutterstockParams


COLORS = {"text_muted": ("#64748B", "#94A3B8"), "border": ("#E2E8F0", "#334155"),
           "success": ("#10B981", "#34D399"), "warning": ("#F59E0B", "#FBBF24"),
           "error": ("#EF4444", "#F87171")}


class PageValidation(ctk.CTkScrollableFrame):
    """Page de validation pré-upload."""

    PAGE_KEY = "validation"

    def __init__(self, parent, params: ShutterstockParams, on_change: Callable, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.params = params
        self._on_change = on_change
        self._check_labels: List[Tuple[ctk.CTkLabel, ctk.CTkLabel]] = []
        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(self, text="Validation", font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(self, text="Vérifiez que tout est prêt avant l'upload",
                     font=ctk.CTkFont(size=11), text_color=COLORS["text_muted"]).pack(anchor="w", pady=(0, 20))

        # Checklist card
        check_card = ctk.CTkFrame(self, corner_radius=12, border_width=1, border_color=COLORS["border"])
        check_card.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(check_card, text="✅  Checklist",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=16, pady=(16, 12))

        self._checks_frame = ctk.CTkFrame(check_card, fg_color="transparent")
        self._checks_frame.pack(fill="x", padx=16, pady=(0, 16))

        checks = [
            ("source", "Dossier source sélectionné"),
            ("images", "Images trouvées dans le dossier"),
            ("model", "Modèle IA configuré"),
            ("analysis", "Analyse terminée"),
            ("valid_images", "Images valides disponibles"),
            ("ftps_creds", "Identifiants FTPS renseignés"),
        ]

        for key, label in checks:
            row = ctk.CTkFrame(self._checks_frame, fg_color="transparent")
            row.pack(fill="x", pady=3)

            icon = ctk.CTkLabel(row, text="○", width=20,
                                font=ctk.CTkFont(size=12), text_color=COLORS["text_muted"])
            icon.pack(side="left")

            text = ctk.CTkLabel(row, text=label, font=ctk.CTkFont(size=11))
            text.pack(side="left", padx=(6, 0))

            self._check_labels.append((icon, text))

        # Résumé
        summary_card = ctk.CTkFrame(self, corner_radius=12, border_width=1, border_color=COLORS["border"])
        summary_card.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(summary_card, text="📊  Résumé",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=16, pady=(16, 8))

        self.summary_label = ctk.CTkLabel(summary_card, text="Exécutez la validation pour voir le résumé",
                                           font=ctk.CTkFont(size=11), text_color=COLORS["text_muted"],
                                           wraplength=350, justify="left")
        self.summary_label.pack(anchor="w", padx=16, pady=(0, 16))

        # Bouton valider
        self.validate_btn = ctk.CTkButton(
            self, text="🔄  Actualiser la validation",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=42, corner_radius=10,
            command=self.run_validation
        )
        self.validate_btn.pack(fill="x", pady=(0, 10))

    def run_validation(self):
        """Exécute les vérifications."""
        results = self._check_params()
        passed = sum(1 for r in results if r)
        total = len(results)

        for i, ok in enumerate(results):
            icon, text = self._check_labels[i]
            if ok:
                icon.configure(text="✔", text_color=COLORS["success"])
            else:
                icon.configure(text="✘", text_color=COLORS["error"])

        if passed == total:
            self.summary_label.configure(
                text=f"Tout est prêt ! {passed}/{total} vérifications réussies.",
                text_color=COLORS["success"]
            )
        else:
            self.summary_label.configure(
                text=f"{passed}/{total} vérifications réussies. Corrigez les éléments manquants.",
                text_color=COLORS["warning"]
            )

    def _check_params(self) -> List[bool]:
        """Vérifie chaque point de la checklist."""
        results = []
        folder = Path(self.params.source_folder) if self.params.source_folder else None

        # 1. Dossier source sélectionné
        results.append(bool(self.params.source_folder))

        # 2. Images trouvées
        if folder and folder.exists():
            exts = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
            images = [f for f in folder.iterdir() if f.suffix.lower() in exts]
            results.append(len(images) > 0)
        else:
            results.append(False)

        # 3. Modèle configuré
        results.append(bool(self.params.model_name))

        # 4. Analyse terminée (vérifie le dossier Shutterstock)
        if folder and folder.exists():
            ss_folder = folder / "Shutterstock"
            results.append(ss_folder.exists() and any(ss_folder.iterdir()) if ss_folder.exists() else False)
        else:
            results.append(False)

        # 5. Images valides disponibles
        if folder and folder.exists():
            valid_folder = folder / "Valid"
            results.append(valid_folder.exists() and any(valid_folder.iterdir()) if valid_folder.exists() else False)
        else:
            results.append(False)

        # 6. FTPS credentials
        results.append(bool(self.params.ftps_username and self.params.ftps_password))

        return results

    def refresh(self, params: ShutterstockParams):
        self.params = params
