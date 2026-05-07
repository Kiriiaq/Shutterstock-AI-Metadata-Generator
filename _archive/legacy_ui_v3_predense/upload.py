"""Upload view — FTPS upload to Shutterstock (placeholder, not implemented)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import customtkinter as ctk

from app.components.empty_state import EmptyState
from app.views.base_view import BaseView

if TYPE_CHECKING:
    from app.app import App


class UploadView(BaseView):
    view_id = "upload"

    def __init__(self, master: ctk.CTkFrame, *, app: "App") -> None:
        super().__init__(master)
        self.app = app
        EmptyState(
            self,
            icon="↑",
            title="Téléversement FTPS — non implémenté",
            subtitle=(
                "Le téléversement direct vers Shutterstock arrivera dans une prochaine itération. "
                "Vous pouvez utiliser un client FTPS externe en attendant ; les fichiers à envoyer "
                "se trouvent dans le dossier validé via l'étape précédente."
            ),
            action_label="Aller à Validation",
            on_action=lambda: self.app.router.navigate_to("validate"),
        ).grid(row=0, column=0, sticky="nsew")
