"""Editor view — IPTC/XMP metadata editor for one image at a time."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import customtkinter as ctk

from app.components.empty_state import EmptyState
from app.components.form_field import FormField, entry_factory, textbox_factory
from app.config.theme import (
    RADIUS_MD,
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    SPACE_XL,
    get_color,
    get_font,
)
from app.views.base_view import BaseView

if TYPE_CHECKING:
    from app.app import App

logger = logging.getLogger(__name__)


class EditorView(BaseView):
    view_id = "editor"

    def __init__(self, master: ctk.CTkFrame, *, app: "App") -> None:
        super().__init__(master)
        self.app = app
        self._current_path: Path | None = None
        self._fields: dict[str, FormField] = {}
        self._build()

    # ------------------------------------------------------------------

    def _build(self) -> None:
        api = self.app.api
        if api is None or api.metadata_reader is None:
            EmptyState(
                self,
                icon="✎",
                title="ExifTool indisponible",
                subtitle="Installez ExifTool puis renseignez son chemin dans Paramètres.",
                action_label="Ouvrir Paramètres",
                on_action=lambda: self.app.router.navigate_to("settings"),
            ).grid(row=0, column=0, sticky="nsew")
            return

        scanned = self.app.app_state.get("scanned_images") or []
        if not scanned:
            EmptyState(
                self,
                icon="✎",
                title="Aucune image disponible",
                subtitle="Scannez d'abord un dossier dans Sources et tri.",
                action_label="Aller à Sources",
                on_action=lambda: self.app.router.navigate_to("sources"),
            ).grid(row=0, column=0, sticky="nsew")
            return

        wrapper = ctk.CTkFrame(self, fg_color="transparent")
        wrapper.grid(row=0, column=0, sticky="nsew", padx=SPACE_XL, pady=SPACE_XL)
        wrapper.grid_columnconfigure(1, weight=1)
        wrapper.grid_rowconfigure(0, weight=1)

        self._build_file_list(wrapper, scanned)
        self._build_editor(wrapper)

    def _build_file_list(self, parent: ctk.CTkFrame, files: list[Path]) -> None:
        side = ctk.CTkFrame(parent, fg_color=get_color("bg_elevated"), corner_radius=RADIUS_MD, width=260)
        side.grid(row=0, column=0, sticky="ns", padx=(0, SPACE_LG))
        side.grid_propagate(False)
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            side, text=f"Fichiers ({len(files)})", font=get_font("body_strong"), text_color=get_color("fg")
        ).grid(row=0, column=0, sticky="w", padx=SPACE_MD, pady=SPACE_MD)

        list_frame = ctk.CTkScrollableFrame(side, fg_color="transparent")
        list_frame.grid(row=1, column=0, sticky="nsew", padx=SPACE_SM, pady=(0, SPACE_MD))
        for path in files:
            ctk.CTkButton(
                list_frame,
                text=path.name,
                anchor="w",
                fg_color="transparent",
                hover_color=get_color("bg_hover"),
                text_color=get_color("fg"),
                font=get_font("body"),
                height=28,
                command=lambda p=path: self._select_file(p),
            ).pack(fill="x", pady=1)

    def _build_editor(self, parent: ctk.CTkFrame) -> None:
        editor = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        editor.grid(row=0, column=1, sticky="nsew")
        editor.grid_columnconfigure(0, weight=1)

        self._title_label = ctk.CTkLabel(
            editor,
            text="Sélectionnez un fichier dans la liste",
            font=get_font("h2"),
            text_color=get_color("fg_muted"),
            anchor="w",
        )
        self._title_label.grid(row=0, column=0, sticky="ew", pady=(0, SPACE_MD))

        for row, (key, label, factory) in enumerate(
            [
                ("headline", "Titre (Headline)", entry_factory),
                ("caption", "Description", textbox_factory(80)),
                ("keywords", "Mots-clés (séparés par des virgules)", textbox_factory(60)),
                ("byline", "Auteur (Byline)", entry_factory),
                ("copyright_notice", "Copyright", entry_factory),
                ("city", "Ville", entry_factory),
                ("country_name", "Pays", entry_factory),
            ],
            start=1,
        ):
            field = FormField(editor, label=label, widget_factory=factory)
            field.grid(row=row, column=0, sticky="ew", pady=(0, SPACE_MD))
            self._fields[key] = field

        actions = ctk.CTkFrame(editor, fg_color="transparent")
        actions.grid(row=99, column=0, sticky="ew", pady=SPACE_MD)
        ctk.CTkButton(actions, text="Recharger", width=120, command=self._reload).pack(side="left", padx=(0, SPACE_SM))
        ctk.CTkButton(
            actions,
            text="Enregistrer",
            width=120,
            fg_color=get_color("accent"),
            hover_color=get_color("accent_hover"),
            text_color=get_color("accent_fg"),
            font=get_font("body_strong"),
            command=self._save,
        ).pack(side="left", padx=SPACE_SM)
        ctk.CTkButton(actions, text="Effacer", width=120, command=self._clear).pack(side="left", padx=SPACE_SM)

    # ------------------------------------------------------------------

    def _select_file(self, path: Path) -> None:
        self._current_path = path
        self._title_label.configure(text=path.name, text_color=get_color("fg"))
        self._reload()

    def _reload(self) -> None:
        if self._current_path is None:
            return
        api = self.app.api
        if api is None:
            return
        try:
            metadata = api.read_metadata(self._current_path)
        except Exception:
            logger.exception("read_metadata failed")
            self.app.toasts.show("Lecture impossible.", kind="error")
            return
        if metadata is None:
            self._clear()
            return
        iptc = metadata.iptc
        self._fields["headline"].set_value(iptc.headline or iptc.object_name or "")
        self._fields["caption"].set_value(iptc.caption or "")
        self._fields["keywords"].set_value(", ".join(iptc.keywords or []))
        self._fields["byline"].set_value(iptc.byline or "")
        self._fields["copyright_notice"].set_value(iptc.copyright_notice or "")
        self._fields["city"].set_value(iptc.city or "")
        self._fields["country_name"].set_value(iptc.country_name or "")

    def _save(self) -> None:
        if self._current_path is None:
            self.app.toasts.show("Aucun fichier sélectionné.", kind="warning")
            return
        api = self.app.api
        if api is None:
            return

        from src.modules.models.metadata_models import IPTCFields

        kw = [k.strip() for k in self._fields["keywords"].value.split(",") if k.strip()]
        iptc = IPTCFields(
            headline=self._fields["headline"].value or None,
            caption=self._fields["caption"].value or None,
            keywords=kw,
            byline=self._fields["byline"].value or None,
            copyright_notice=self._fields["copyright_notice"].value or None,
            city=self._fields["city"].value or None,
            country_name=self._fields["country_name"].value or None,
        )
        try:
            ok = api.write_metadata(self._current_path, iptc=iptc)
        except Exception:
            logger.exception("write_metadata failed")
            self.app.toasts.show("Écriture impossible.", kind="error")
            return
        if ok:
            self.app.toasts.show(f"Métadonnées écrites : {self._current_path.name}", kind="success")
            self._reload()
        else:
            self.app.toasts.show("Échec de l'écriture (voir le journal).", kind="error")

    def _clear(self) -> None:
        for field in self._fields.values():
            field.set_value("")
