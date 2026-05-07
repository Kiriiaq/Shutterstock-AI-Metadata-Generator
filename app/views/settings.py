"""Settings view — multi-section form backed by Database settings table."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import customtkinter as ctk

from app.components.form_field import FormField, combo_factory, entry_factory, switch_factory
from app.config.theme import (
    RADIUS_MD,
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    SPACE_XL,
    get_font,
    palette_pair,
)
from app.views.base_view import BaseView

if TYPE_CHECKING:
    from app.app import App

logger = logging.getLogger(__name__)


DEFAULTS: dict[str, Any] = {
    "ollama_url": "http://localhost:11434",
    "ollama_model": "llama3.2-vision:11b",
    "ollama_timeout": 120,
    "max_workers": 4,
    "batch_size": 50,
    "min_resolution_mp": 4.0,
    "default_byline": "",
    "default_copyright": "",
    "write_iptc": True,
    "write_xmp": True,
    "create_backup": True,
    "exiftool_path": "",
    "ftps_host": "ftps.shutterstock.com",
    "ftps_port": 21,
    "ftps_username": "",
    "debug_mode": False,
    "log_level": "INFO",
}


class SettingsView(BaseView):
    view_id = "settings"

    def __init__(self, master: ctk.CTkFrame, *, app: "App") -> None:
        super().__init__(master)
        self.app = app
        self._fields: dict[str, FormField] = {}
        self._current: dict[str, Any] = self._load_settings()
        self._build()

    # ------------------------------------------------------------------

    def _load_settings(self) -> dict[str, Any]:
        merged = dict(DEFAULTS)
        api = self.app.api
        if api is not None:
            try:
                merged.update(api.get_all_settings())
            except Exception:
                logger.exception("Could not load settings; using defaults")
        return merged

    def _build(self) -> None:
        wrapper = ctk.CTkScrollableFrame(self, fg_color="transparent")
        wrapper.grid(row=0, column=0, sticky="nsew", padx=SPACE_XL, pady=SPACE_XL)
        wrapper.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(wrapper, text="Paramètres", font=get_font("h1"), text_color=palette_pair("fg")).grid(
            row=0, column=0, sticky="w", pady=(0, SPACE_LG)
        )

        row = 1
        row = self._build_section(
            wrapper,
            row,
            "Ollama",
            [
                ("ollama_url", "URL du serveur", entry_factory),
                (
                    "ollama_model",
                    "Modèle vision",
                    combo_factory(
                        ["llama3.2-vision:11b", "llama3.2-vision:90b", "llava:7b", "llava:13b", "moondream:1.8b"]
                    ),
                ),
                ("ollama_timeout", "Timeout (s)", entry_factory),
            ],
        )
        row = self._build_section(
            wrapper,
            row,
            "Traitement",
            [
                ("max_workers", "Workers parallèles", entry_factory),
                ("batch_size", "Taille des lots", entry_factory),
                ("min_resolution_mp", "Résolution minimale (MP)", entry_factory),
            ],
        )
        row = self._build_section(
            wrapper,
            row,
            "Métadonnées par défaut",
            [
                ("default_byline", "Auteur (Byline)", entry_factory),
                ("default_copyright", "Copyright", entry_factory),
                ("exiftool_path", "Chemin ExifTool", entry_factory),
                ("write_iptc", "Écrire IPTC", switch_factory("Activer")),
                ("write_xmp", "Écrire XMP", switch_factory("Activer")),
                ("create_backup", "Créer une sauvegarde _original", switch_factory("Activer")),
            ],
        )
        row = self._build_section(
            wrapper,
            row,
            "FTPS",
            [
                ("ftps_host", "Hôte", entry_factory),
                ("ftps_port", "Port", entry_factory),
                ("ftps_username", "Identifiant", entry_factory),
            ],
        )
        row = self._build_section(
            wrapper,
            row,
            "Avancé",
            [
                ("debug_mode", "Mode debug", switch_factory("Activer")),
                ("log_level", "Niveau de journalisation", combo_factory(["DEBUG", "INFO", "WARNING", "ERROR"])),
            ],
        )

        actions = ctk.CTkFrame(wrapper, fg_color="transparent")
        actions.grid(row=row, column=0, sticky="ew", pady=SPACE_LG)
        ctk.CTkButton(
            actions,
            text="Enregistrer",
            fg_color=palette_pair("accent"),
            hover_color=palette_pair("accent_hover"),
            text_color=palette_pair("accent_fg"),
            font=get_font("body_strong"),
            command=self._save,
        ).pack(side="left", padx=(0, SPACE_SM))
        ctk.CTkButton(actions, text="Réinitialiser aux valeurs par défaut", command=self._reset).pack(
            side="left", padx=SPACE_SM
        )

    def _build_section(
        self,
        parent: ctk.CTkFrame,
        row: int,
        title: str,
        items: list[tuple[str, str, Any]],
    ) -> int:
        section = ctk.CTkFrame(
            parent,
            fg_color=palette_pair("bg_elevated"),
            border_color=palette_pair("border"),
            border_width=1,
            corner_radius=RADIUS_MD,
        )
        section.grid(row=row, column=0, sticky="ew", pady=(0, SPACE_LG))
        section.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(section, text=title, font=get_font("body_strong"), text_color=palette_pair("fg")).grid(
            row=0, column=0, sticky="w", padx=SPACE_LG, pady=(SPACE_MD, SPACE_SM)
        )

        body = ctk.CTkFrame(section, fg_color="transparent")
        body.grid(row=1, column=0, sticky="ew", padx=SPACE_LG, pady=(0, SPACE_MD))
        body.grid_columnconfigure(0, weight=1)

        for idx, (key, label, factory) in enumerate(items):
            field = FormField(body, label=label, widget_factory=factory)
            field.grid(row=idx, column=0, sticky="ew", pady=(0, SPACE_SM))
            field.set_value(self._current.get(key, DEFAULTS.get(key, "")))
            self._fields[key] = field

        return row + 1

    # ------------------------------------------------------------------

    def _save(self) -> None:
        api = self.app.api
        if api is None:
            self.app.toasts.show("Backend indisponible.", kind="error")
            return
        try:
            for key, field in self._fields.items():
                value = field.value
                if key in ("ollama_timeout", "max_workers", "batch_size", "ftps_port"):
                    value = int(value or 0)
                elif key == "min_resolution_mp":
                    value = float(value or 0)
                api.set_setting(key, value)
            self.app.toasts.show("Paramètres enregistrés.", kind="success")
        except Exception as e:
            logger.exception("Settings save failed")
            self.app.toasts.show(f"Échec : {e}", kind="error")

    def _reset(self) -> None:
        for key, field in self._fields.items():
            field.set_value(DEFAULTS.get(key, ""))
        self.app.toasts.show("Valeurs par défaut chargées (non enregistrées).", kind="info")
