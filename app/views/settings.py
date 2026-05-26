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
    SPACE_XS,
    get_font,
    palette_pair,
)
from app.views.base_view import BaseView, _modal_header

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

        # Header: icon + h1 title, anchored top-left.
        _modal_header(wrapper, icon="⚙", title="Paramètres", row=0)

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
        row = self._build_license_section(wrapper, row)

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
                if key in ("ollama_timeout", "max_workers", "batch_size"):
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

    # ------------------------------------------------------------------
    # Licence section — Community status + paste-key + activate/deactivate
    # ------------------------------------------------------------------

    def _build_license_section(self, parent: ctk.CTkFrame, row: int) -> int:
        """Section dédiée à la licence Pro.

        Contient :
        - Affichage du tier courant + email + date d'expiration
        - Textbox pour coller la clé JSON
        - Bouton Activer (parse + verify + write file + reload)
        - Bouton Retirer (suppression du license.json local)
        - Lien Gumroad d'achat
        """
        api = self.app.api
        section = ctk.CTkFrame(
            parent,
            fg_color=palette_pair("bg_elevated"),
            border_color=palette_pair("border"),
            border_width=1,
            corner_radius=RADIUS_MD,
        )
        section.grid(row=row, column=0, sticky="ew", pady=(0, SPACE_LG))
        section.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            section, text="Licence", font=get_font("body_strong"),
            text_color=palette_pair("fg"),
        ).grid(row=0, column=0, sticky="w", padx=SPACE_LG, pady=(SPACE_MD, SPACE_SM))

        body = ctk.CTkFrame(section, fg_color="transparent")
        body.grid(row=1, column=0, sticky="ew", padx=SPACE_LG, pady=(0, SPACE_MD))
        body.grid_columnconfigure(0, weight=1)

        # Statut courant
        self._license_status = ctk.CTkLabel(
            body, text="", font=get_font("body"),
            text_color=palette_pair("fg"), anchor="w", justify="left",
        )
        self._license_status.grid(row=0, column=0, sticky="ew", pady=(0, SPACE_SM))
        self._refresh_license_status()

        # Zone collage clé
        ctk.CTkLabel(
            body, text="Coller votre clé de licence (JSON) :",
            font=get_font("small"), text_color=palette_pair("fg_muted"),
            anchor="w",
        ).grid(row=1, column=0, sticky="w")

        self._license_textbox = ctk.CTkTextbox(
            body, font=get_font("code"), height=120,
            fg_color=palette_pair("bg"), text_color=palette_pair("fg"),
            border_color=palette_pair("border"), border_width=1,
            corner_radius=4,
        )
        self._license_textbox.grid(row=2, column=0, sticky="ew", pady=(SPACE_SM, SPACE_SM))

        # Actions
        actions = ctk.CTkFrame(body, fg_color="transparent")
        actions.grid(row=3, column=0, sticky="ew")
        actions.grid_columnconfigure(99, weight=1)

        ctk.CTkButton(
            actions, text="Activer", width=110, height=28,
            fg_color=palette_pair("accent"),
            hover_color=palette_pair("accent_hover"),
            text_color=palette_pair("accent_fg"),
            font=get_font("body_strong"),
            command=self._activate_license,
        ).grid(row=0, column=0, padx=(0, SPACE_SM))

        ctk.CTkButton(
            actions, text="Retirer la licence", width=160, height=28,
            fg_color=palette_pair("bg_hover"),
            hover_color=palette_pair("bg_active"),
            text_color=palette_pair("fg"),
            border_width=1, border_color=palette_pair("border"),
            command=self._deactivate_license,
            state="normal" if (api and api.license.is_pro()) else "disabled",
        ).grid(row=0, column=1, padx=SPACE_SM)
        self._deactivate_btn = actions.winfo_children()[-1]  # ref pour refresh

        # Lien d'achat (Gumroad — à brancher quand le listing est créé)
        ctk.CTkLabel(
            actions, text="Pas encore de licence ?",
            font=get_font("small"), text_color=palette_pair("fg_muted"),
        ).grid(row=0, column=10, padx=(SPACE_LG, SPACE_XS))
        ctk.CTkButton(
            actions, text="Acheter Pro →", width=120, height=28,
            fg_color="transparent",
            hover_color=palette_pair("bg_hover"),
            text_color=palette_pair("accent"),
            border_width=1, border_color=palette_pair("accent"),
            command=self._open_purchase_link,
        ).grid(row=0, column=11)

        return row + 1

    def _refresh_license_status(self) -> None:
        """Met à jour le label de statut + état du bouton Retirer."""
        api = self.app.api
        if api is None:
            self._license_status.configure(text="Backend indisponible.")
            return
        lic = api.license
        if lic.is_pro():
            tier_label = {
                "pro_solo": "Pro Solo",
                "pro_studio": "Pro Studio",
                "lifetime": "Lifetime",
            }.get(lic.tier.value, lic.tier.value)
            email = lic.email or "—"
            exp = lic.expires_at.strftime("%d/%m/%Y") if lic.expires_at else "jamais"
            text = (
                f"✅  Édition {tier_label}\n"
                f"     Email : {email}\n"
                f"     Expire : {exp}"
            )
            self._license_status.configure(text=text, text_color=palette_pair("success"))
        else:
            text = (
                "🆓  Édition Community (gratuite)\n"
                "     Scan + IPTC + export CSV mono-plateforme + FTP · 2 aperçus gratuits du rapport expert\n"
                "     Pro débloque : rapport expert illimité, dual CSV Adobe+Shutterstock, IA Ollama,\n"
                "     anti-stuffing automatique, batch > 50 images"
            )
            self._license_status.configure(text=text, text_color=palette_pair("fg"))

    def _activate_license(self) -> None:
        api = self.app.api
        if api is None:
            self.app.toasts.show("Backend indisponible.", kind="error")
            return
        text = self._license_textbox.get("1.0", "end").strip()
        if not text:
            self.app.toasts.show("Collez d'abord votre clé.", kind="warning")
            return
        ok, msg = api.activate_license(text)
        if ok:
            self.app.toasts.show(msg, kind="success")
            self._license_textbox.delete("1.0", "end")
            self._refresh_license_status()
            try:
                self._deactivate_btn.configure(state="normal")
            except Exception:
                pass
        else:
            self.app.toasts.show(f"Activation échouée : {msg}", kind="error")

    def _deactivate_license(self) -> None:
        api = self.app.api
        if api is None:
            return
        if not self.app.confirm_destructive(
            title="Retirer la licence",
            message=(
                "Repasser en édition Community ? La clé n'est PAS supprimée "
                "côté serveur — tu pourras la réactiver plus tard en la "
                "recollant dans ce champ."
            ),
        ):
            return
        ok, msg = api.deactivate_license()
        if ok:
            self.app.toasts.show(msg, kind="info")
            self._refresh_license_status()
            try:
                self._deactivate_btn.configure(state="disabled")
            except Exception:
                pass
        else:
            self.app.toasts.show(f"Échec : {msg}", kind="error")

    def _open_purchase_link(self) -> None:
        """Ouvre le listing Gumroad/Lemon Squeezy dans le navigateur."""
        import webbrowser
        url = "https://gumroad.com/l/shutterstockanalyzer-pro"  # TODO: remplacer par l'URL réelle
        try:
            webbrowser.open(url)
        except Exception:
            self.app.toasts.show(f"Lien : {url}", kind="info")
