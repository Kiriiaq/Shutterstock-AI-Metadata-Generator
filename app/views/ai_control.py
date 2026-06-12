"""AI Control view — Ollama server management and model selection."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

import customtkinter as ctk

from app.config.theme import (
    RADIUS_MD,
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    SPACE_XL,
    get_font,
    palette_pair,
)
from app.views.base_view import BaseView, _modal_header

if TYPE_CHECKING:
    from app.app import App

logger = logging.getLogger(__name__)


class AIControlView(BaseView):
    view_id = "ai_control"

    DEFAULT_URL = "http://localhost:11434"

    def __init__(self, master: ctk.CTkFrame, *, app: "App") -> None:
        super().__init__(master)
        self.app = app
        self._build()

    # ------------------------------------------------------------------

    def _build(self) -> None:
        wrapper = ctk.CTkScrollableFrame(self, fg_color="transparent")
        wrapper.grid(row=0, column=0, sticky="nsew", padx=SPACE_XL, pady=SPACE_XL)
        wrapper.grid_columnconfigure(0, weight=1)

        # Header: icon + h1 title, anchored top-left.
        _modal_header(wrapper, icon="🤖", title="Modèle IA", row=0)

        self._build_connection(wrapper, row=1)
        self._build_model(wrapper, row=2)
        self._build_test(wrapper, row=3)

    def _build_connection(self, parent: ctk.CTkFrame, row: int) -> None:
        section = self._section(parent, row, "Connexion au serveur Ollama")
        ctk.CTkLabel(section, text="URL :", font=get_font("body"), text_color=palette_pair("fg")).grid(
            row=0, column=0, padx=SPACE_LG, pady=SPACE_MD, sticky="w"
        )
        self._url = ctk.CTkEntry(section, width=320, font=get_font("body"))
        api = self.app.api
        default = (api.get_setting("ollama_url", self.DEFAULT_URL) if api else self.DEFAULT_URL) or self.DEFAULT_URL
        self._url.insert(0, default)
        self._url.grid(row=0, column=1, padx=(0, SPACE_LG), pady=SPACE_MD, sticky="ew")

        ctk.CTkButton(section, text="Tester la connexion", command=self._check).grid(
            row=0, column=2, padx=SPACE_LG, pady=SPACE_MD
        )

        self._status_label = ctk.CTkLabel(
            section, text="Statut inconnu", font=get_font("body_strong"), text_color=palette_pair("fg_muted")
        )
        self._status_label.grid(row=1, column=0, columnspan=3, sticky="w", padx=SPACE_LG, pady=(0, SPACE_MD))

    def _build_model(self, parent: ctk.CTkFrame, row: int) -> None:
        section = self._section(parent, row, "Modèle vision")
        ctk.CTkLabel(section, text="Modèle :", font=get_font("body"), text_color=palette_pair("fg")).grid(
            row=0, column=0, padx=SPACE_LG, pady=SPACE_MD, sticky="w"
        )
        # ``command`` persiste la sélection — avant l'audit (B-09) le
        # combo était purement décoratif : choisir un modèle n'avait
        # aucun effet (ni persistance, ni préchargement).
        self._model_combo = ctk.CTkComboBox(
            section, values=["—"], width=320, font=get_font("body"), command=self._on_model_selected
        )
        self._model_combo.set("—")
        self._model_combo.grid(row=0, column=1, sticky="ew", padx=(0, SPACE_LG), pady=SPACE_MD)
        ctk.CTkButton(section, text="Rafraîchir", command=self._refresh_models).grid(
            row=0, column=2, padx=(SPACE_LG, SPACE_SM), pady=SPACE_MD
        )
        ctk.CTkButton(
            section,
            text="⬇ Charger",
            fg_color=palette_pair("accent"),
            hover_color=palette_pair("accent_hover"),
            text_color=palette_pair("accent_fg"),
            command=self._load_model,
        ).grid(row=0, column=3, padx=(0, SPACE_LG), pady=SPACE_MD)

        self._model_info = ctk.CTkLabel(section, text="—", font=get_font("small"), text_color=palette_pair("fg_muted"))
        self._model_info.grid(row=1, column=0, columnspan=3, sticky="w", padx=SPACE_LG, pady=(0, SPACE_MD))

    def _build_test(self, parent: ctk.CTkFrame, row: int) -> None:
        section = self._section(parent, row, "Test de réponse")
        ctk.CTkButton(
            section,
            text="Lancer un test",
            fg_color=palette_pair("accent"),
            hover_color=palette_pair("accent_hover"),
            text_color=palette_pair("accent_fg"),
            font=get_font("body_strong"),
            command=self._test,
        ).grid(row=0, column=0, padx=SPACE_LG, pady=SPACE_MD, sticky="w")

        self._test_result = ctk.CTkTextbox(
            section,
            height=120,
            font=get_font("code"),
            fg_color=palette_pair("bg"),
            text_color=palette_pair("fg"),
            border_color=palette_pair("border"),
            border_width=1,
        )
        self._test_result.grid(row=1, column=0, columnspan=3, sticky="ew", padx=SPACE_LG, pady=(0, SPACE_MD))
        self._test_result.insert("1.0", "Cliquez sur Lancer un test pour vérifier le serveur Ollama.")
        self._test_result.configure(state="disabled")

    def _section(self, parent: ctk.CTkFrame, row: int, title: str) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(
            parent,
            fg_color=palette_pair("bg_elevated"),
            border_color=palette_pair("border"),
            border_width=1,
            corner_radius=RADIUS_MD,
        )
        frame.grid(row=row, column=0, sticky="ew", pady=(0, SPACE_LG))
        frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(frame, text=title, font=get_font("body_strong"), text_color=palette_pair("fg")).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=SPACE_LG, pady=(SPACE_MD, SPACE_SM)
        )
        # offset content by one row using add() helpers
        # We just use rows starting at 1 below by re-anchoring children — instead, return a
        # sub-frame that callers add into.
        body = ctk.CTkFrame(frame, fg_color="transparent")
        body.grid(row=1, column=0, columnspan=3, sticky="ew")
        body.grid_columnconfigure(1, weight=1)
        return body

    # ------------------------------------------------------------------
    # Toute la plomberie passe par la facade ShutterstockAIv2 — avant
    # l'audit (B-09) la vue construisait son propre OllamaClient (bypass
    # de la règle « l'UI passe uniquement par la facade ») et l'URL
    # saisie n'était jamais persistée.

    def _check(self) -> None:
        api = self.app.api
        if api is None:
            self._status_label.configure(text="Backend indisponible.", text_color=palette_pair("error"))
            return
        url = self._url.get().strip() or self.DEFAULT_URL
        self._status_label.configure(text="Vérification…", text_color=palette_pair("warning"))

        def worker() -> None:
            try:
                api.set_setting("ollama_url", url)
                api.init_ai(ollama_url=url)  # re-pointe le client sur l'URL saisie
                status = api.check_ai_status()
            except Exception as exc:  # noqa: BLE001
                logger.exception("AI status check failed")
                status = {"available": False, "message": str(exc)}
            self.after(0, lambda s=status: self._on_check(s))

        threading.Thread(target=worker, daemon=True).start()

    def _on_check(self, status: dict[str, Any]) -> None:
        if status.get("available"):
            self._status_label.configure(
                text=f"Connecté — Ollama {status.get('version', 'inconnu')}",
                text_color=palette_pair("success"),
            )
            self._refresh_models()
        else:
            self._status_label.configure(text="Hors ligne ou erreur.", text_color=palette_pair("error"))

    def _refresh_models(self) -> None:
        api = self.app.api
        if api is None:
            return

        def worker() -> None:
            names = api.list_vision_models(refresh=True)
            self.after(0, lambda n=names: self._on_models(n))

        threading.Thread(target=worker, daemon=True).start()

    def _on_models(self, names: list[str]) -> None:
        values = names or ["Aucun modèle vision détecté"]
        self._model_combo.configure(values=values)
        api = self.app.api
        saved = api.get_setting("ollama_model", "") if api else ""
        self._model_combo.set(saved if saved in names else values[0])
        self._model_info.configure(text=f"{len(names)} modèle(s) disponible(s)")

    def _on_model_selected(self, name: str) -> None:
        """Persiste le modèle choisi dans les settings (clé ``ollama_model``)."""
        api = self.app.api
        if api is None or not name or name in ("—",) or name.startswith("Aucun"):
            return
        try:
            api.set_setting("ollama_model", name)
            self._model_info.configure(text=f"Modèle retenu : {name} — « Charger » pour le précharger.")
        except Exception:
            logger.exception("Could not persist model selection")

    def _load_model(self) -> None:
        """Précharge le modèle sélectionné en RAM via la facade."""
        api = self.app.api
        name = self._model_combo.get().strip()
        if api is None or not name or name in ("—",) or name.startswith("Aucun"):
            self._model_info.configure(text="Choisissez d'abord un modèle (Tester la connexion, puis Rafraîchir).")
            return
        self._model_info.configure(text=f"Chargement de {name}…")

        def worker() -> None:
            try:
                ok, msg = api.preload_model(name)
            except Exception as exc:  # noqa: BLE001
                ok, msg = False, str(exc)
            self.after(0, lambda m=msg, o=ok: self._model_info.configure(text=m if o else f"Échec : {m}"))

        threading.Thread(target=worker, daemon=True).start()

    def _test(self) -> None:
        api = self.app.api
        if api is None:
            return
        self._test_result.configure(state="normal")
        self._test_result.delete("1.0", "end")
        self._test_result.insert("1.0", "Test en cours…")
        self._test_result.configure(state="disabled")

        def worker() -> None:
            try:
                if not hasattr(api, "ollama_client"):
                    api.init_ai()
                result = api.ollama_client.test_connection()
            except Exception as e:  # noqa: BLE001
                logger.exception("test_connection failed")
                result = {"success": False, "message": str(e)}
            self.after(0, lambda r=result: self._on_test(r))

        threading.Thread(target=worker, daemon=True).start()

    def _on_test(self, result: dict[str, Any]) -> None:
        self._test_result.configure(state="normal")
        self._test_result.delete("1.0", "end")
        if result.get("success"):
            text = (
                f"Modèle : {result.get('model', '?')}\n"
                f"Réponse : {result.get('message', '')}\n"
                f"Temps de réponse : {result.get('response_time_ms', '?')} ms"
            )
        else:
            text = f"Échec : {result.get('message', 'erreur inconnue')}"
        self._test_result.insert("1.0", text)
        self._test_result.configure(state="disabled")
