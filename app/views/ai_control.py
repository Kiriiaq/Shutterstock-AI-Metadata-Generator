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
from app.views.base_view import BaseView

if TYPE_CHECKING:
    from app.app import App

logger = logging.getLogger(__name__)


class AIControlView(BaseView):
    view_id = "ai_control"

    DEFAULT_URL = "http://localhost:11434"

    def __init__(self, master: ctk.CTkFrame, *, app: "App") -> None:
        super().__init__(master)
        self.app = app
        self._client: Any | None = None
        self._build()

    # ------------------------------------------------------------------

    def _build(self) -> None:
        wrapper = ctk.CTkScrollableFrame(self, fg_color="transparent")
        wrapper.grid(row=0, column=0, sticky="nsew", padx=SPACE_XL, pady=SPACE_XL)
        wrapper.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(wrapper, text="Modèle IA", font=get_font("h1"), text_color=palette_pair("fg")).grid(
            row=0, column=0, sticky="w", pady=(0, SPACE_LG)
        )

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
        self._model_combo = ctk.CTkComboBox(section, values=["—"], width=320, font=get_font("body"))
        self._model_combo.set("—")
        self._model_combo.grid(row=0, column=1, sticky="ew", padx=(0, SPACE_LG), pady=SPACE_MD)
        ctk.CTkButton(section, text="Rafraîchir", command=self._refresh_models).grid(
            row=0, column=2, padx=SPACE_LG, pady=SPACE_MD
        )

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

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from src.modules.ai.ollama_client import OllamaClient

            self._client = OllamaClient(base_url=self._url.get().strip() or self.DEFAULT_URL)
            return self._client
        except Exception:
            logger.exception("Could not build OllamaClient")
            return None

    def _check(self) -> None:
        client = self._ensure_client()
        if client is None:
            self._status_label.configure(text="Client indisponible.", text_color=palette_pair("error"))
            return
        client.base_url = self._url.get().strip() or self.DEFAULT_URL
        self._status_label.configure(text="Vérification…", text_color=palette_pair("warning"))
        threading.Thread(target=self._check_worker, args=(client,), daemon=True).start()

    def _check_worker(self, client) -> None:
        ok = client.check_connection()
        info = client.get_status_info()
        self.after(0, lambda: self._on_check(ok, info))

    def _on_check(self, ok: bool, info: dict[str, Any]) -> None:
        if ok:
            self._status_label.configure(
                text=f"Connecté — Ollama {info.get('version', 'inconnu')}",
                text_color=palette_pair("success"),
            )
            self._refresh_models()
        else:
            self._status_label.configure(text="Hors ligne ou erreur.", text_color=palette_pair("error"))

    def _refresh_models(self) -> None:
        client = self._ensure_client()
        if client is None:
            return
        threading.Thread(target=self._refresh_worker, args=(client,), daemon=True).start()

    def _refresh_worker(self, client) -> None:
        try:
            models = client.list_vision_models()
        except Exception:
            logger.exception("list_vision_models failed")
            models = []
        names = [m.name for m in models] or ["Aucun modèle vision détecté"]
        self.after(0, lambda: self._on_models(names))

    def _on_models(self, names: list[str]) -> None:
        self._model_combo.configure(values=names)
        self._model_combo.set(names[0])
        self._model_info.configure(text=f"{len(names)} modèle(s) disponible(s)")

    def _test(self) -> None:
        client = self._ensure_client()
        if client is None:
            return
        self._test_result.configure(state="normal")
        self._test_result.delete("1.0", "end")
        self._test_result.insert("1.0", "Test en cours…")
        self._test_result.configure(state="disabled")
        threading.Thread(target=self._test_worker, args=(client,), daemon=True).start()

    def _test_worker(self, client) -> None:
        try:
            result = client.test_connection()
        except Exception as e:
            logger.exception("test_connection failed")
            result = {"success": False, "message": str(e)}
        self.after(0, lambda r=result: self._on_test(r))

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
