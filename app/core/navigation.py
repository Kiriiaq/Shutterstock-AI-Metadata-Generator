"""Router — central view container with history."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import customtkinter as ctk

from app.core.events import EventBus
from app.i18n.fr import t

logger = logging.getLogger(__name__)

ViewFactory = Callable[[ctk.CTkFrame], ctk.CTkFrame]


class Router:
    """Owns the central view container, instantiates views on demand.

    The back/forward history stack was removed in the 2026-06-12 audit:
    the workspace is the single navigable view since the sidebar was
    dropped (Phase 8), so the stack never held more than one entry.
    """

    def __init__(self, container: ctk.CTkFrame, bus: EventBus) -> None:
        self._container = container
        self._bus = bus
        self._factories: dict[str, ViewFactory | None] = {}
        self._labels: dict[str, str] = {}
        self._current_view: ctk.CTkFrame | None = None
        self._current_id: str | None = None

    # ------------------------------------------------------------------
    # Registration

    def register(self, view_id: str, label: str, factory: ViewFactory | None) -> None:
        """Register *factory* under *view_id*. ``factory=None`` → placeholder."""
        self._factories[view_id] = factory
        self._labels[view_id] = label

    @property
    def current_id(self) -> str | None:
        return self._current_id

    def label_for(self, view_id: str) -> str:
        return self._labels.get(view_id, view_id)

    # ------------------------------------------------------------------
    # Navigation

    def navigate_to(self, view_id: str, **kwargs: Any) -> None:
        """Replace the central frame with the view registered under *view_id*."""
        if view_id == self._current_id:
            return
        if view_id not in self._labels:
            logger.warning("Unknown view_id: %s", view_id)
            return

        self._destroy_current()
        view = self._build_view(view_id, kwargs)
        view.grid(row=0, column=0, sticky="nsew")
        self._current_view = view
        self._current_id = view_id

        self._bus.emit("router.navigated", view_id, kwargs)

    # ------------------------------------------------------------------
    # Internals

    def _destroy_current(self) -> None:
        if self._current_view is not None:
            on_leave = getattr(self._current_view, "on_leave", None)
            if callable(on_leave):
                try:
                    on_leave()
                except Exception:
                    logger.exception("View on_leave failed")
            try:
                self._current_view.destroy()
            except Exception:
                logger.exception("View destroy failed")
        self._current_view = None
        self._current_id = None

    def _build_view(self, view_id: str, kwargs: dict[str, Any]) -> ctk.CTkFrame:
        factory = self._factories.get(view_id)
        if factory is None:
            return self._build_placeholder(view_id)
        try:
            view = factory(self._container)
            on_enter = getattr(view, "on_enter", None)
            if callable(on_enter):
                on_enter(**kwargs)
            return view
        except Exception:
            logger.exception("Failed to build view %s", view_id)
            return self._build_error(view_id)

    def _build_placeholder(self, view_id: str) -> ctk.CTkFrame:
        from app.config.theme import SPACE_LG, SPACE_SM, get_font, palette_pair

        frame = ctk.CTkFrame(self._container, fg_color=palette_pair("bg"))
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)
        inner = ctk.CTkFrame(frame, fg_color="transparent")
        inner.grid(row=0, column=0)
        ctk.CTkLabel(
            inner,
            text=t("placeholder.under_construction", label=self._labels.get(view_id, view_id)),
            font=get_font("h2"),
            text_color=palette_pair("fg"),
        ).pack(pady=(0, SPACE_SM))
        ctk.CTkLabel(
            inner,
            text=t("placeholder.under_construction_body"),
            font=get_font("body"),
            text_color=palette_pair("fg_muted"),
        ).pack(pady=(0, SPACE_LG))
        return frame

    def _build_error(self, view_id: str) -> ctk.CTkFrame:
        from app.config.theme import SPACE_LG, get_font, palette_pair

        frame = ctk.CTkFrame(self._container, fg_color=palette_pair("error_bg"))
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)
        ctk.CTkLabel(
            frame,
            text=f"Erreur lors du chargement de la vue « {view_id} ».",
            font=get_font("h3"),
            text_color=palette_pair("error"),
        ).grid(row=0, column=0, padx=SPACE_LG, pady=SPACE_LG)
        return frame
