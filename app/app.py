"""Application shell — composes Sidebar + Topbar + central container.

Owns the singletons: EventBus, AppState, Router, ToastManager.
Resolves keyboard shortcut action_ids to real callbacks at bind time so
``app/config/shortcuts.py`` stays declarative and the help dialog can be
auto-generated from the same source.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import customtkinter as ctk

from app.components.confirm_dialog import confirm
from app.components.toast import ToastManager
from app.components.topbar import Topbar
from app.config.shortcuts import GLOBAL_SHORTCUTS, display_label
from app.config.theme import (
    RADIUS_LG,
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    get_font,
    palette_pair,
    toggle_theme,
)
from app.core.events import EventBus
from app.core.navigation import Router
from app.core.state import AppState
from app.i18n.fr import t

logger = logging.getLogger(__name__)


def _resource_path(relative: str) -> Path:
    """Resolve a path under the source tree or PyInstaller bundle."""
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent.parent))
    else:
        base = Path(__file__).parent.parent
    return base / relative


class App(ctk.CTk):
    """Main application window."""

    INITIAL_GEOMETRY = "1400x900"
    MIN_SIZE = (1200, 800)

    def __init__(self, *, api: Any | None = None) -> None:
        super().__init__()
        self.api = api  # ShutterstockAIv2 | None
        self.bus = EventBus()
        # NOTE: must NOT be named `state` — that shadows tkinter.Wm.state(),
        # which CustomTkinter's DPI scaling tracker calls periodically and
        # would crash with "'AppState' object is not callable".
        self.app_state = AppState(self.bus)
        self.toasts = ToastManager(self)
        self._open_modals: list[ctk.CTkToplevel] = []
        # Detail-view factories used by ``open_in_modal`` — populated in
        # ``_register_views``. Each tool's detail view is reachable from
        # exactly one place: a button in its workspace panel.
        self._modal_factories: dict[str, Any] = {}
        self._modal_titles: dict[str, str] = {}

        self._configure_window()
        self._build_layout()
        self.router = Router(self._center, self.bus)
        self._register_views()
        self._register_shortcuts()

        # The workspace is the only navigable view — no sidebar, no nav.
        self.router.navigate_to("home")

    # ------------------------------------------------------------------
    # Window setup

    def _configure_window(self) -> None:
        self.title(t("app.title"))
        self.geometry(self.INITIAL_GEOMETRY)
        self.minsize(*self.MIN_SIZE)
        self.configure(fg_color=palette_pair("bg"))

        icon = _resource_path("assets/icons/icone.ico")
        if icon.exists():
            try:
                self.iconbitmap(str(icon))
            except Exception as e:
                logger.warning("Could not set window icon: %s", e)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        # No sidebar — every tool has its own panel inside the workspace.
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.topbar = Topbar(
            self,
            on_theme_toggle=self._toggle_theme,
            on_help=self._open_help,
            health_provider=self._global_health,
        )
        self.topbar.grid(row=0, column=0, sticky="new")

        self._center = ctk.CTkFrame(self, fg_color=palette_pair("bg"), corner_radius=0)
        self._center.grid(row=1, column=0, sticky="nsew")
        self._center.grid_columnconfigure(0, weight=1)
        self._center.grid_rowconfigure(0, weight=1)

    # ------------------------------------------------------------------
    # Routing

    def _register_views(self) -> None:
        """Register the workspace as the only navigable view, and store
        modal factories for the 5 detail views (settings, audit,
        ai_control, validate, upload) so each tool's panel can open
        its full surface in a Toplevel from exactly one button.
        """
        from app.views.ai_control import AIControlView
        from app.views.audit import AuditView
        from app.views.settings import SettingsView
        from app.views.upload import UploadView
        from app.views.validate import ValidateView
        from app.views.workspace import WorkspaceView

        self.router.register("home", "Atelier", factory=lambda parent: WorkspaceView(parent, app=self))

        # Detail views — opened only from the corresponding workspace panel.
        self._modal_factories = {
            "settings": lambda parent: SettingsView(parent, app=self),
            "audit": lambda parent: AuditView(parent, app=self),
            "ai_control": lambda parent: AIControlView(parent, app=self),
            "validate": lambda parent: ValidateView(parent, app=self),
            "upload": lambda parent: UploadView(parent, app=self),
        }
        self._modal_titles = {
            "settings": "Paramètres",
            "audit": "Historique",
            "ai_control": "Modèle IA",
            "validate": "Validation",
            "upload": "Téléversement FTPS",
        }

    def show_details(self, title: str, builder: Callable[[ctk.CTkFrame], None]) -> None:
        """Open a small Toplevel filled by *builder* — replaces the old
        right-side ContextPanel for one-shot detail panes (audit row,
        validation issues, etc.). Closed via Esc / window-close.
        """
        modal = ctk.CTkToplevel(self)
        modal.title(title)
        modal.geometry("420x520")
        modal.transient(self)
        modal.configure(fg_color=palette_pair("bg"))

        inner = ctk.CTkFrame(
            modal,
            fg_color=palette_pair("bg_elevated"),
            border_color=palette_pair("border"),
            border_width=1,
            corner_radius=RADIUS_LG,
        )
        inner.pack(fill="both", expand=True, padx=SPACE_LG, pady=SPACE_LG)

        ctk.CTkLabel(
            inner,
            text=title,
            font=get_font("h3"),
            text_color=palette_pair("fg"),
            anchor="w",
        ).pack(fill="x", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM))

        body = ctk.CTkFrame(inner, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=SPACE_LG, pady=(0, SPACE_LG))
        try:
            builder(body)
        except Exception:
            logger.exception("show_details builder failed")
            ctk.CTkLabel(
                body,
                text="Erreur de chargement.",
                text_color=palette_pair("error"),
                font=get_font("body"),
            ).pack()

        ctk.CTkButton(
            inner,
            text=t("common.close"),
            command=modal.destroy,
            fg_color=palette_pair("accent"),
            hover_color=palette_pair("accent_hover"),
            text_color=palette_pair("accent_fg"),
            font=get_font("body_strong"),
            height=32,
        ).pack(side="right", padx=SPACE_LG, pady=(0, SPACE_LG))

        modal.bind("<Escape>", lambda _e: modal.destroy())
        try:
            modal.grab_set()
        except Exception:
            pass
        self._open_modals.append(modal)

    def open_in_modal(self, view_id: str) -> None:
        """Open the detail view registered under *view_id* as a modal Toplevel.

        Called from the workspace panels' "Détail…" / "Modifier…" / "Tout
        voir…" buttons. Exactly one entry-point per tool — no sidebar
        duplication.
        """
        factory = self._modal_factories.get(view_id)
        if factory is None:
            self.toasts.show(f"Vue inconnue : {view_id}", kind="error")
            return
        modal = ctk.CTkToplevel(self)
        modal.title(self._modal_titles.get(view_id, view_id))
        modal.geometry("1100x780")
        modal.transient(self)
        modal.configure(fg_color=palette_pair("bg"))

        container = ctk.CTkFrame(modal, fg_color=palette_pair("bg"))
        container.pack(fill="both", expand=True)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)
        try:
            view = factory(container)
            on_enter = getattr(view, "on_enter", None)
            if callable(on_enter):
                on_enter()
            view.grid(row=0, column=0, sticky="nsew")
        except Exception:
            logger.exception("Failed to open modal view %s", view_id)
            ctk.CTkLabel(container, text=f"Erreur : impossible d'ouvrir « {view_id} ».").grid(row=0, column=0)

        modal.bind("<Escape>", lambda _e: modal.destroy())
        try:
            modal.grab_set()
        except Exception:
            pass
        self._open_modals.append(modal)

    def _global_health(self) -> dict[str, tuple[str, str]]:
        """Return (label, color_kind) tuples for the topbar health strip.

        color_kind ∈ {success, warning, error, muted}. Topbar resolves
        to a real palette colour at draw time.
        """
        api = self.api
        return {
            "Backend": ("Disponible", "success") if api else ("Absent", "warning"),
            "ExifTool": (("OK", "success") if (api and api.exiftool_available) else ("Absent", "warning"))
            if api
            else ("—", "muted"),
        }

    # ------------------------------------------------------------------
    # Shortcut wiring

    def _register_shortcuts(self) -> None:
        action_map = self._build_action_map()
        for binding, _label_key, action_id in GLOBAL_SHORTCUTS:
            handler = action_map.get(action_id)
            if handler is None:
                logger.warning("No handler for action_id=%s", action_id)
                continue
            self.bind_all(binding, lambda _e, h=handler: self._safe_call(h))

    def _build_action_map(self) -> dict[str, Callable[[], None]]:
        # No sidebar / nav router → most shortcut actions are now no-ops or
        # mapped to modal-opening. Kept declarative so help dialog still
        # mirrors the keys.
        return {
            "open_command_palette": lambda: None,  # no global nav anymore
            "toggle_sidebar": lambda: None,
            "toggle_theme": self._toggle_theme,
            "navigate_settings": lambda: self.open_in_modal("settings"),
            "focus_view_search": lambda: None,
            "save_current_view": lambda: None,
            "open_help": self._open_help,
            "close_modal": self._close_top_modal,
            "history_back": lambda: None,
            "history_forward": lambda: None,
        }

    def _safe_call(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:
            logger.exception("Shortcut handler error")

    # ------------------------------------------------------------------
    # Action implementations (some are placeholders until later phases)

    def _toggle_theme(self) -> None:
        """Switch light <-> dark.

        Every CTk widget in app/ is now constructed with tuple
        ``(light, dark)`` colors via ``palette_pair(...)``, so
        ``ctk.set_appearance_mode(...)`` (called inside ``toggle_theme``)
        propagates the new theme to every live widget without a
        destroy/rebuild — user state in panels (folder path, IPTC
        drafts, scroll positions, audit table contents) is preserved.

        Two surfaces don't auto-switch and need an explicit refresh:
        - ``ttk.Style`` used by DataTable's Treeview (Tk-native, not CTk).
        - The topbar health strip, whose chip colors are computed at
          build time by a provider — rebuilt via ``refresh_theme``.
        """
        new = toggle_theme()
        self.topbar.refresh_theme()
        # Refresh the ttk.Style used by every DataTable. The style is
        # global, so one call covers all live tables; we still iterate
        # to update each tree's row-tag background.
        from app.components.data_table import DataTable, apply_treeview_style

        apply_treeview_style()
        for table in self._iter_widgets(self, DataTable):
            try:
                table.refresh_theme()
            except Exception:
                logger.exception("DataTable refresh_theme failed")
        logger.info("Theme switched to: %s", new)

    def _iter_widgets(self, root: ctk.CTkBaseClass, kind: type):
        """Yield every descendant of *root* that ``isinstance(.., kind)``."""
        try:
            children = root.winfo_children()
        except Exception:
            return
        for child in children:
            if isinstance(child, kind):
                yield child
            yield from self._iter_widgets(child, kind)

    def _open_help(self) -> None:
        """Render GLOBAL_SHORTCUTS as a 2-column dialog (binding · description).

        Duplicates that differ only in letter-case (Ctrl+k vs Ctrl+K) are
        collapsed in the display.
        """
        seen: set[str] = set()
        rows: list[tuple[str, str]] = []
        for binding, label_key, _action in GLOBAL_SHORTCUTS:
            disp = display_label(binding)
            key = disp.upper()
            if key in seen:
                continue
            seen.add(key)
            rows.append((disp, t(label_key)))

        modal = ctk.CTkToplevel(self)
        modal.title(t("help.title"))
        modal.transient(self)
        modal.configure(fg_color=palette_pair("bg"))
        modal.geometry("520x460")

        outer = ctk.CTkFrame(
            modal,
            fg_color=palette_pair("bg_elevated"),
            corner_radius=RADIUS_LG,
            border_color=palette_pair("border"),
            border_width=1,
        )
        outer.pack(fill="both", expand=True, padx=SPACE_LG, pady=SPACE_LG)

        ctk.CTkLabel(
            outer,
            text=t("help.title"),
            font=get_font("h2"),
            text_color=palette_pair("fg"),
            anchor="w",
        ).pack(fill="x", padx=SPACE_LG, pady=(SPACE_LG, SPACE_SM))

        scroll = ctk.CTkScrollableFrame(outer, fg_color="transparent", corner_radius=0)
        scroll.pack(fill="both", expand=True, padx=SPACE_LG, pady=(0, SPACE_MD))
        scroll.grid_columnconfigure(1, weight=1)

        for idx, (binding, label) in enumerate(rows):
            ctk.CTkLabel(
                scroll,
                text=binding,
                font=get_font("code"),
                text_color=palette_pair("accent"),
                anchor="w",
                width=140,
            ).grid(row=idx, column=0, sticky="w", padx=(0, SPACE_MD), pady=2)
            ctk.CTkLabel(
                scroll,
                text=label,
                font=get_font("body"),
                text_color=palette_pair("fg"),
                anchor="w",
                justify="left",
            ).grid(row=idx, column=1, sticky="w", pady=2)

        ctk.CTkButton(
            outer,
            text=t("common.close"),
            command=modal.destroy,
            fg_color=palette_pair("accent"),
            hover_color=palette_pair("accent_hover"),
            text_color=palette_pair("accent_fg"),
            font=get_font("body_strong"),
            height=36,
        ).pack(side="right", padx=SPACE_LG, pady=(0, SPACE_LG))

        modal.bind("<Escape>", lambda _e: modal.destroy())
        try:
            modal.grab_set()
        except Exception:
            pass
        self._open_modals.append(modal)

    def _close_top_modal(self) -> None:
        """Esc handler: close the top-most modal, or cancel processing.

        Escape priority:
        1. If any modal Toplevel is alive, destroy the most recent one.
        2. Else, if the workspace is mid-processing (analyse IA), cancel
           it via the workspace's stop method — matches the user's
           expectation that Esc and the "Arrêter" button do the same
           thing.
        """
        # Prune destroyed modals (X-clicked or self-destroyed via Esc).
        self._open_modals[:] = [m for m in self._open_modals if m.winfo_exists()]
        if self._open_modals:
            try:
                self._open_modals.pop().destroy()
            except Exception:
                logger.exception("Modal close failed")
            return
        # Workspace fallback: cancel running batch if any.
        workspace = getattr(self.router, "_current_view", None)
        if workspace is not None and getattr(workspace, "_processing", False):
            stop = getattr(workspace, "_analyze_stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    logger.exception("Workspace cancel failed")

    # ------------------------------------------------------------------
    # Lifecycle

    def _on_close(self) -> None:
        if self.api is not None:
            try:
                self.api.close()
            except Exception:
                logger.exception("Backend cleanup failed")
        # confirm() is reserved for destructive actions — closing the
        # window is a normal flow and should not nag.
        self.destroy()

    def confirm_destructive(self, *, title: str, message: str) -> bool:
        """Helper exposed to vues for delete-style confirmations."""
        return confirm(self, title=title, message=message, destructive=True)
