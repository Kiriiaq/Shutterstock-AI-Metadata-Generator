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

from app.components.command_palette import Command, CommandPalette
from app.components.confirm_dialog import confirm
from app.components.context_panel import ContextPanel
from app.components.sidebar import NAV_ENTRIES, Sidebar
from app.components.toast import ToastManager
from app.components.topbar import Topbar
from app.config.shortcuts import GLOBAL_SHORTCUTS, display_label
from app.config.theme import (
    RADIUS_LG,
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    get_color,
    get_font,
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
        self._palette: CommandPalette | None = None

        self._configure_window()
        self._build_layout()
        self.router = Router(self._center, self.bus)
        self._register_views()
        self._register_shortcuts()
        self._wire_navigation()

        # Initial view
        self.router.navigate_to("home")

    # ------------------------------------------------------------------
    # Window setup

    def _configure_window(self) -> None:
        self.title(t("app.title"))
        self.geometry(self.INITIAL_GEOMETRY)
        self.minsize(*self.MIN_SIZE)
        self.configure(fg_color=get_color("bg"))

        icon = _resource_path("assets/icons/icone.ico")
        if icon.exists():
            try:
                self.iconbitmap(str(icon))
            except Exception as e:
                logger.warning("Could not set window icon: %s", e)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Sidebar (col 0, spans both rows)
        self.sidebar = Sidebar(self, on_navigate=self._navigate)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="ns")

        # Topbar (col 1-2, row 0)
        self.topbar = Topbar(
            self,
            on_search_trigger=self._open_command_palette,
            on_theme_toggle=self._toggle_theme,
            on_help=self._open_help,
        )
        self.topbar.grid(row=0, column=1, columnspan=2, sticky="new")

        # Central container (col 1, row 1)
        self._center = ctk.CTkFrame(self, fg_color=get_color("bg"), corner_radius=0)
        self._center.grid(row=1, column=1, sticky="nsew")
        self._center.grid_columnconfigure(0, weight=1)
        self._center.grid_rowconfigure(0, weight=1)

        # Context panel (col 2, row 1) — hidden by default (width 0).
        self.context_panel = ContextPanel(self)
        self.context_panel.grid(row=1, column=2, sticky="ns")

    # ------------------------------------------------------------------
    # Routing

    def _register_views(self) -> None:
        """Wire each sidebar entry to its concrete view factory.

        The default ``home`` is now the dense WorkspaceView — a single
        screen that surfaces sources, editor, analyse, system status
        and live audit at once. The other views remain registered so
        the SystemPanel quick-action buttons and the sidebar can still
        deep-link into them when the user wants details.
        """
        from app.views.ai_control import AIControlView
        from app.views.analyze import AnalyzeView
        from app.views.audit import AuditView
        from app.views.editor import EditorView
        from app.views.settings import SettingsView
        from app.views.sources import SourcesView
        from app.views.upload import UploadView
        from app.views.validate import ValidateView
        from app.views.workspace import WorkspaceView

        factories: dict[str, Any] = {
            "home": lambda parent: WorkspaceView(parent, app=self),
            "sources": lambda parent: SourcesView(parent, app=self),
            "analyze": lambda parent: AnalyzeView(parent, app=self),
            "editor": lambda parent: EditorView(parent, app=self),
            "validate": lambda parent: ValidateView(parent, app=self),
            "upload": lambda parent: UploadView(parent, app=self),
            "ai_control": lambda parent: AIControlView(parent, app=self),
            "audit": lambda parent: AuditView(parent, app=self),
            "settings": lambda parent: SettingsView(parent, app=self),
        }
        for view_id, _icon, label_key, _section in NAV_ENTRIES:
            self.router.register(view_id, t(label_key), factory=factories.get(view_id))

    def _wire_navigation(self) -> None:
        self.bus.on("router.navigated", self._on_router_navigated)

    def _on_router_navigated(self, view_id: str, _kwargs: dict[str, Any]) -> None:
        self.sidebar.set_active(view_id)
        self.topbar.set_breadcrumb(self.router.label_for(view_id))

    def _navigate(self, view_id: str) -> None:
        self.router.navigate_to(view_id)

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
        return {
            "open_command_palette": self._open_command_palette,
            "toggle_sidebar": self.sidebar.toggle_collapsed,
            "toggle_theme": self._toggle_theme,
            "navigate_settings": lambda: self.router.navigate_to("settings"),
            "focus_view_search": self._focus_view_search,
            "save_current_view": self._save_current_view,
            "open_help": self._open_help,
            "close_modal": self._close_top_modal,
            "history_back": self.router.back,
            "history_forward": self.router.forward,
        }

    def _safe_call(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:
            logger.exception("Shortcut handler error")

    # ------------------------------------------------------------------
    # Action implementations (some are placeholders until later phases)

    def _toggle_theme(self) -> None:
        new = toggle_theme()
        self.sidebar.refresh_theme()
        self.topbar.refresh_theme()
        self.context_panel.refresh_theme()
        self.configure(fg_color=get_color("bg"))
        self._center.configure(fg_color=get_color("bg"))
        logger.info("Theme switched to: %s", new)

    def _open_command_palette(self) -> None:
        if self._palette is None:
            self._palette = CommandPalette(self, provider=self._build_commands)
        self._palette.open()

    def _build_commands(self) -> list[Command]:
        """Source of truth for the command palette.

        Combines navigation commands (one per registered view) with the
        global actions that have a meaningful display label.
        """
        cmds: list[Command] = []
        for view_id, _icon, label_key, _section in NAV_ENTRIES:
            cmds.append(
                Command(
                    id=f"nav.{view_id}",
                    label=f"Aller à : {t(label_key)}",
                    callback=lambda vid=view_id: self.router.navigate_to(vid),
                    keywords=("naviguer", "ouvrir"),
                )
            )
        cmds.extend(
            [
                Command(
                    id="toggle_theme",
                    label="Basculer le thème clair / sombre",
                    callback=self._toggle_theme,
                    shortcut="Ctrl+Shift+T",
                ),
                Command(
                    id="toggle_sidebar",
                    label="Replier / déplier la barre latérale",
                    callback=self.sidebar.toggle_collapsed,
                    shortcut="Ctrl+B",
                ),
                Command(
                    id="history_back",
                    label="Vue précédente",
                    callback=self.router.back,
                    shortcut="Alt+←",
                ),
                Command(
                    id="history_forward",
                    label="Vue suivante",
                    callback=self.router.forward,
                    shortcut="Alt+→",
                ),
                Command(
                    id="open_help",
                    label="Afficher l'aide des raccourcis",
                    callback=self._open_help,
                    shortcut="F1",
                ),
            ]
        )
        return cmds

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
        modal.configure(fg_color=get_color("bg"))
        modal.geometry("520x460")

        outer = ctk.CTkFrame(
            modal,
            fg_color=get_color("bg_elevated"),
            corner_radius=RADIUS_LG,
            border_color=get_color("border"),
            border_width=1,
        )
        outer.pack(fill="both", expand=True, padx=SPACE_LG, pady=SPACE_LG)

        ctk.CTkLabel(
            outer,
            text=t("help.title"),
            font=get_font("h2"),
            text_color=get_color("fg"),
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
                text_color=get_color("accent"),
                anchor="w",
                width=140,
            ).grid(row=idx, column=0, sticky="w", padx=(0, SPACE_MD), pady=2)
            ctk.CTkLabel(
                scroll,
                text=label,
                font=get_font("body"),
                text_color=get_color("fg"),
                anchor="w",
                justify="left",
            ).grid(row=idx, column=1, sticky="w", pady=2)

        ctk.CTkButton(
            outer,
            text=t("common.close"),
            command=modal.destroy,
            fg_color=get_color("accent"),
            hover_color=get_color("accent_hover"),
            text_color=get_color("accent_fg"),
            font=get_font("body_strong"),
            height=36,
        ).pack(side="right", padx=SPACE_LG, pady=(0, SPACE_LG))

        modal.bind("<Escape>", lambda _e: modal.destroy())
        try:
            modal.grab_set()
        except Exception:
            pass
        self._open_modals.append(modal)

    def _focus_view_search(self) -> None:
        # Each view will eventually expose a focus_search() hook. Stub for now.
        self.toasts.show("Recherche par vue — à connecter par chaque vue.", kind="info")

    def _save_current_view(self) -> None:
        self.toasts.show("Enregistrement contextuel à connecter par chaque vue.", kind="info")

    def _close_top_modal(self) -> None:
        for modal in reversed(self._open_modals):
            try:
                if modal.winfo_exists():
                    modal.destroy()
                    return
            except Exception:
                continue

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
