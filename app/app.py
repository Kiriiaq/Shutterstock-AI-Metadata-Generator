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
    ThemeManager,
    get_font,
    palette_pair,
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
    MIN_SIZE = (900, 600)  # workspace columns are CTkScrollableFrames so panels stay reachable when reduced

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
        # Phase G (2026-05-18) — état Ollama cached pour la chip topbar.
        # Mis à jour par le WorkspaceView._refresh_dynamic_worker qui
        # tourne en background toutes les 5 s ; on évite ainsi un appel
        # HTTP synchrone à chaque ``topbar.refresh_health()``.
        self._ollama_health: tuple[str, str] = ("Inconnu", "muted")

        self._configure_window()
        self._build_layout()
        self.router = Router(self._center, self.bus)
        self._register_views()
        self._register_shortcuts()
        self._register_theme_hooks()

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

    def _apply_modal_icon(self, modal: ctk.CTkToplevel) -> None:
        """Phase G (2026-05-16) : pose l'icône ShutterstockAnalyzer sur
        une fenêtre modale (CTkToplevel) — sinon CTk affiche son icône
        Tcl par défaut en haut-gauche de la barre de titre.

        ``iconbitmap`` sur un CTkToplevel doit être appelé après que le
        Toplevel a fini son setup interne (sinon CTk écrase l'icône
        avec son défaut juste après). On retarde via ``after(200, …)``
        ce qui est le pattern recommandé dans la doc customtkinter.
        Une erreur ici est non bloquante (loggée puis ignorée).
        """
        icon = _resource_path("assets/icons/icone.ico")
        if not icon.exists():
            return

        def _set():
            try:
                modal.iconbitmap(str(icon))
            except Exception as e:
                logger.debug("Could not set modal icon: %s", e)

        try:
            modal.after(200, _set)
        except Exception:
            logger.debug("Could not schedule modal icon set", exc_info=True)

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

        # Unified scroll: ``self._center`` is now a single
        # ``CTkScrollableFrame`` so the entire workspace scrolls as one
        # surface. The previous design had two side-by-side scrollable
        # columns inside the workspace, plus the workspace inside this
        # center frame — three scrollbars competing. Now the workspace
        # columns are plain CTkFrames and overflow flows to *this* one
        # vertical scrollbar at the window level. If content fits, the
        # scrollbar stays hidden; if it overflows, the scroll-window
        # interaction also surfaces layout-overflow regressions early.
        self._center = ctk.CTkScrollableFrame(
            self,
            fg_color=palette_pair("bg"),
            corner_radius=0,
            scrollbar_button_color=palette_pair("border"),
            scrollbar_button_hover_color=palette_pair("border_strong"),
        )
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
        from app.views.expert_report import ExpertReportView
        from app.views.export_batch import ExportBatchView
        from app.views.settings import SettingsView
        from app.views.validate import ValidateView
        from app.views.workspace import WorkspaceView

        self.router.register("home", "Atelier", factory=lambda parent: WorkspaceView(parent, app=self))

        # Detail views — opened only from the corresponding workspace panel.
        self._modal_factories = {
            "settings": lambda parent: SettingsView(parent, app=self),
            "audit": lambda parent: AuditView(parent, app=self),
            "ai_control": lambda parent: AIControlView(parent, app=self),
            "validate": lambda parent: ValidateView(parent, app=self),
            "expert_report": lambda parent: ExpertReportView(parent, app=self),
            "export_batch": lambda parent: ExportBatchView(parent, app=self),
        }
        self._modal_titles = {
            "settings": "Paramètres",
            "audit": "Historique",
            "ai_control": "Modèle IA",
            "validate": "Validation",
            "expert_report": "Rapport expert",
            "export_batch": "Exporter le lot",
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
        self._apply_modal_icon(modal)

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
        # Poids de grille pour les builders qui posent leurs widgets via
        # ``grid(sticky="nsew")`` (détails d'audit) — sans eux la cellule
        # ne s'étire pas et la textbox reste minuscule (audit B-08). Sans
        # effet pour les builders qui utilisent ``pack`` (validation).
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)
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
        self._apply_modal_icon(modal)

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

        Phase G (2026-05-18) : ajout d'une chip ``Ollama`` qui reflète
        l'état du serveur local. Lit ``self._ollama_health`` (mis à
        jour par le WorkspaceView en arrière-plan), ne fait pas
        d'appel HTTP synchrone.
        """
        api = self.api
        return {
            "Édition": self._edition_health(),
            "Backend": ("Disponible", "success") if api else ("Absent", "warning"),
            "ExifTool": (("OK", "success") if (api and api.exiftool_available) else ("Absent", "warning"))
            if api
            else ("—", "muted"),
            "Ollama": self._ollama_health,
        }

    def _edition_health(self) -> tuple[str, str]:
        """Edition chip: ``Pro`` (success) when licensed, else ``Gratuite``.

        Surfaced permanently in the topbar so the freemium state is
        always visible at a glance. We deliberately do *not* render the
        expert-report preview count here: that quota gates a single
        feature, not the whole app, so a global ``n/2`` chip would
        misrepresent everything else (scan, IPTC, mono CSV, FTP) as
        rationed. The precise remaining count lives in the Expert
        Report view, where it is contextually accurate.
        """
        api = self.api
        if api is None:
            return ("—", "muted")
        try:
            if api.license.is_pro():
                return ("Pro", "success")
        except Exception:
            logger.debug("license read failed for edition chip", exc_info=True)
        return ("Gratuite", "muted")

    def _on_license_changed(self, _payload: object = None) -> None:
        """React to Settings activating/deactivating a Pro licence.

        Refreshes the topbar edition chip immediately and re-opens the
        active gated view (if any) so lock badges and the quota banner
        reflect the new tier without forcing a restart.
        """
        try:
            self.topbar.refresh_health()
        except Exception:
            logger.debug("topbar.refresh_health failed on license change", exc_info=True)

    def set_ollama_health(self, label: str, kind: str) -> None:
        """Met à jour l'état Ollama caché + déclenche un refresh de la
        chip dans la topbar. Doit être appelé depuis le main thread —
        les threads doivent passer par ``self.after(0, …)``."""
        self._ollama_health = (label, kind)
        try:
            self.topbar.refresh_health()
        except Exception:
            logger.debug("topbar.refresh_health failed", exc_info=True)

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
        # Phase F (2026-05-14) : registre allégé après suppression des
        # no-ops (palette de commandes, sidebar toggle, search, save view,
        # history back/forward), et ajout des cinq raccourcis panneau
        # Ctrl+1..5 demandés par la campagne de tests T-016..T-020.
        # Les trois actions "focus_panel_*" délèguent à WorkspaceView qui
        # fait le focus (et le scroll si nécessaire) sur le widget cible.
        return {
            "focus_panel_sources": lambda: self._focus_workspace_panel("sources"),
            "focus_panel_editor": lambda: self._focus_workspace_panel("editor"),
            "focus_panel_analyze": lambda: self._focus_workspace_panel("analyze"),
            "open_validate": lambda: self.open_in_modal("validate"),
            "open_history": lambda: self.open_in_modal("audit"),
            "toggle_theme": self._toggle_theme,
            "navigate_settings": lambda: self.open_in_modal("settings"),
            "open_help": self._open_help,
            "close_modal": self._close_top_modal,
        }

    def _focus_workspace_panel(self, panel: str) -> None:
        """Ctrl+1..3 helper: delegate to the workspace's focus_panel().

        The workspace exposes ``focus_panel(name)`` (sources/editor/
        analyze) which performs focus_set on the relevant widget and
        scrolls the surrounding ``CTkScrollableFrame`` so the panel is
        within the viewport. Falls back silently when the workspace
        view isn't current (e.g. window just opened, router still in
        transition).
        """
        view = getattr(self.router, "_current_view", None)
        if view is None:
            return
        focus = getattr(view, "focus_panel", None)
        if callable(focus):
            try:
                focus(panel)
            except Exception:
                logger.exception("focus_panel(%r) failed", panel)

    def _safe_call(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:
            logger.exception("Shortcut handler error")

    # ------------------------------------------------------------------
    # Action implementations (some are placeholders until later phases)

    def _toggle_theme(self) -> None:
        """Cycle light → dark → system → light via the ThemeManager.

        ThemeManager.toggle() does three things atomically:
        1. ``ctk.set_appearance_mode(...)`` — every widget that uses
           ``palette_pair`` tuples auto-bascule.
        2. ``notify_all()`` calls ``apply_theme()`` on every Themeable
           observer (e.g. Topbar, the pilot widget; Phase B will add
           more).
        3. Fires the global hooks registered via
           ``ThemeManager.add_global_hook`` — used here to refresh the
           ttk.Style consumed by DataTable.

        Phase G workaround — bug rétrécissement colonne droite
        ----------------------------------------------------------
        ``set_appearance_mode`` de CTk recalcule le DPI scaling et
        peut grignoter quelques pixels sur le ``CTkScrollableFrame``
        interne à chaque toggle (effet cumulatif : la colonne droite
        du workspace rétrécit visiblement après quelques toggles).

        Le workaround initial (1 seul ``after(50, …)`` qui rappelle
        ``self.geometry(saved)``) ne suffisait pas — CTk continue à
        recalculer plusieurs frames après le ``set_appearance_mode``,
        ce qui peut écraser la 1ʳᵉ restauration.

        Solution (2026-05-19) :
        1. Sauve la geometry AVANT le toggle.
        2. Restaure la geometry à 3 instants (50, 150, 350 ms) pour
           survivre aux recalculs en cascade de CTk.
        3. Re-applique explicitement les ``grid_columnconfigure`` du
           workspace (poids 3 / 2 + minsize 320 sur la col droite)
           pour forcer le scrollable frame interne à respecter à
           nouveau les contraintes de largeur.
        """
        # Snapshot taille fenêtre AVANT le toggle pour la restaurer après.
        try:
            saved_geometry = self.geometry()
        except Exception:
            saved_geometry = None

        new = ThemeManager.get_instance().toggle()
        logger.info("Theme switched to: %s", new)

        def _restore() -> None:
            # 1) Geometry — combat le grignotage CTkScrollableFrame
            if saved_geometry:
                try:
                    self.geometry(saved_geometry)
                except Exception:
                    logger.debug("Geometry restore failed", exc_info=True)
            # 2) Re-trigger les contraintes de largeur du workspace
            #    (sinon le CTkScrollableFrame peut « oublier » le
            #    minsize=320 et laisser la colonne droite rétrécir).
            view = getattr(self.router, "_current_view", None)
            if view is not None:
                try:
                    view.grid_columnconfigure(0, weight=3)
                    view.grid_columnconfigure(1, weight=2, minsize=320)
                except Exception:
                    logger.debug("Column reconfigure failed", exc_info=True)
            # 3) Force un repaint complet du shell
            try:
                self.update_idletasks()
            except Exception:
                pass

        # Trois passes pour survivre aux recalculs en cascade de CTk
        # après ``set_appearance_mode``. Chaque délai cible une étape
        # différente : 50 ms (premier idle), 150 ms (DPI tracker
        # poll), 350 ms (re-render final).
        for delay in (50, 150, 350):
            self.after(delay, _restore)

    def _register_theme_hooks(self) -> None:
        """Wire the ttk.Style refresh into the ThemeManager dispatch.

        ttk.Style is process-global and not CTk-aware, so a Treeview
        keeps its old colours after ``set_appearance_mode`` unless we
        re-call ``apply_treeview_style()``. We piggyback on the
        ThemeManager's ``notify_all()`` chain so this stays a single
        observer registration instead of being scattered.
        """
        from app.components.data_table import DataTable, apply_treeview_style

        def _refresh_ttk() -> None:
            apply_treeview_style()
            for table in self._iter_widgets(self, DataTable):
                try:
                    table.refresh_theme()
                except Exception:
                    logger.exception("DataTable refresh_theme failed")

        ThemeManager.get_instance().add_global_hook(_refresh_ttk)

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
        self._apply_modal_icon(modal)

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
                else:
                    # Surface the action — without this, Esc looks like it
                    # does nothing because the cancel is cooperative and
                    # takes effect on the next image boundary. The toast +
                    # log give the user (and the test campaign) clear
                    # confirmation that Esc was acted upon.
                    logger.info("Escape → cancel processing requested")
                    try:
                        self.toasts.show("Annulation demandée…", kind="info", timeout_ms=2500)
                    except Exception:
                        logger.exception("Cancel toast failed")

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
