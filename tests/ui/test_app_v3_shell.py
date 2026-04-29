"""End-to-end smoke for the new ``app/`` UI layer.

Tk does not allow re-creating a root within a single Python process,
so this single test owns the only ``CTk`` instance for the whole UI
test session. It exercises:

- App shell construction and layout (sidebar + topbar + central frame).
- Navigation through all 9 registered views (home, sources, analyze,
  editor, validate, upload, ai_control, audit, settings) — proves each
  view's factory builds without raising even when ``api=None``.
- History back/forward.
- Sidebar collapse/expand.
- Theme toggle with persistence to a tmp prefs file.
- display_label() regression guards.
- Command palette + context panel wiring.
- The 5 reusable components attached as children of the App root:
  CommandPalette, DataTable (set/sort/select/refresh), FormField
  (validate, set_error, set_value), EmptyState, ContextPanel
  (open/close/refresh_theme).
"""

from __future__ import annotations

import pytest


def test_ui_v3_full_lifecycle(tmp_path, monkeypatch):
    pytest.importorskip("customtkinter")

    from app.config import theme as theme_mod

    fake_prefs = tmp_path / "ui_prefs.json"
    monkeypatch.setattr(theme_mod, "get_prefs_path", lambda: fake_prefs)

    import customtkinter as ctk

    from app.app import App
    from app.components.command_palette import Command, CommandPalette
    from app.components.context_panel import ContextPanel
    from app.components.data_table import Column, DataTable
    from app.components.empty_state import EmptyState
    from app.components.form_field import FormField, entry_factory
    from app.config.shortcuts import GLOBAL_SHORTCUTS, display_label

    app = App(api=None)
    try:
        app.update_idletasks()

        _check_shell(app)
        _check_navigation(app)
        _check_history(app)
        _check_sidebar_collapse(app)
        _check_theme_toggle(app, theme_mod, fake_prefs)
        _check_shortcut_labels(display_label, GLOBAL_SHORTCUTS)
        _check_palette_and_panel(app, CommandPalette)

        # ----- Component-level checks (children of App root) -----
        _check_command_palette_filter(app, Command, CommandPalette)
        _check_data_table(app, ctk, Column, DataTable)
        _check_form_field(app, ctk, FormField, entry_factory)
        _check_empty_state(app, ctk, EmptyState)
        _check_context_panel_standalone(app, ContextPanel)

    finally:
        try:
            app.destroy()
        except Exception:
            pass


# ============================================================================
# Shell-level helpers
# ============================================================================


def _check_shell(app) -> None:
    assert "ShutterstockAnalyzer" in app.title()
    assert app.sidebar is not None
    assert app.topbar is not None
    assert app._center.winfo_exists()
    assert app.router.current_id == "home"


def _check_navigation(app) -> None:
    from app.components.sidebar import NAV_ENTRIES

    for view_id, _icon, _label_key, _section in NAV_ENTRIES:
        app.router.navigate_to(view_id)
        app.update_idletasks()
        assert app.router.current_id == view_id


def _check_history(app) -> None:
    app.router.back()
    app.update_idletasks()
    app.router.forward()
    app.update_idletasks()


def _check_sidebar_collapse(app) -> None:
    original_width = app.sidebar.cget("width")
    app.sidebar.toggle_collapsed()
    app.update_idletasks()
    assert app.sidebar.cget("width") != original_width
    app.sidebar.toggle_collapsed()
    app.update_idletasks()


def _check_theme_toggle(app, theme_mod, fake_prefs) -> None:
    before = theme_mod.load_theme_pref()
    app._toggle_theme()
    after = theme_mod.load_theme_pref()
    assert before != after
    assert fake_prefs.exists()


def _check_shortcut_labels(display_label, GLOBAL_SHORTCUTS) -> None:
    assert display_label("<Alt-Left>") == "Alt+Left"
    assert display_label("<Control-k>") == "Ctrl+K"
    assert display_label("<Control-Shift-T>") == "Ctrl+Shift+T"
    assert display_label("<F1>") == "F1"
    assert len(GLOBAL_SHORTCUTS) >= 12


def _check_palette_and_panel(app, CommandPalette) -> None:
    app._open_command_palette()
    app.update_idletasks()
    assert isinstance(app._palette, CommandPalette)
    cmds = app._build_commands()
    assert len(cmds) >= 9 + 5
    cmd_ids = {c.id for c in cmds}
    assert "nav.home" in cmd_ids and "toggle_theme" in cmd_ids
    app._palette.close()

    assert app.context_panel.is_open is False
    app.context_panel.set_content("Détails", lambda _p: None)
    app.context_panel.open()
    app.update_idletasks()
    assert app.context_panel.is_open is True
    app.context_panel.close()
    app.update_idletasks()
    assert app.context_panel.is_open is False


# ============================================================================
# Component-level helpers (children of the App root)
# ============================================================================


def _check_command_palette_filter(app, Command, CommandPalette) -> None:
    fired: list[str] = []
    cmds = [
        Command(id="a", label="Aller à : Sources", callback=lambda: fired.append("a")),
        Command(id="b", label="Basculer le thème", callback=lambda: fired.append("b")),
        Command(id="c", label="Vue précédente", callback=lambda: fired.append("c"), keywords=("history",)),
    ]
    palette = CommandPalette(app, provider=lambda: cmds)
    palette.open()
    app.update_idletasks()
    assert palette._row_widgets and len(palette._row_widgets) == 3

    palette._refresh_results("thème")
    assert len(palette._row_widgets) == 1 and palette._row_widgets[0][1].id == "b"

    palette._refresh_results("history")
    assert len(palette._row_widgets) == 1 and palette._row_widgets[0][1].id == "c"

    palette._highlighted = 0
    palette._execute_highlighted()
    assert fired == ["c"]
    assert palette._win is None


def _check_data_table(app, ctk, Column, DataTable) -> None:
    container = ctk.CTkFrame(app)
    container.grid(row=99, column=99)  # off-grid; app root manages by grid
    table = DataTable(
        container,
        columns=[
            Column(id="name", label="Nom", width=120),
            Column(id="size", label="Taille", width=80, sort_key=int),
        ],
    )
    table.pack(fill="both", expand=True)
    app.update_idletasks()

    table.set_rows(
        [
            {"name": "b.jpg", "size": 200},
            {"name": "a.jpg", "size": 50},
            {"name": "c.jpg", "size": 100},
        ]
    )
    app.update_idletasks()
    assert len(table._tree.get_children()) == 3

    table._sort_by("size")
    rows_asc = [table._row_data[iid]["size"] for iid in table._tree.get_children()]
    assert rows_asc == sorted(rows_asc)

    table._sort_by("size")
    rows_desc = [table._row_data[iid]["size"] for iid in table._tree.get_children()]
    assert rows_desc == sorted(rows_desc, reverse=True)

    table._tree.selection_set(table._tree.get_children()[0])
    selected = table.get_selected()
    assert len(selected) == 1
    assert selected[0]["size"] == 200

    table.refresh_theme()
    container.destroy()


def _check_form_field(app, ctk, FormField, entry_factory) -> None:
    container = ctk.CTkFrame(app)
    container.grid(row=99, column=99)  # off-grid; app root manages by grid
    field = FormField(
        container,
        label="Nom",
        required=True,
        widget_factory=entry_factory,
        validator=lambda v: None if v else "Ce champ est requis.",
    )
    field.pack()
    app.update_idletasks()

    assert field.validate() is False
    assert field._error_message == "Ce champ est requis."

    field.set_value("Marc")
    assert field.value == "Marc"
    assert field.validate() is True
    assert field._error_message is None

    field.set_error("Erreur custom")
    assert field._error_message == "Erreur custom"
    field.set_error(None)
    assert field._error_message is None
    container.destroy()


def _check_empty_state(app, ctk, EmptyState) -> None:
    container = ctk.CTkFrame(app)
    container.grid(row=99, column=99)  # off-grid; app root manages by grid
    fired: list[bool] = []
    es = EmptyState(
        container,
        icon="📁",
        title="Aucune image scannée",
        subtitle="Sélectionnez un dossier source pour commencer.",
        action_label="Choisir un dossier",
        on_action=lambda: fired.append(True),
    )
    es.pack()
    app.update_idletasks()
    assert es.winfo_exists()
    container.destroy()


def _check_context_panel_standalone(app, ContextPanel) -> None:
    container = app  # use app root directly
    panel = ContextPanel(container)
    panel.grid(row=99, column=99)  # off-screen-ish; won't affect main layout
    app.update_idletasks()
    assert panel.is_open is False
    assert int(panel.cget("width")) == 0

    builder_called: list[bool] = []

    def builder(parent):
        import customtkinter as ctk_local

        ctk_local.CTkLabel(parent, text="Détails").grid(row=0, column=0)
        builder_called.append(True)

    panel.set_content("Détails", builder)
    panel.open()
    app.update_idletasks()
    assert panel.is_open is True
    assert int(panel.cget("width")) == panel.WIDTH
    assert builder_called == [True]

    panel.close()
    app.update_idletasks()
    assert panel.is_open is False
    panel.destroy()
