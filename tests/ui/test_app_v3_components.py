"""Smoke tests for the Phase 4 components (palette, table, form, empty, panel).

Each test instantiates the component in isolation against a hidden Tk
root, exercises its public API, and tears down. Real interaction (mouse
clicks, theme switches) is covered by the higher-level ``test_app_v3_shell``
suite — here we want to catch construction-level breakage early.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def root():
    pytest.importorskip("customtkinter")
    import customtkinter as ctk

    r = ctk.CTk()
    r.withdraw()
    yield r
    try:
        r.destroy()
    except Exception:
        pass


def test_command_palette_filters_and_executes(root):
    from app.components.command_palette import Command, CommandPalette

    fired: list[str] = []
    cmds = [
        Command(id="a", label="Aller à : Sources", callback=lambda: fired.append("a")),
        Command(id="b", label="Basculer le thème", callback=lambda: fired.append("b")),
        Command(id="c", label="Vue précédente", callback=lambda: fired.append("c"), keywords=("history",)),
    ]
    palette = CommandPalette(root, provider=lambda: cmds)
    palette.open()
    root.update_idletasks()
    assert palette._row_widgets and len(palette._row_widgets) == 3

    # Substring filter
    palette._refresh_results("thème")
    assert len(palette._row_widgets) == 1
    assert palette._row_widgets[0][1].id == "b"

    # Token filter from keywords
    palette._refresh_results("history")
    assert len(palette._row_widgets) == 1 and palette._row_widgets[0][1].id == "c"

    # Execute the highlighted entry
    palette._highlighted = 0
    palette._execute_highlighted()
    assert fired == ["c"]
    assert palette._win is None  # closed after execute


def test_data_table_set_sort_select(root):
    import customtkinter as ctk

    from app.components.data_table import Column, DataTable

    container = ctk.CTkFrame(root)
    container.pack()
    table = DataTable(
        container,
        columns=[
            Column(id="name", label="Nom", width=120),
            Column(id="size", label="Taille", width=80, sort_key=int),
        ],
    )
    table.pack(fill="both", expand=True)
    root.update_idletasks()

    table.set_rows(
        [
            {"name": "b.jpg", "size": 200},
            {"name": "a.jpg", "size": 50},
            {"name": "c.jpg", "size": 100},
        ]
    )
    root.update_idletasks()
    assert table._tree.get_children()
    assert len(table._tree.get_children()) == 3

    # Sort ascending then descending by size
    table._sort_by("size")
    rows = [table._row_data[iid]["size"] for iid in table._tree.get_children()]
    assert rows == sorted(rows)
    table._sort_by("size")
    rows_desc = [table._row_data[iid]["size"] for iid in table._tree.get_children()]
    assert rows_desc == sorted(rows_desc, reverse=True)

    # Selection round-trip
    first_iid = table._tree.get_children()[0]
    table._tree.selection_set(first_iid)
    selected = table.get_selected()
    assert len(selected) == 1
    assert selected[0]["size"] == max(
        r["size"]
        for r in [
            {"size": 50},
            {"size": 100},
            {"size": 200},
        ]
    )

    # Theme refresh must not raise
    table.refresh_theme()


def test_form_field_validate_and_error_display(root):
    import customtkinter as ctk

    from app.components.form_field import FormField, entry_factory

    def required_validator(v):
        return None if v else "Ce champ est requis."

    field = FormField(
        ctk.CTkFrame(root),
        label="Nom",
        required=True,
        widget_factory=entry_factory,
        validator=required_validator,
    )
    field.pack()
    root.update_idletasks()

    # Empty value fails
    assert field.validate() is False
    assert field._error_message == "Ce champ est requis."

    # Filled value passes
    field.set_value("Marc")
    assert field.value == "Marc"
    assert field.validate() is True
    assert field._error_message is None

    # Manual error
    field.set_error("Erreur custom")
    assert field._error_message == "Erreur custom"
    field.set_error(None)
    assert field._error_message is None


def test_empty_state_with_action(root):
    import customtkinter as ctk

    from app.components.empty_state import EmptyState

    fired = []
    es = EmptyState(
        ctk.CTkFrame(root),
        icon="📁",
        title="Aucune image scannée",
        subtitle="Sélectionnez un dossier source pour commencer.",
        action_label="Choisir un dossier",
        on_action=lambda: fired.append(True),
    )
    es.pack()
    root.update_idletasks()
    assert es.winfo_exists()


def test_context_panel_open_close(root):
    from app.components.context_panel import ContextPanel

    panel = ContextPanel(root)
    panel.grid(row=0, column=0)
    root.update_idletasks()
    assert panel.is_open is False
    assert int(panel.cget("width")) == 0

    builder_called = []

    def builder(parent):
        import customtkinter as ctk

        ctk.CTkLabel(parent, text="Détails").grid(row=0, column=0)
        builder_called.append(True)

    panel.set_content("Détails", builder)
    panel.open()
    root.update_idletasks()
    assert panel.is_open is True
    assert int(panel.cget("width")) == panel.WIDTH
    assert builder_called == [True]

    panel.close()
    root.update_idletasks()
    assert panel.is_open is False
    assert int(panel.cget("width")) == 0


def test_app_wires_palette_and_context_panel(tmp_path, monkeypatch):
    pytest.importorskip("customtkinter")
    from app.config import theme as theme_mod

    monkeypatch.setattr(theme_mod, "get_prefs_path", lambda: tmp_path / "ui_prefs.json")

    from app.app import App
    from app.components.command_palette import CommandPalette

    app = App(api=None)
    try:
        app.update_idletasks()

        # Palette is lazily built by _open_command_palette
        app._open_command_palette()
        app.update_idletasks()
        assert isinstance(app._palette, CommandPalette)
        # Provider returns at least the 9 nav commands + 5 globals
        cmds = app._build_commands()
        assert len(cmds) >= 9 + 5
        ids = {c.id for c in cmds}
        assert "nav.home" in ids and "toggle_theme" in ids

        # ContextPanel exists and is closed
        assert app.context_panel.is_open is False
        app.context_panel.set_content(
            "Test",
            lambda parent: None,
        )
        app.context_panel.open()
        app.update_idletasks()
        assert app.context_panel.is_open is True
        app.context_panel.close()
        app.update_idletasks()
        assert app.context_panel.is_open is False

        # Theme toggle now also refreshes the context panel — must not raise
        app._toggle_theme()
        app._toggle_theme()
    finally:
        try:
            app.destroy()
        except Exception:
            pass
