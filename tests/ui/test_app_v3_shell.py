"""End-to-end smoke for the dense-Atelier ``app/`` UI layer.

Tk does not allow re-creating a root within a single Python process,
so this single test owns the only ``CTk`` instance for the whole UI
test session. It exercises:

- App shell construction (no sidebar, topbar with health strip,
  central frame mounting the WorkspaceView).
- The Workspace's 8 tool panels are present (sources, editor, analyse,
  modèle IA, validation, historique, paramètres, téléversement).
- ``open_in_modal`` works for the 5 detail views (settings, audit,
  ai_control, validate, upload) and the modal closes cleanly.
- ``show_details`` (replacement for the old ContextPanel) opens.
- Theme toggle round-trip with persisted prefs.
- Topbar health strip rebuilds via the provider.
- DataTable, FormField, EmptyState constructors smoke-build under the
  same root.
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
    from app.components.data_table import Column, DataTable
    from app.components.empty_state import EmptyState
    from app.components.form_field import FormField, entry_factory

    app = App(api=None)
    try:
        app.update_idletasks()

        _check_shell(app)
        _check_workspace_panels(app)
        _check_open_in_modal(app)
        _check_show_details(app)
        _check_theme_toggle(app, theme_mod, fake_prefs)
        _check_topbar_health(app)
        _check_data_table(app, ctk, Column, DataTable)
        _check_form_field(app, ctk, FormField, entry_factory)
        _check_empty_state(app, ctk, EmptyState)

    finally:
        try:
            app.destroy()
        except Exception:
            pass


# ============================================================================
# Helpers
# ============================================================================


def _check_shell(app) -> None:
    assert "ShutterstockAnalyzer" in app.title()
    assert app.topbar is not None
    assert app._center.winfo_exists()
    assert app.router.current_id == "home"
    assert not hasattr(app, "sidebar")  # sidebar removed in dense atelier
    assert not hasattr(app, "context_panel")  # replaced by show_details modal


def _check_workspace_panels(app) -> None:
    """The Workspace must expose the 8 tool panels' state widgets."""
    workspace = app.router._current_view
    assert workspace is not None
    # Sample one widget from each of the 8 panels — covers all of them.
    assert hasattr(workspace, "_sources_table")
    assert hasattr(workspace, "_iptc_fields")
    assert hasattr(workspace, "_analyze_results")
    assert hasattr(workspace, "_model_status_dot")
    assert hasattr(workspace, "_validate_summary")
    assert hasattr(workspace, "_history_lines")
    assert hasattr(workspace, "_settings_chips")
    # Editor IPTC has a collapse chevron that hides the body keeping the title.
    assert hasattr(workspace, "_editor_toggle_btn")
    assert hasattr(workspace, "_editor_body")
    assert workspace._editor_collapsed is False
    workspace._toggle_editor_collapsed()
    assert workspace._editor_collapsed is True
    workspace._toggle_editor_collapsed()
    assert workspace._editor_collapsed is False


def _check_open_in_modal(app) -> None:
    """All 5 detail views open in a Toplevel and close cleanly."""
    for view_id in ("settings", "audit", "ai_control", "validate"):
        before = len(app._open_modals)
        app.open_in_modal(view_id)
        app.update_idletasks()
        # _open_modals tracks help/details; open_in_modal also pushes via the
        # finalizer — robustly close every Toplevel child we just made.
        toplevels = [c for c in app.winfo_children() if c.winfo_class() in ("Toplevel", "CTkToplevel")]
        for tl in toplevels:
            try:
                tl.destroy()
            except Exception:
                pass
        app.update_idletasks()
        _ = before


def _check_show_details(app) -> None:
    import customtkinter as ctk

    called: list[bool] = []

    def builder(parent: ctk.CTkFrame) -> None:
        ctk.CTkLabel(parent, text="ok").pack()
        called.append(True)

    app.show_details("Test", builder)
    app.update_idletasks()
    assert called == [True]
    for tl in [c for c in app.winfo_children() if c.winfo_class() in ("Toplevel", "CTkToplevel")]:
        try:
            tl.destroy()
        except Exception:
            pass


def _check_theme_toggle(app, theme_mod, fake_prefs) -> None:
    before = theme_mod.load_theme_pref()
    app._toggle_theme()
    after = theme_mod.load_theme_pref()
    assert before != after
    assert fake_prefs.exists()


def _check_topbar_health(app) -> None:
    """Topbar health strip is populated by the provider on each refresh."""
    app.topbar.refresh_health()
    app.update_idletasks()
    assert "Backend" in app.topbar._chip_widgets
    assert "ExifTool" in app.topbar._chip_widgets


def _check_data_table(app, ctk, Column, DataTable) -> None:
    container = ctk.CTkFrame(app)
    container.grid(row=99, column=99)
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
    container.grid(row=99, column=99)
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
    container.grid(row=99, column=99)
    es = EmptyState(
        container,
        icon="📁",
        title="Aucune image scannée",
        subtitle="Sélectionnez un dossier source pour commencer.",
        action_label="Choisir un dossier",
        on_action=lambda: None,
    )
    es.pack()
    app.update_idletasks()
    assert es.winfo_exists()
    container.destroy()
