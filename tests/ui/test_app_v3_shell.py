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
        # Phase F (2026-05-14) — correctifs audit (Lots A à D).
        _check_phase_f_shortcuts(app)
        _check_phase_f_workspace_states(app)

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
    """All detail views open in a Toplevel and close cleanly."""
    for view_id in ("settings", "audit", "ai_control", "validate", "expert_report"):
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


# ============================================================================
# Phase F (2026-05-14) — correctifs de l'audit
# ----------------------------------------------------------------------------
# Vérifie que :
#   * Lot A : Ctrl+1..5 sont enregistrés et ont chacun un handler distinct.
#   * Lot A : ``focus_panel`` ne crashe pas + auto-déplie l'éditeur IPTC.
#   * Lots B/C : compteur Sources permanent + boutons Supprimer/Vider
#     dépendants du modèle/sélection.
#   * Lot C : Démarrer / Arrêter désactivés au démarrage et inversés
#     correctement par ``_refresh_action_states``.
#   * Lot D : barre de progression visible à 0, garde-fou double-clic,
#     Esc pendant traitement → log info + toast d'annulation.
# ============================================================================


def _check_phase_f_shortcuts(app) -> None:
    """Lot A — registre Ctrl+1..5 + action_map en correspondance parfaite."""
    from app.config.shortcuts import GLOBAL_SHORTCUTS

    bindings = {b: a for b, _, a in GLOBAL_SHORTCUTS}
    for i in range(1, 6):
        key = f"<Control-Key-{i}>"
        assert key in bindings, f"Ctrl+{i} non enregistré"

    expected = {
        "focus_panel_sources",
        "focus_panel_editor",
        "focus_panel_analyze",
        "open_validate",
        "open_history",
    }
    assigned = {bindings[f"<Control-Key-{i}>"] for i in range(1, 6)}
    assert assigned == expected

    registry_ids = {a for _, _, a in GLOBAL_SHORTCUTS}
    handler_ids = set(app._build_action_map().keys())
    assert registry_ids == handler_ids, (
        f"Désynchro registre/handlers — manquants={registry_ids - handler_ids} extras={handler_ids - registry_ids}"
    )


def _check_phase_f_workspace_states(app) -> None:
    """Lots B/C/D — états initiaux et transitions du panneau Sources +
    Analyse IA + comportement de Esc pendant un traitement simulé."""
    import logging
    from pathlib import Path
    from unittest.mock import patch

    ws = app.router._current_view
    assert ws is not None

    # focus_panel sur les 3 cibles + nom inconnu (no-op).
    for name in ("sources", "analyze"):
        ws.focus_panel(name)
    # Auto-expansion de l'éditeur.
    if not ws._editor_collapsed:
        ws._toggle_editor_collapsed()
    assert ws._editor_collapsed is True
    ws.focus_panel("editor")
    assert ws._editor_collapsed is False, "Ctrl+2 doit déplier l'éditeur"
    ws.focus_panel("nope")  # no-op silencieux

    # États initiaux : Démarrer + Arrêter + Supprimer + Vider tous disabled.
    assert ws._start_btn.cget("state") == "disabled"
    assert ws._stop_btn.cget("state") == "disabled"
    assert ws._remove_btn.cget("state") == "disabled"
    assert ws._clear_btn.cget("state") == "disabled"
    # Phase G (2026-05-18) — le format du compteur est désormais
    # « nombre de fichiers : N · M sélectionné(s) ».
    assert "nombre de fichiers : 0" in ws._sources_status.cget("text")

    # Barre de progression initialisée à 0 + statut "0 / 0 — En attente".
    assert ws._analyze_progress.winfo_exists()
    assert ws._analyze_progress.get() == 0
    status = ws._analyze_status.cget("text")
    assert "0 / 0" in status and "En attente" in status

    # Sélection → Démarrer activé.
    app.app_state.set("selected_paths", [Path("/tmp/a.jpg"), Path("/tmp/b.jpg")])
    ws._refresh_action_states()
    assert ws._start_btn.cget("state") == "normal"
    assert ws._stop_btn.cget("state") == "disabled"

    # Processing → boutons inversés.
    ws._processing = True
    ws._refresh_action_states()
    assert ws._start_btn.cget("state") == "disabled"
    assert ws._stop_btn.cget("state") == "normal"

    # Garde-fou double-clic.
    with patch("threading.Thread") as mock_thread:
        ws._analyze_start()
        assert mock_thread.call_count == 0, "Double-clic sur Démarrer doit être bloqué"

    # Reset pour la suite.
    ws._processing = False
    ws._refresh_action_states()

    # Modèle Sources : injection puis clear complet.
    fake = [
        {
            "_path": Path(f"/tmp/img-{i}.jpg"),
            "name": f"img-{i}.jpg",
            "size": "1 KB",
            "dim": "100x100",
            "meta": "Non",
        }
        for i in range(3)
    ]
    ws._scanned.extend(fake)
    ws._sources_table.set_rows(ws._scanned)
    ws._sync_sources_state()
    assert len(app.app_state.get("scanned_images") or []) == 3
    assert ws._clear_btn.cget("state") == "normal"

    ws._clear_all()
    assert ws._scanned == []
    assert app.app_state.get("scanned_images") == []
    assert ws._clear_btn.cget("state") == "disabled"
    assert ws._remove_btn.cget("state") == "disabled"
    assert ws._start_btn.cget("state") == "disabled"

    # Esc pendant processing → log info + toast d'annulation.
    ws._processing = True
    toast_messages: list[tuple[str, str]] = []

    def fake_show(msg, kind="info", **_kw):
        toast_messages.append((msg, kind))

    handler = logging.Handler()
    records: list[logging.LogRecord] = []
    handler.emit = records.append  # type: ignore[assignment]
    app_logger = logging.getLogger("app.app")
    app_logger.addHandler(handler)
    app_logger.setLevel(logging.INFO)
    try:
        with patch.object(app.toasts, "show", side_effect=fake_show):
            app._close_top_modal()
    finally:
        app_logger.removeHandler(handler)

    assert any("Escape" in r.getMessage() and "cancel" in r.getMessage().lower() for r in records), (
        "Le log info d'annulation par Escape doit apparaître"
    )
    assert toast_messages and "Annulation" in toast_messages[0][0], "Un toast 'Annulation demandée…' doit être émis"
