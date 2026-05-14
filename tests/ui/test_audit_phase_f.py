"""Régressions de l'audit Phase F (2026-05-14) — tests sans root Tk.

Tkinter / customtkinter n'autorise qu'un seul root ``CTk`` par
processus, et l'instance est consommée par
``tests/ui/test_app_v3_shell.py::test_ui_v3_full_lifecycle``. Les
vérifications qui exigent un ``App`` ont donc été intégrées dans ce
test composite.

Ce module ne couvre que les vérifications STATIQUES qui ne nécessitent
pas d'instancier l'``App`` — Ctrl+1..5 dans le registre + cohérence
des libellés.
"""

from __future__ import annotations

CTRL_1_5_ACTIONS = {
    "focus_panel_sources",
    "focus_panel_editor",
    "focus_panel_analyze",
    "open_validate",
    "open_history",
}


def test_ctrl_1_to_5_registered_in_registry():
    """T-016..T-020 : Ctrl+1..5 doivent être dans le registre et chacun
    doit pointer vers un action_id distinct."""
    from app.config.shortcuts import GLOBAL_SHORTCUTS

    bindings = {b: a for b, _, a in GLOBAL_SHORTCUTS}
    for i in range(1, 6):
        key = f"<Control-Key-{i}>"
        assert key in bindings, f"Ctrl+{i} non enregistré"

    assigned = {bindings[f"<Control-Key-{i}>"] for i in range(1, 6)}
    assert assigned == CTRL_1_5_ACTIONS


def test_no_residual_no_op_shortcuts():
    """Lot A — nettoyage : les action_ids "lambda: None" historiques
    (palette/sidebar/search/save/history) ne doivent plus apparaître
    dans le registre. Ils polluaient l'aide F1."""
    from app.config.shortcuts import GLOBAL_SHORTCUTS

    no_ops = {
        "open_command_palette",
        "toggle_sidebar",
        "focus_view_search",
        "save_current_view",
        "history_back",
        "history_forward",
    }
    assigned = {a for _, _, a in GLOBAL_SHORTCUTS}
    leftovers = assigned & no_ops
    assert not leftovers, f"Raccourcis no-op encore présents : {leftovers}"


def test_shortcut_display_labels_match_expected_format():
    """display_label(<Control-Key-1>) doit produire "Ctrl+1" (l'aide F1
    affiche cette chaîne)."""
    from app.config.shortcuts import display_label

    assert display_label("<Control-Key-1>") == "Ctrl+1"
    assert display_label("<Control-Key-5>") == "Ctrl+5"
    assert display_label("<Escape>") == "Escape"
    assert display_label("<F1>") == "F1"
    assert display_label("<Control-Shift-T>") == "Ctrl+Shift+T"
    assert display_label("<Control-comma>") == "Ctrl+,"
    assert display_label("<Control-slash>") == "Ctrl+/"
