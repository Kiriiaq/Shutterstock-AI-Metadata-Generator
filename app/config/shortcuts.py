"""Global keyboard shortcuts.

Each entry is ``(binding, label_i18n_key, action_id)``. ``App`` resolves
``action_id`` to a real callback at bind time so the mapping stays
declarative and the help dialog is generated from the same source.

Both upper- and lower-case letter bindings are listed because Tk reports
the actual key state — explicit duplicates are simpler than xmodmap.

Phase F (2026-05-14) — registre nettoyé :
* Retrait des entrées no-op (palette de commandes, sidebar toggle,
  search, save view, history back/forward) qui pointaient toutes vers
  des handlers ``lambda: None`` depuis la suppression de la sidebar
  (Phase 8). Elles polluaient l'aide F1 en promettant des raccourcis
  sans effet.
* Ajout de Ctrl+1..5 pour la navigation rapide entre panneaux /
  modales détail, suite à la campagne de tests T-016..T-020.
  Mapping :
    Ctrl+1 → focus panneau Sources & tri
    Ctrl+2 → focus panneau Édition IPTC (ouvre si collapsed)
    Ctrl+3 → focus panneau Analyse IA
    Ctrl+4 → ouvre la modale Validation
    Ctrl+5 → ouvre la modale Historique
"""

from __future__ import annotations

from typing import Final

# (binding, label_i18n_key, action_id)
GLOBAL_SHORTCUTS: Final[list[tuple[str, str, str]]] = [
    ("<Control-Key-1>", "help.shortcut.panel_sources", "focus_panel_sources"),
    ("<Control-Key-2>", "help.shortcut.panel_editor", "focus_panel_editor"),
    ("<Control-Key-3>", "help.shortcut.panel_analyze", "focus_panel_analyze"),
    ("<Control-Key-4>", "help.shortcut.panel_validate", "open_validate"),
    ("<Control-Key-5>", "help.shortcut.panel_history", "open_history"),
    ("<Control-Shift-T>", "help.shortcut.toggle_theme", "toggle_theme"),
    ("<Control-comma>", "help.shortcut.settings", "navigate_settings"),
    ("<F1>", "help.shortcut.help", "open_help"),
    ("<Control-slash>", "help.shortcut.help", "open_help"),
    ("<Escape>", "help.shortcut.escape", "close_modal"),
]


def display_label(binding: str) -> str:
    """``"<Control-k>"`` → ``"Ctrl+K"`` for the help dialog.

    Capitalises the last token only when it is a single letter (so ``Ctrl+k``
    becomes ``Ctrl+K`` but ``Alt+Left`` stays ``Alt+Left``).
    """
    s = binding.strip("<>")
    s = s.replace("Control", "Ctrl")
    s = s.replace("Shift-", "Shift+")
    s = s.replace("Alt-", "Alt+")
    s = s.replace("Ctrl-", "Ctrl+")
    s = s.replace("Key-", "")
    s = s.replace("comma", ",").replace("slash", "/")

    if "+" in s:
        head, sep, last = s.rpartition("+")
        if len(last) == 1 and last.isalpha():
            return head + sep + last.upper()
        return s
    if len(s) == 1 and s.isalpha():
        return s.upper()
    return s
