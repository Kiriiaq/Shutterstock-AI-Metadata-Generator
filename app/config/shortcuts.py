"""Global keyboard shortcuts.

Each entry is ``(binding, label_i18n_key, action_id)``. ``App`` resolves
``action_id`` to a real callback at bind time so the mapping stays
declarative and the help dialog is generated from the same source.

Both upper- and lower-case letter bindings are listed because Tk reports
the actual key state — explicit duplicates are simpler than xmodmap.
"""

from __future__ import annotations

from typing import Final

# (binding, label_i18n_key, action_id)
GLOBAL_SHORTCUTS: Final[list[tuple[str, str, str]]] = [
    ("<Control-k>", "help.shortcut.cmd_k", "open_command_palette"),
    ("<Control-K>", "help.shortcut.cmd_k", "open_command_palette"),
    ("<Control-b>", "help.shortcut.collapse_sidebar", "toggle_sidebar"),
    ("<Control-B>", "help.shortcut.collapse_sidebar", "toggle_sidebar"),
    ("<Control-Shift-T>", "help.shortcut.toggle_theme", "toggle_theme"),
    ("<Control-comma>", "help.shortcut.settings", "navigate_settings"),
    ("<Control-f>", "help.shortcut.search", "focus_view_search"),
    ("<Control-F>", "help.shortcut.search", "focus_view_search"),
    ("<Control-s>", "help.shortcut.save", "save_current_view"),
    ("<Control-S>", "help.shortcut.save", "save_current_view"),
    ("<F1>", "help.shortcut.help", "open_help"),
    ("<Control-slash>", "help.shortcut.help", "open_help"),
    ("<Escape>", "help.shortcut.escape", "close_modal"),
    ("<Alt-Left>", "help.shortcut.history_back", "history_back"),
    ("<Alt-Right>", "help.shortcut.history_forward", "history_forward"),
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
    s = s.replace("comma", ",").replace("slash", "/")

    if "+" in s:
        head, sep, last = s.rpartition("+")
        if len(last) == 1 and last.isalpha():
            return head + sep + last.upper()
        return s
    if len(s) == 1 and s.isalpha():
        return s.upper()
    return s
