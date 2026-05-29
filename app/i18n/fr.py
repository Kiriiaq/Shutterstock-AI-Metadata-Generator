"""Centralised French UI strings.

Every visible label, message, and tooltip is keyed in ``T``. Vues import
``t(key, **fmt)`` instead of writing literal strings — dropping a second
language later means swapping this module's import target.
"""

from __future__ import annotations

from src import __version__ as _VERSION

T: dict[str, str] = {
    # ============================== App ==============================
    "app.title": f"ShutterstockAnalyzer v{_VERSION} — Générateur de métadonnées IA",
    "app.topbar_title": f"ShutterstockAnalyzer v{_VERSION} — Atelier",
    "app.welcome_title": "Bienvenue",
    # NOTE Phase F : la sidebar a été supprimée — on remplace l'ancien
    # message "Sélectionnez une vue dans la barre latérale" par un
    # rappel aligné sur l'architecture Atelier actuelle (Ctrl+1..5).
    "app.welcome_body": "Utilisez Ctrl+1 à Ctrl+5 pour naviguer entre les panneaux et modales.",
    # ============================== Topbar ===========================
    "topbar.theme_toggle_tooltip": "Basculer le thème (Ctrl+Shift+T)",
    "topbar.help_tooltip": "Raccourcis clavier (F1)",
    # ============================== Confirm dialog ===================
    "dialog.confirm_default_title": "Confirmation requise",
    "dialog.ok": "Confirmer",
    "dialog.cancel": "Annuler",
    "dialog.delete": "Supprimer",
    "dialog.discard": "Abandonner",
    # ============================== Toasts ===========================
    "toast.success_default": "Opération réussie.",
    "toast.error_default": "Une erreur est survenue.",
    "toast.warning_default": "Attention.",
    "toast.info_default": "Information.",
    # ============================== Common ===========================
    "common.close": "Fermer",
    "common.cancel": "Annuler",
    "common.save": "Enregistrer",
    "common.next_step": "Étape suivante",
    "common.previous_step": "Étape précédente",
    # ============================== Help / shortcuts =================
    "help.title": "Raccourcis clavier",
    "help.shortcut.panel_sources": "Focus panneau Sources & tri",
    "help.shortcut.panel_editor": "Focus panneau Édition IPTC",
    "help.shortcut.panel_analyze": "Focus panneau Analyse IA",
    "help.shortcut.panel_validate": "Ouvrir la modale Validation",
    "help.shortcut.panel_history": "Ouvrir la modale Historique",
    "help.shortcut.toggle_theme": "Basculer thème clair / sombre",
    "help.shortcut.settings": "Ouvrir les paramètres",
    "help.shortcut.help": "Afficher cette aide",
    "help.shortcut.escape": "Fermer la modale active ou annuler le traitement",
    # ============================== Placeholders =====================
    "placeholder.under_construction": "Vue « {label} » en construction.",
    "placeholder.under_construction_body": "Cette section sera disponible dans une prochaine itération.",
}


def t(key: str, **fmt: object) -> str:
    """Look up *key* in ``T`` and ``str.format`` it with kwargs.

    Falls back to the key itself if missing — silent failure here is fine
    because the missing string is shown verbatim, easy to spot.
    """
    template = T.get(key, key)
    return template.format(**fmt) if fmt else template
