"""Centralised French UI strings.

Every visible label, message, and tooltip is keyed in ``T``. Vues import
``t(key, **fmt)`` instead of writing literal strings — dropping a second
language later means swapping this module's import target.
"""

from __future__ import annotations

T: dict[str, str] = {
    # ============================== App ==============================
    "app.title": "ShutterstockAnalyzer v2.0.0 — Générateur de métadonnées IA",
    "app.welcome_title": "Bienvenue",
    "app.welcome_body": "Sélectionnez une vue dans la barre latérale ou utilisez Ctrl+K.",
    # ============================== Sidebar ==========================
    "nav.section.production": "Production",
    "nav.section.system": "Pilotage et système",
    "nav.home": "Atelier",
    "nav.sources": "Sources et tri",
    "nav.analyze": "Analyse IA",
    "nav.editor": "Édition métadonnées",
    "nav.validate": "Validation",
    "nav.ai_control": "Modèle IA",
    "nav.audit": "Historique",
    "nav.settings": "Paramètres",
    "nav.collapse_tooltip": "Replier la barre (Ctrl+B)",
    "nav.expand_tooltip": "Déplier la barre (Ctrl+B)",
    # ============================== Topbar ===========================
    "topbar.search_placeholder": "Rechercher… (Ctrl+K)",
    "topbar.theme_toggle_tooltip": "Basculer le thème (Ctrl+Shift+T)",
    "topbar.help_tooltip": "Raccourcis clavier (F1)",
    "topbar.profile_tooltip": "Profil et préférences",
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
    "help.shortcut.cmd_k": "Palette de commandes",
    "help.shortcut.collapse_sidebar": "Replier / déplier la barre latérale",
    "help.shortcut.toggle_theme": "Basculer thème clair / sombre",
    "help.shortcut.settings": "Ouvrir les paramètres",
    "help.shortcut.search": "Rechercher dans la vue active",
    "help.shortcut.save": "Enregistrer",
    "help.shortcut.help": "Afficher cette aide",
    "help.shortcut.escape": "Fermer la modale active",
    "help.shortcut.history_back": "Vue précédente",
    "help.shortcut.history_forward": "Vue suivante",
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
