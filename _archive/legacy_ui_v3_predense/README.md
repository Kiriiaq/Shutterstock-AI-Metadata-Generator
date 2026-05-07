# UI v3 — composants pré-Atelier-Dense (archivés)

Ces 4 composants ont vécu pendant les itérations v3 :

- `sidebar.py` — barre latérale 9 entrées, retirée par Phase 8 (l'utilisateur a demandé "pas de duplication" entre nav sidebar / SystemPanel quick-actions / Command Palette).
- `system_panel.py` — panneau droit avec status rows + quick-actions, fusionné dans les 5 sous-panneaux de droite du Workspace dense.
- `command_palette.py` — modale `Ctrl+K`, retirée car elle n'avait plus de cible de navigation.
- `context_panel.py` — panneau droit slide-in pour détails, remplacé par `App.show_details()` (Toplevel direct).

Aucun n'est importé par le code actif (`app/` ou `src/`). Conservés pour
référence et restauration éventuelle.
