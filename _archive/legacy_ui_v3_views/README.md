# Legacy UI v3 views

Vues archivées le 2026-05-14 (audit Phase F, ticket D-02 / défaut additionnel n°2).

Ces 4 fichiers étaient présents dans `app/views/` mais n'étaient référencés nulle part :

| Fichier | Classe | Statut |
|---|---|---|
| `home.py` | `HomeView` | Dashboard d'accueil de l'ancienne UI v3 pré-dense. Remplacé par le `WorkspaceView` Atelier. Collision de `view_id="home"` avec WorkspaceView. |
| `editor.py` | `EditorView` | Vue d'édition IPTC plein écran. Le workflow actuel utilise le mini-éditeur intégré au panneau "Édition IPTC" du Workspace. |
| `sources.py` | `SourcesView` | Page de scan dossier plein écran. Remplacée par le panneau "Sources & tri" du Workspace. |
| `analyze.py` | `AnalyzeView` | Page d'analyse IA plein écran. Remplacée par le panneau "Analyse IA" du Workspace. |

Les 4 contenaient des `navigate_to("sources" | "analyze" | "settings")` qui pointaient vers des `view_id` désinscrits du Router — appel silencieusement loggé en `warning`.

Ces vues n'avaient aucune autre dépendance interne, donc le déplacement n'a cassé aucun import. Le test `tests/ui/test_app_v3_shell.py` ne les référence pas.
