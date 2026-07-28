# Architecture IHM — StockMeta Pro (UI v3)

**Statut** : implémenté et actif (commit `386741c` + Phase 6).
**Stack** : Python 3.11+, customtkinter, tkinter/ttk, Pillow (icônes seulement), stdlib.
**Langue interface** : 100 % français, vouvoiement / impersonnel.
**Entrée** : `python main.py` ou `python -m app.main`. EXE produit par
`python build.py release` (`dist/StockMetaPro.exe`).

## 1. Philosophie

Trois principes :

1. **Workflow linéaire visible**. L'utilisateur produit un livrable Shutterstock en suivant 5 étapes : `Sources → Analyse IA → Édition → Validation → Téléversement`. Chaque étape est une vue dédiée, avec un bouton « Étape suivante » qui pré-remplit la vue suivante.
2. **Pilotage et configuration séparés**. Le pilotage du modèle IA, l'historique des actions et les paramètres ne sont pas dans le workflow : ils vivent en bas de la sidebar pour rester accessibles sans encombrer la production.
3. **Aucune décoration gratuite**. Une icône Unicode + un libellé. Pas de gradients, pas d'ombres lourdes, un seul accent (bleu). Densité élevée mais pas serrée — les espacements suivent l'échelle 4/8/12/16/24/32/48.

## 2. Layout shell (3 zones, `grid`)

```
┌────────────────────────────────────────────────────────────────────────┐
│ [Topbar 44px]  Fil d'Ariane │ Recherche / Ctrl+K │ Thème · Profil      │
├──────────┬─────────────────────────────────────────────────────┬───────┤
│          │                                                     │       │
│ Sidebar  │                                                     │ Pan.  │
│ 220→56px │              Zone centrale (vue active)             │ ctx.  │
│          │                                                     │ 320px │
│ (replia- │                                                     │ (ouv. │
│  ble)    │                                                     │ à la  │
│          │                                                     │ dem.) │
└──────────┴─────────────────────────────────────────────────────┴───────┘
```

- **Sidebar** : navigation par domaine, repliable via `Ctrl+B`. En mode replié (56 px) seules les icônes sont visibles, le libellé apparaît en tooltip.
- **Topbar** : fil d'Ariane à gauche, champ de recherche déclencheur de Command Palette au centre, à droite un toggle thème + un bouton info utilisateur (qui ouvre l'aide raccourcis).
- **Zone centrale** : la vue active. La vue gère son propre scroll si nécessaire (`CTkScrollableFrame`).
- **Panneau contextuel droit** : `CTkFrame` glissant, optionnel, ouvert via un bouton dans la topbar de la vue. Cas d'usage : prévisualisation d'image, détails d'un log, formulaire d'édition rapide.

## 3. Arborescence fonctionnelle (sidebar)

Sept entrées, regroupées en deux blocs séparés visuellement par un séparateur.

### Bloc « Production » (workflow linéaire)

| ID | Libellé sidebar | Icône | Vue (`app/views/`) | Rôle |
|---|---|---|---|---|
| `home` | Tableau de bord | ⌂ | `home.py` | Résumé : dossier source en cours, dernier batch, état Ollama, raccourcis |
| `sources` | Sources et tri | 📁 | `sources_view.py` | Choix dossier, scan récursif, tri par taille/résolution/métadonnées, sélection multiple |
| `analyze` | Analyse IA | 🧠 | `analyze_view.py` | Lancement batch IA sur sélection, suivi progression, résultats temps réel |
| `editor` | Édition métadonnées | ✎ | `editor_view.py` | Lecture/édition IPTC + XMP par image, application templates, preview |
| `validate` | Validation | ✔ | `validate_view.py` | Checklist pré-export : conformité IPTC, dimensions, doublons, résolution |
| `upload` | Téléversement FTPS | ↑ | `upload_view.py` | Connexion FTPS Shutterstock, transfert par batch de 50, suivi débit |

### Bloc « Pilotage et système »

| ID | Libellé sidebar | Icône | Vue | Rôle |
|---|---|---|---|---|
| `ai_control` | Modèle IA | ⚙ | `ai_control_view.py` | Connexion Ollama, sélection modèle vision, test, monitoring GPU/VRAM |
| `audit` | Historique | 📜 | `audit_view.py` | Treeview filtré par action/date/statut, export JSON/CSV, détails |
| `settings` | Paramètres | ⚙ | `settings_view.py` | Onglets internes : Général, Traitement, Métadonnées, FTPS, Avancé |

### Aide (toujours accessible)

- `F1` ou `Ctrl+/` : modale `HelpView` listant tous les raccourcis. N'apparaît pas dans la sidebar.

## 4. Workflow utilisateur principal

```
home ──► sources ──► analyze ──► editor ──► validate ──► upload
                       ▲          ▲           │            │
                       └──────────┴───────────┘            │
                          (corrections cycliques)          │
                                                           ▼
                                            (batch suivant ou home)
```

- Chaque vue de production expose un bouton « Étape suivante » dans son footer.
- Le bouton n'est actif que si la vue est dans un état complet (ex : `sources` active « Étape suivante » uniquement quand au moins une image est sélectionnée).
- Le retour est libre : la sidebar permet de revenir à n'importe quelle vue à tout moment.

## 5. Mappage avec le backend existant

Les vues n'instancient jamais directement les classes de `src/modules/`. Elles passent toutes par la façade `ShutterstockAIv2` (instanciée une fois dans `app.app.App`) :

| Vue | Méthodes appelées |
|---|---|
| `home` | `api.get_statistics()`, `api.check_ai_status()`, `api.exiftool_available` |
| `sources` | `api.metadata_reader.read_quick_info`, `worker_pool.collect_image_files` |
| `analyze` | `api.init_ai()`, `api.analyze_batch_ai(...)`, `api.vision_analyzer.cancel()` |
| `editor` | `api.read_metadata`, `api.write_metadata`, `api.apply_template`, `api.iptc_engine` |
| `validate` | `api.validate_image`, `api.validate_shutterstock_metadata` |
| `upload` | (à câbler quand FTPS implémenté ; stub désactivé d'ici là) |
| `ai_control` | `api.ollama_client.*`, `api.check_ai_status()` |
| `audit` | `api.database.get_audit_logs`, `api.database.export_audit_log` |
| `settings` | `api.get_setting`, `api.set_setting`, `api.iptc_engine.list_templates` |

## 6. Routing et navigation

- `app.core.navigation.Router` maintient `current_view_id`, une pile `history` et un index `cursor`.
- `router.navigate_to(view_id, **kwargs)` détruit la vue précédente, instancie la nouvelle dans le conteneur central, met à jour la sidebar (état actif) et la topbar (fil d'Ariane).
- `Alt+←` / `Alt+→` rejouent l'historique.
- Pas de deep-linking URL — pas pertinent pour une app desktop offline.
- Les vues exposent une méthode `on_enter(**kwargs)` (chargement données) et `on_leave()` (cleanup, sauvegarde brouillon).

## 7. État applicatif

`app.core.state.AppState` est un singleton observable simple (pattern *signal* maison via `app.core.events.EventBus`).

État partagé minimal :

```
AppState
├─ source_folder: Path | None
├─ scanned_images: list[ImageItem]   (après sources)
├─ selected_paths: list[Path]        (sélection courante)
├─ current_batch_id: str | None      (analyze en cours)
├─ ai_status: OllamaStatus
└─ exiftool_available: bool
```

Les vues s'abonnent aux changements (`state.on("selected_paths", callback)`) et la sidebar peut afficher des compteurs (« 12 sélectionnées »).

## 8. Bi-thème

- `customtkinter.set_appearance_mode("light"|"dark"|"system")` natif.
- `app.config.theme.LIGHT` et `DARK` exposent toutes les couleurs sémantiques.
- `get_color(name)` lit `ctk.get_appearance_mode()` à chaque appel — la bascule est instantanée.
- Le `ttk.Style` du Treeview (DataTable) s'auto-resync via une callback `on_theme_change` posée par le composant.
- Préférence persistée dans `%APPDATA%/ShutterstockAnalyzer/ui_prefs.json`
  (nom de dossier historique conservé au renommage v2.4.0, pour ne pas
  orpheliner les préférences existantes) (Windows) ou `~/.shutterstock_analyzer/ui_prefs.json` (autres).
- `Ctrl+Shift+T` bascule clair ↔ sombre (ne touche pas au mode `system`).

## 9. Composants réutilisables (`app/components/`)

| Fichier | Rôle | Notes |
|---|---|---|
| `sidebar.py` | Navigation gauche | Sections, icônes Unicode, état actif, repli, compteurs dynamiques |
| `topbar.py` | Barre supérieure | Fil d'Ariane, recherche, toggle thème, profil |
| `tooltip.py` | Tooltip | `Toplevel` sans bordure, déclenchement `<Enter>` 500 ms, hide `<Leave>` |
| `command_palette.py` | Palette `Ctrl+K` | `CTkToplevel` modale, `CTkEntry` + `CTkScrollableFrame` filtré |
| `data_table.py` | Tableau dense | Wrapper `ttk.Treeview` stylé, tri colonne, sélection multi, alternance |
| `form_field.py` | Champ formulaire | Label + input + zone d'erreur + astérisque requis |
| `empty_state.py` | État vide | Icône + titre + sous-titre + bouton d'action |
| `confirm_dialog.py` | Confirmation | Modale destructive (rouge) ou neutre (accent) |
| `toast.py` | Notification éphémère | Bas-droite, 4 s, success/error/info |
| `context_panel.py` | Panneau contextuel | Slide-in droite 320 px, fermable |

Chaque composant : navigable au clavier, focus visible, types annotés, docstring publique.

## 10. Raccourcis clavier (`app/config/shortcuts.py`)

```
Ctrl+K            Command Palette
Ctrl+B            Replier/déplier sidebar
Ctrl+Shift+T      Basculer thème clair/sombre
Ctrl+,            Ouvrir Paramètres
Ctrl+F            Rechercher dans la vue active
Ctrl+S            Enregistrer (selon vue)
Ctrl+N            Nouvelle entrée (selon vue)
F1 ou Ctrl+/      Modale d'aide
Échap             Fermer modale active
Alt+←  /  Alt+→   Historique navigation
```

Ces raccourcis sont posés via `bind_all` au démarrage de `App`. Chaque vue peut ajouter ses propres raccourcis contextuels, qui sont retirés en `on_leave()`.

## 11. Conventions de code

- Type hints partout. Docstrings sur classes et méthodes publiques.
- Fonctions ≤ 50 lignes, classes ≤ 300 lignes (sauf `App` shell, justifié).
- `logging` stdlib (`logger = logging.getLogger(__name__)`) — jamais de `print` en production.
- Tout texte affiché passe par `app.i18n.fr.T` (dict ou fonction `t(key)`), aucune chaîne en dur dans les vues.
- Numéros et dates : `app.utils.formatters.fmt_int(n)` (espace insécable), `fmt_date(dt)` (`JJ/MM/AAAA`).
- Pas de `pack` dans la même fenêtre que `grid` — `grid` partout.

## 12. Évolutions futures non bloquantes

- **i18n** : passer de `i18n/fr.py` à `gettext` quand une 2e langue arrive.
- **Tests UI** : étendre `tests/ui/` avec un test par vue (instanciation headless + assertions sur les widgets exposés).
- **Préférences avancées** : raccourcis personnalisables (édition de `shortcuts.py` à chaud).
- **Templates IPTC** : éditeur visuel (le stub actuel est désactivé).
- **FTPS upload** : implémentation réelle (le stub actuel est désactivé).
