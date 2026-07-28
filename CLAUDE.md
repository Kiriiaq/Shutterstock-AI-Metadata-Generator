# Contexte projet pour Claude Code

> Fichier de contexte à la racine, lu en priorité par Claude Code à chaque
> nouvelle session sur ce projet. Concis et factuel. Mis à jour à chaque
> phase d'évolution.

---

## Identité du projet

- **Nom** : dépôt **StockMeta** ; binaire et nom technique
  `ShutterstockAnalyzer` ; nom commercial *StockMeta Pro*.
- **Pitch** : générateur local de métadonnées microstock (Adobe Stock +
  Shutterstock) avec IA optionnelle via Ollama. **Freemium** depuis
  v2.2 : **tout est gratuit** (scan, IPTC, rapport expert, IA, dual CSV,
  FTP) ; **seul l'export de données est payant** — 3 exports gratuits
  puis clé à vie **10 € (paiement unique)**.
- **Version actuelle** : `v2.3.0` — voir `CHANGELOG.md` pour le détail.
- **Statut** : **stable et publié** (132 tests verts, ruff propre, 2 EXE
  PyInstaller en GitHub Release, dossier de qualification IHM).

---

## Stack & contraintes techniques

- **Python 3.11+** (`requires-python = ">=3.11"` dans `pyproject.toml`).
- **UI** : CustomTkinter `>=5.2,<6.0` (CTk + Tkinter stdlib).
- **Images** : Pillow `>=10,<12` + `pillow-heif >=0.16,<2` (ouverture
  HEIC/HEIF/AVIF ; optionnelle — dégradation gracieuse si absente).
- **HTTP** : `requests` + `urllib3`.
- **Tests** : `pytest`. **Lint/format** : `ruff` (line 120, py311 target).
- **Packaging** : PyInstaller (debug + release profiles via `build.py`).
- **OS cibles** : Windows 10/11 testé. macOS/Linux non testés mais le code
  est portable (sauf `AppUserModelID` Windows + ICO).
- **Dépendances externes (non bundlées)** :
  - **ExifTool** (binaire, subprocess) — IPTC read/write
  - **Ollama** serveur localhost:11434 — **optionnel**, vision IA

### Commandes essentielles

```bash
pip install -e ".[dev]"                                                # install
python main.py                                                         # run from source
pytest tests/ -q                                                       # tests (132)
ruff check app/ src/ main.py build.py tests/                           # lint
ruff format app/ src/ main.py build.py tests/                          # format
python build.py debug | release | all | clean                         # PyInstaller
python test/scripts/run_tests.py                                       # pipeline E2E
python test/scripts/compare_outputs.py                                 # diff vs référence
python test/scripts/_make_inputs.py                                    # régénérer images Pillow
python test/scripts/_make_matrix.py                                    # régénérer matrice XLSX
python test/scripts/_make_html.py                                      # régénérer validation_ihm.html
```

---

## Architecture

```
main.py  →  app.main:main()
            ├─ ShutterstockAIv2 (facade)   ← src/modules/integration.py
            │   ├─ Database                ← src/modules/storage
            │   ├─ MetadataReader/Writer   ← src/modules/engines (ExifTool)
            │   ├─ WorkerPool              ← src/modules/workers
            │   ├─ VisionAnalyzer          ← src/modules/ai (Ollama, optionnel)
            │   ├─ build_expert_report     ← src/modules/analysis/expert_report
            │   ├─ run_export_batch        ← src/modules/export/batch
            │   ├─ csv_exporter            ← src/modules/export/csv_exporter
            │   └─ ftp_uploader            ← src/modules/export/ftp_uploader
            └─ App(CTk)                    ← app/app.py
                ├─ WorkspaceView (home, vue unique)
                └─ 6 modaux : settings, audit, ai_control, validate,
                              expert_report, export_batch
```

**Règle absolue** : l'UI passe **uniquement** par la facade
`ShutterstockAIv2`. Jamais d'import direct depuis
`src/modules/storage/database`, `src/modules/engines/*`, etc.

**Points d'entrée** :
- `main.py` (wrapper 13 lignes) → `app.main:main()`
- `build.py` (PyInstaller debug/release/all/clean)

---

## Conventions de ce projet

### Code
- Type hints sur toutes les API publiques.
- Docstrings sur classes et méthodes publiques.
- **Fonctions ≤ 50 lignes, classes ≤ 300 lignes** (`App` shell justifié).
- `logging` stdlib, **jamais** `print()`.
- Toutes les strings UI passent par `app.i18n.fr.t(key)`.
- Format français via `app.utils.formatters` (NBSP, virgule décimale,
  JJ/MM/AAAA).

### UI
- Layout `grid()` uniquement. Mélanger `pack` dans un même parent → crash.
- Opérations > 300 ms en `threading.Thread(daemon=True)` + retour UI via
  `widget.after(0, cb)`.
- Accessible clavier (Tab/Enter/Esc). Pas d'info portée par la couleur
  seule.
- Couleurs via `palette_pair("key")` (tuple light/dark, CTk gère le swap).

### Tests
- `pytest`, fichiers `test_*.py`, classes `Test*`.
- Tests UI dans `tests/ui/` — un seul `App()` par session (Tk limit).
- Tests backend ne doivent jamais nécessiter ExifTool ou Ollama (mocks).
- Marker `requires_exiftool` pour les exceptions.

### Où ajouter quoi

| Type | Emplacement |
|---|---|
| Nouvelle vue UI | `app/views/<slug>.py` + register dans `app.py::_register_views` |
| Widget réutilisable | `app/components/<name>.py` |
| Logique backend | `src/modules/<area>/<name>.py`, exposer via facade |
| Export | `src/modules/export/` |
| Analyse / heuristique | `src/modules/analysis/` |
| Test backend | `tests/test_core/` |
| Test UI | `tests/ui/` (un seul `App()` autorisé par session) |

### Patterns à respecter
- **Heuristique d'abord** : toute feature doit fonctionner **sans Ollama
  ni ExifTool**. Ils sont des enrichisseurs, pas des prérequis.
- **Posture lax sur la validation** : warnings, pas d'erreurs bloquantes.
  Les reviewers Adobe / Shutterstock font le QA final.
- **Pas de nouvelle dépendance runtime** sans discussion. Liste actuelle
  (`customtkinter`, `Pillow`, `pillow-heif`, `requests`, `urllib3`)
  intentionnellement minimale → **EXE PyInstaller ≤ 35 Mo**. Le plafond
  est passé de 25 à 35 Mo en v2.3.0 : `pillow-heif` ajoute ~8 Mo mais
  débloque HEIC/AVIF, sans quoi les photos de smartphone (cœur de cible)
  ne sont pas traitables.

### Patterns à éviter
- Imports directs `src.modules.*` depuis `app/views/*` (utiliser la facade).
- Secrets en clair, chemins absolus en dur, paths Linux-only.
- `print()` (utiliser `logger`).
- Création d'un nouveau `App(CTk)` par test (Tkinter ne supporte pas).

---

## État actuel

**v2.3.0 — stable.** 132 tests verts, ruff propre, 2 EXE PyInstaller
(release + debug) publiés en GitHub Release.

> L'historique détaillé par version vit dans **`CHANGELOG.md`**, seule
> source de vérité. Ne pas le recopier ici : c'est ce qui a provoqué la
> dérive documentaire de juillet 2026 (ce fichier annonçait encore
> v2.2.0 et 110 tests alors que le code était en v2.3.0 / 132 tests).

Capacités livrées, en une ligne : scan multi-format (JPEG/PNG/TIFF +
HEIC/HEIF/AVIF/WebP/DNG), éditeur IPTC avec écriture *autoritaire*
(IPTC + XMP + EXIF synchronisés, suppression réellement appliquée),
rapport expert heuristique, enrichissement IA Ollama optionnel, export
CSV double plateforme borné aux limites Adobe/Shutterstock, push FTP/FTPS.

### Bugs connus
- Aucun bloquant.

### Non couvert / limites assumées
- **macOS / Linux non testés** — le code est portable, seuls
  l'`AppUserModelID` et l'ICO sont Windows-only.
- **HEIC validé sur fichier généré**, pas sur une photo Samsung/iPhone
  native (pas d'échantillon au moment du dev). Les chemins sont les
  mêmes, mais un test sur vrai fichier reste à faire.
- **Licence = honor-system HMAC**, contournable ; assumé au prix pratiqué.

---

## Historique (archive)

<details>
<summary>Détail des versions antérieures — conservé pour contexte</summary>

### Fini (v2.0.0)
- Multi-plateforme Adobe + Shutterstock (CSV double export, BOM UTF-8,
  keywords séparateur virgule).
- Rapport expert heuristique (sans IA), 8 sections, 4 scores 0-10.
- IA Ollama optionnelle, avec sélection + préchargement modèle dans la
  modale, et indicateur topbar enrichi (nom modèle).
- FTP / FTPS push intégré (stdlib `ftplib`).
- Compactage UI workspace (analyse panel 3 → 2 rows, bouton `📤 Exporter…`
  dans Sources).
- Dossier qualification IHM (méthodo Edvance) complet sous `test/`.
- 90 tests verts, ruff propre.
- 2 EXE PyInstaller (debug + release) 24,8 Mo chacun.

### Fini (v2.1.0 — pivot Pro = évaluation qualité)
- Trois nouvelles features Pro registrées : `expert_report`,
  `dual_csv_export`, `ai_enrichment` (en plus de `batch_unlimited` et
  des 5 features roadmap déjà en place).
- Quota Community **2 aperçus gratuits** sur le rapport expert,
  persisté dans la table `settings` SQLite
  (clé `community_expert_reports_used`).
- Facade : `expert_report_quota_remaining()`,
  `consume_expert_report_quota()`, `reset_expert_report_quota()`.
- UI Expert Report : bandeau de quota + checkbox IA tag « 🔒 Pro » +
  bouton export CSV double tag « 🔒 Pro » + **écran upsell** (benefits
  + Acheter Pro + J'ai déjà une clé) quand quota épuisé.
- UI Export Batch : radio « Les deux 🔒 Pro » + checkbox IA « 🔒 Pro »
  + gates au Start (toast + status badge explicites).
- 9 nouveaux tests (`TestPivotFeatures`, `TestCommunityExpertReportQuota`).
  Total suite : **120 verts**.

### Fini (audit 2026-06-12, branche `audit/2026-06-12`)
- Scan complet du code + rapport d'audit (40 fonctionnalités, 9 bugs).
- **Lot B** : « Ignorer si méta » du batch IA réparé (pré-filtre IPTC
  côté facade + 3 tests) ; colonne « Méta » des Sources honnête.
- **Lot C** : 6 bugs mineurs UI (modale Validation, refresh chip
  licence, export rapport expert en thread, dédoublonnage Sources,
  textbox audit, modale Modèle IA refondue via la facade avec
  persistance URL/modèle + bouton Charger) + docstrings à jour.
- **Lot D/E** : ~2 800 lignes de code mort v1 supprimées (surface
  facade inutilisée, ProcessingPipeline, sidecars XMP, `src/core/`,
  `src/utils/file_utils|validators`, table `processing_queue`,
  historique Router). Suite : **110 tests verts**, ruff propre.
  Détail dans CHANGELOG `[Unreleased]`.

### Fini (v2.3.0 — formats smartphone + fiabilité métadonnées)
- HEIC/HEIF/AVIF/WebP/DNG en scan, analyse et écriture ;
  `src/modules/formats.py` = source de vérité des extensions.
- Écriture autoritaire : un champ vidé est réellement supprimé du
  fichier, miroirs IPTC/XMP/EXIF synchronisés.
- `src/modules/analysis/limits.py` : bornes Adobe/Shutterstock +
  `smart_truncate` (coupe au mot, jamais d'ellipse).
- Prompt IA borné (un titre + une description par image).
- Suite : **132 tests verts**.

</details>

---

## Décisions techniques actées

- **MIT license** — usage libre, simple à comprendre, base commune des
  bibliothèques utilisées. Compat freemium (l'API publique reste MIT, la
  version pro packagera des features supplémentaires sous licence
  commerciale distincte).
- **CustomTkinter** plutôt que Qt/PySide — bundling ~33 Mo PyInstaller vs
  ~120 Mo pour PyQt6. Restriction CTk acceptable pour ce projet.
- **`ftplib` stdlib** plutôt que `paramiko`/`pysftp` — Adobe/Shutterstock
  exigent FTPS (TLS over FTP), pas SFTP. Stdlib suffit, zéro dep extra.
- **Pas de Pydantic** pour les dataclasses — `@dataclass` stdlib + helpers
  manuels. Évite la dépendance, suffisant pour le périmètre.
- **SQLite** plutôt que JSON pour la persistance — concurrent-safe, requêtes
  filtrables (audit log), schéma versionnable.
- **Ollama** plutôt qu'OpenAI/Anthropic API — local, gratuit, sans token,
  sans télémétrie. Le choix « heuristique d'abord » assure que l'outil reste
  utilisable même sans modèle.
- **`_archive/` supprimé** (Phase 2) — historique des UI legacy v1/v2/v3
  conservé en `git log` seulement.
- **Anti-stuffing keywords codé en dur** — listes statiques `BRAND_KEYWORDS`
  + `STUFFING_KEYWORDS` dans `src/modules/analysis/expert_report.py`.
  Le prompt IA les répète aussi, défense en profondeur.
- **Monétisation = export-only, 10 € à vie** (pivot 2026-05-29, v2.2.0)
  — **supersede** le pivot « Pro = évaluation qualité » du 2026-05-27.
  La seule fonctionnalité payante est désormais l'**export de données**
  (`data_export`) ; tout le reste (rapport expert, IA, dual CSV, FTP)
  est gratuit et illimité. Un seul tier : `lifetime` à 10 € (paiement
  unique — plus d'abonnement ni de Solo/Studio). Raison : zéro barrière
  à l'adoption, prix d'impulsion, conversion au moment où l'utilisateur
  a déjà produit ses CSV.
- **Quota Community 3 exports** sur l'export de données — persisté dans
  `settings` SQLite (`community_exports_used`). Partagé par les deux
  points d'export (Export Batch + bouton CSV du rapport expert).
  À ajuster vers 2 si la conversion est faible.
- **HMAC honor-system maintenu** — ed25519 sur la roadmap. Le
  contournement reste possible mais à 10 € one-shot le coût de cracker
  dépasse le prix : le public paie par convenance.

---

## Instructions opérationnelles pour Claude Code

### Ce que tu peux faire seul
- Ajouter un test à `tests/test_core/` pour une nouvelle fonction.
- Renommer / déplacer un fichier en utilisant `git mv` (pas `mv` simple).
- Corriger un warning ruff (`ruff check --fix`).
- Mettre à jour un docstring, un commentaire.
- Ajouter une feature backend isolée + son test.
- Mettre à jour `CHANGELOG.md` après un changement.

### Ce que tu dois valider avant d'exécuter
- Toute modification de l'**API publique** de la facade `ShutterstockAIv2`
  (signature de méthode, suppression d'une méthode).
- Ajout d'une **dépendance runtime** dans `requirements.txt` ou
  `pyproject.toml` (impact bundle PyInstaller).
- Refactor de fichier > 500 LOC (e.g. `workspace.py`, `integration.py`,
  `database.py`).
- Suppression de fichier dans `src/` ou `app/` (vérifier qu'il n'est pas
  utilisé par un test ou un build).
- Bump de version sémantique (`pyproject.toml`, `build.py`, `main.py`).
- Changement de la stratégie freemium / monétisation.

### Tests à lancer après modification

| Type de modif | Vérifs obligatoires |
|---|---|
| Backend (`src/`) | `pytest tests/test_core/ -q` + `pytest tests/smoke/ -q` |
| UI (`app/`) | `pytest tests/ui/ -q` |
| Export / CSV | `python test/scripts/run_tests.py && python test/scripts/compare_outputs.py` |
| Refacto cross-cutting | `pytest tests/ -q` + `ruff check app/ src/ main.py build.py tests/` |
| Avant release | `python build.py all` (smoke test mainloop inclus) |

### Ce que tu ne dois jamais toucher
- `dist/` (généré, dans `.gitignore`).
- `htmlcov/`, `.benchmarks/`, `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`.
- `assets/icons/icone.ico` (binaire).
- `~/.shutterstock_ai/shutterstock_ai.db` (base utilisateur, hors repo).
- Le binaire ExifTool si présent (chemin user).
- La méthodo Edvance sous `test/` (matrice + HTML) sans régénérer via les
  scripts `_make_*.py`.

### Fichiers vivants à mettre à jour

- **`CHANGELOG.md`** — **source de vérité** de l'historique. Alimenter
  `## [Unreleased]` au fil de l'eau, figer en version sémantique au
  moment du release.
- **`CLAUDE.md`** (ce fichier) — uniquement « État actuel » (2-3 lignes),
  « Décisions techniques » et « Instructions ». **Ne jamais y recopier le
  détail du CHANGELOG** : la duplication est ce qui a créé la dérive.
- **`README.md`** — badges (version, nombre de tests) et taille de l'EXE
  à resynchroniser à chaque release, ils dérivent vite.

> Les docs de travail internes (stratégie commerciale, historique
> d'audit, livrables Gumroad) sont volontairement **hors du dépôt
> public** — voir `.gitignore`. Elles restent sur le disque du mainteneur.

---

*Dernière mise à jour : 2026-07-28 (v2.3.0 — formats smartphone, écriture
autoritaire des métadonnées, 132 tests verts ; docs internes sorties du
dépôt public).*
