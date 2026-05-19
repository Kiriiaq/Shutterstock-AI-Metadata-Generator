# AUDIT — ShutterstockAnalyzer v2.0.0

> Phase 1 du protocole d'audit & valorisation. Inventaire factuel, sans
> modification. Sert de baseline pour les phases suivantes (nettoyage,
> README, monétisation, communication).

> **⚠ Snapshot Phase 1 (pré-nettoyage).** Les sections 1 à 10 décrivent l'état
> du repo AVANT l'application de la Phase 2. Certaines observations
> (`_archive/` présent, author divergent, .md historiques à la racine) sont
> donc devenues caduques. Voir la **section 11 (Gaps prioritized)** ajoutée
> en Phase 4 pour l'état courant. La Phase 2 a corrigé ces points :
> - `_archive/` supprimé (315 Ko, 4 sous-dossiers)
> - `RAPPORT_AUDIT.md` / `JOURNAL_CORRECTIONS.md` / `BASELINE_TESTS.md` /
>   `RELEASE_1.0.0.md` déplacés vers `audit/`
> - `pyproject.toml` auteur harmonisé sur `Emmanuel Grolleau`
> - `docs/architecture.md` renommé en `docs/ARCHITECTURE.md`
> - 3 warnings ruff corrigés

**Date** : 2026-05-19
**Périmètre scanné** : racine + `app/`, `src/`, `tests/`, `test/`, `audit/`,
`docs/`, `_archive/`, `assets/`, `tools/`, `dist/`, `.github/`.

---

## 1. Identité projet

| Item | Valeur |
|---|---|
| Nom | **ShutterstockAnalyzer** (alias *Shutterstock AI Metadata Generator*) |
| Version | `v2.0.0` (déclarée dans `pyproject.toml`, `build.py`, `main.py`) |
| Pitch | Générateur local de métadonnées microstock (Adobe Stock + Shutterstock) avec IA optionnelle via Ollama. |
| Statut | **Stable / WIP enrichissement** — release v1.0.0 publiée, v2.0.0 en cours de packaging final |
| License | **MIT** (`LICENSE`, © 2024-2025 Emmanuel Grolleau) |
| OS cible | Windows 10/11 (testé) — Linux/macOS non couvert |
| Auteur déclaré | Emmanuel Grolleau (LICENSE) / `Kiriiaq` (pyproject — divergence à uniformiser) |

---

## 2. Stack technique

| Couche | Choix | Version | Source |
|---|---|---|---|
| Langage | Python | 3.11+ (`requires-python`) | `pyproject.toml` |
| UI | CustomTkinter | `>=5.2,<6.0` | `requirements.txt` |
| Images | Pillow | `>=10,<12` | `requirements.txt` |
| HTTP | requests + urllib3 | `>=2.31` / `>=2.0` | `requirements.txt` |
| Tests | pytest | `>=7.4` (dev) | `pyproject.toml` |
| Lint | ruff | `>=0.4` (dev) | `pyproject.toml` |
| Packaging | PyInstaller | `>=6` (dev) | `pyproject.toml` |
| CI | GitHub Actions | windows-latest | `.github/workflows/ci.yml` |
| Format CSV export | UTF-8 BOM | — | `csv_exporter.py` |
| Persistance | SQLite local + JSON settings | stdlib | `~/.shutterstock_ai/shutterstock_ai.db` |

**Dépendances externes (non packagées)** :
- **ExifTool** binaire (lecture/écriture IPTC) — exécution via `subprocess`
- **Ollama serveur** (`http://localhost:11434`) — optionnel pour l'enrichissement IA
- Modèles vision : `llama3.2-vision:11b`, `llava:7b`, `moondream:1.8b` (au choix)

---

## 3. Inventaire fichiers

### 3.1 Racine

| Fichier | Taille | Rôle | État |
|---|---|---|---|
| `README.md` | 12 K | Pitch + install + workflow | ✅ — à jour pour v1, à enrichir v2 |
| `BUILD_REPORT.md` | 17 K | Trace des builds PyInstaller | ✅ |
| `JOURNAL_CORRECTIONS.md` | 8,4 K | Changelog technique par commit (campagne audit 20260428) | ✅ |
| `BASELINE_TESTS.md` | 3,5 K | État test suite pré-refactor (24 tests historiques) | 🟡 obsolète (90 tests aujourd'hui) |
| `RAPPORT_AUDIT.md` | 8,2 K | Audit interne d'avril 2026 | ✅ archive |
| `RELEASE_1.0.0.md` | 930 B | Notes release v1 | 🟡 (n'évoque pas v2) |
| `LICENSE` | 1,1 K | MIT | ✅ |
| `pyproject.toml` | 1,4 K | Build/lint/test config | ✅ |
| `requirements.txt` | 587 B | Runtime deps | ✅ (4 deps, propre) |
| `main.py` | 355 B | Wrapper 13 lignes → `app.main:main` | ✅ |
| `build.py` | 7,4 K | PyInstaller debug / release / clean | ✅ |
| `.gitignore` | — | Python + secrets + build/ + venv/ + PyInstaller | ✅ |
| `AUDIT.md` (ce fichier) | — | Phase 1 — inventaire | ✅ nouveau |

### 3.2 Dossiers de premier niveau

| Dossier | Taille | Rôle | État |
|---|---|---|---|
| `app/` | 756 K (7 000 LOC) | Couche UI v3 — CustomTkinter | ✅ active |
| `src/` | 928 K (10 600 LOC) | Backend (AI, engines, export, models, storage, workers) | ✅ active |
| `tests/` | 508 K (2 600 LOC) | Suite pytest (90 tests verts) | ✅ |
| `test/` | 55 M | Dossier de qualification IHM (méthodologie de qualification) — matrice XLSX + HTML + inputs Pillow | ✅ récent |
| `audit/` | 356 K | Audits internes successifs + captures écran | ✅ historique |
| `docs/` | 12 K | `architecture.md` (cartographie v3) | 🟡 minimal |
| `_archive/` | 315 K | UI v1, v2, v3-predense, v3-views legacy | ⚪ archive (à garder ou supprimer cf. §7) |
| `dist/` | 50 M | 2 EXE PyInstaller (release + debug) | ⚪ généré (à ignorer du repo) |
| `htmlcov/` | 3,6 M | Coverage report HTML | ⚪ généré (.gitignore OK) |
| `assets/icons/` | 32 K | `icone.ico` (Windows) | ✅ |
| `tools/` | 8 K | `wcag_check.py` (accessibilité couleurs) | 🟡 utilité ponctuelle |
| `.github/workflows/` | — | `ci.yml` (lint + test) + `release.yml` (tag push) | ✅ |
| `.benchmarks/`, `.pytest_cache/`, `.ruff_cache/` | 43 K | Caches outils dev | ⚪ ignorés .gitignore |

---

## 4. Cartographie code

### 4.1 Point d'entrée

```
main.py  ─ → app.main:main()
            └─ ShutterstockAIv2(facade)  ← src/modules/integration.py
                ├─ Database           ← src/modules/storage
                ├─ MetadataReader     ← src/modules/engines (ExifTool)
                ├─ MetadataWriter     ← src/modules/engines (ExifTool)
                ├─ WorkerPool         ← src/modules/workers
                ├─ VisionAnalyzer    ←(optionnel) src/modules/ai (Ollama)
                ├─ run_export_batch   ← src/modules/export/batch
                ├─ FtpConfig/upload   ← src/modules/export/ftp_uploader
                └─ build_expert_report← src/modules/analysis/expert_report
            └─ App(CTk)              ← app/app.py
                ├─ WorkspaceView (home, vue principale)
                └─ 6 modaux : settings, audit, ai_control, validate,
                              expert_report, export_batch
```

### 4.2 Modules backend (`src/`)

| Module | LOC | Rôle | État |
|---|---|---|---|
| `src.core.params` | 80 | Dataclass `ShutterstockParams` (defaults projet) | ✅ |
| `src.core.config_manager` | — | JSON settings + path resolver | ✅ |
| `src.core.logger` | — | Setup `logging` stdlib | ✅ |
| `src.utils.validators` | 343 | Validations format fichier, dimensions, complétude metadata | ✅ |
| `src.utils.file_utils` | — | Helpers chemins, hash | ✅ |
| `src.utils.subprocess_helper` | — | `CREATE_NO_WINDOW` Windows | ✅ |
| `src.modules.storage.database` | 824 | SQLite : audit log, file flags, batch, metadata history | ✅ |
| `src.modules.workers.worker_pool` | — | Pool de threads + `collect_image_files` + clean_keywords | ✅ |
| `src.modules.engines.metadata_reader` | — | Lecture EXIF/IPTC/XMP via ExifTool JSON | ✅ |
| `src.modules.engines.metadata_writer` | — | Écriture IPTC/XMP via ExifTool | ✅ |
| `src.modules.engines.iptc_engine` | — | Templates IPTC réutilisables | ✅ |
| `src.modules.models.metadata_models` | 720 | `ShutterstockMetadata`, `IPTCFields`, `ExpertMetadataReport`, etc. | ✅ |
| `src.modules.ai.ollama_client` | 523 | Client HTTP Ollama (list, load, generate, vision) | ✅ |
| `src.modules.ai.vision_analyzer` | — | Orchestrateur batch IA | ✅ |
| `src.modules.ai.prompt_templates` | — | Prompts paramétrés Shutterstock/Adobe/Getty | ✅ |
| **`src.modules.analysis.expert_report`** | 852 | **Builder heuristique sans IA + enrichissement IA optionnel** | ✅ récent |
| **`src.modules.analysis.platform_compliance`** | — | **Checks lax Adobe/Shutterstock (warnings non bloquants)** | ✅ récent |
| **`src.modules.export.csv_exporter`** | 128 | **CSV Adobe + Shutterstock** | ✅ récent (bug P0 fixé) |
| **`src.modules.export.batch`** | 322 | **Orchestrateur batch (reports → CSV → IPTC → FTP)** | ✅ récent |
| **`src.modules.export.ftp_uploader`** | 315 | **FTP/FTPS via stdlib `ftplib`** | ✅ récent |
| `src.modules.integration` | 1617 | Facade `ShutterstockAIv2` (point unique pour UI) | ✅ — gros fichier mais cohérent |

### 4.3 Couche UI (`app/`)

| Module | LOC | Rôle | État |
|---|---|---|---|
| `app.main` | 75 | Bootstrap (logging, theme, backend init, mainloop) | ✅ |
| `app.app:App(CTk)` | 628 | Shell, router, modal manager, raccourcis | ✅ |
| `app.core.events` | — | `EventBus` léger | ✅ |
| `app.core.state` | — | `AppState` clé/valeur partagé | ✅ |
| `app.core.navigation` | — | Router (home + modals) | ✅ |
| `app.config.theme` | — | 2 palettes + `ThemeManager` + `palette_pair()` | ✅ |
| `app.config.shortcuts` | — | Raccourcis Ctrl+1..3, Ctrl+?, Esc | ✅ |
| `app.i18n.fr` | — | Toutes les strings UI en français | ✅ |
| `app.utils.formatters` | — | NBSP / virgule décimale / JJ/MM/AAAA | ✅ |
| `app.components.*` | 8 widgets | data_table, form_field, toast, tooltip, confirm_dialog, empty_state, topbar | ✅ |
| `app.views.workspace` | **1856** | Vue principale, 7 panneaux | ⚠️ **gros fichier, candidat split** |
| `app.views.expert_report` | 677 | Modal rapport expert (8 sections) | ✅ récent |
| `app.views.export_batch` | **744** | Modal batch export (Adobe/SH + IPTC + FTP + Ollama) | ✅ récent |
| `app.views.ai_control` | — | Modal contrôle Ollama | ✅ |
| `app.views.audit` | 249 | Modal historique opérations | ✅ |
| `app.views.validate` | 179 | Modal validation pré-upload | ✅ |
| `app.views.settings` | — | Modal paramètres | ✅ |

---

## 5. Inventaire fonctionnel

### 5.1 Fonctionnalités utilisateur

| ID | Feature | Module backend | Vue UI | État |
|---|---|---|---|---|
| F-01 | Scanner un dossier d'images | `workers.worker_pool.collect_image_files` | WorkspaceView/Sources | ✅ |
| F-02 | Ajout incrémental fichiers / dossier | idem | WorkspaceView/Sources | ✅ |
| F-03 | Multi-sélection avec checkbox | DataTable | WorkspaceView/Sources | ✅ |
| F-04 | Lire IPTC depuis une image | `engines.metadata_reader` | WorkspaceView/IPTC editor | ✅ requires ExifTool |
| F-05 | Écrire IPTC dans une image | `engines.metadata_writer` | WorkspaceView/IPTC editor | ✅ requires ExifTool |
| F-06 | **Rapport expert SANS IA** (heuristique) | `analysis.expert_report.build_expert_report` | `ExpertReportView` | ✅ récent |
| F-07 | **Rapport expert AVEC IA** (Ollama) | `analysis.expert_report.enrich_with_ai_result` | `ExpertReportView` (checkbox) | ✅ récent |
| F-08 | Analyse IA batch (legacy) | `ai.vision_analyzer.analyze_batch` | WorkspaceView/Analyse IA | ✅ |
| F-09 | Validation pré-upload | `integration.validate_image` | `ValidateView` | ✅ |
| F-10 | Historique opérations + filtres + export | `storage.database.get_audit_logs` | `AuditView` | ✅ |
| F-11 | Theme toggle (light/dark/system) | `config.theme.ThemeManager` | Topbar | ✅ |
| F-12 | **Export CSV Adobe Stock** | `export.csv_exporter.write_adobe_csv` | `ExportBatchView` | ✅ récent |
| F-13 | **Export CSV Shutterstock** (fix P0 virgules) | `export.csv_exporter.write_shutterstock_csv` | `ExportBatchView` | ✅ récent |
| F-14 | **Export double CSV** (les deux d'un coup) | `export.csv_exporter.export_double_csv` | `ExportBatchView` | ✅ récent |
| F-15 | **Orchestrateur batch (reports → CSV → IPTC → FTP)** | `export.batch.run_export_batch` | `ExportBatchView` | ✅ récent |
| F-16 | **Push FTP/FTPS** vers portail contributeur | `export.ftp_uploader.upload_files` | `ExportBatchView` (toggle) | ✅ récent |
| F-17 | **Test connexion FTP** | `export.ftp_uploader.test_connection` | `ExportBatchView` (bouton) | ✅ récent |
| F-18 | **Sélection + préchargement modèle Ollama** | `integration.preload_model` | `ExportBatchView` (bandeau IA) | ✅ récent |
| F-19 | **Chip Ollama topbar enrichie** (nom modèle) | `workspace._refresh_dynamic_worker` | Topbar | ✅ récent |
| F-20 | Raccourcis clavier Ctrl+1/2/3 / Ctrl+? / Esc | `config.shortcuts` | global | ✅ |

### 5.2 Backend exposé par la facade `ShutterstockAIv2`

| Méthode | Rôle |
|---|---|
| `read_metadata` / `write_metadata` / `write_shutterstock_metadata` | I/O IPTC |
| `process_folder` / `process_folder_ai` | Batch legacy |
| `analyze_image_ai` / `analyze_batch_ai` | Pipeline IA Ollama |
| `init_ai` / `check_ai_status` | Cycle de vie Ollama |
| `list_vision_models` / `preload_model` / `get_current_model` | **Gestion modèles Ollama** ✅ récent |
| `build_expert_report` / `build_expert_reports_batch` | **Rapport heuristique** ✅ récent |
| `export_double_csv` | **CSV Adobe + Shutterstock** ✅ récent |
| `export_batch` | **Orchestrateur batch complet** ✅ récent |
| `test_ftp_connection` | **Probe FTP** ✅ récent |
| `validate_image` / `validate_shutterstock_metadata` | Validation lax |
| `compare_ai_with_existing` / `get_metadata_diff` | Comparaison versions IPTC |
| `get_setting` / `set_setting` / `get_all_settings` | Settings persistés DB |
| `get_audit_logs` / `get_statistics` | Audit + métriques |
| `get_templates` / `apply_template` | Templates IPTC |

---

## 6. Tests & qualité

| Item | État | Détail |
|---|---|---|
| Suite automatisée | ✅ **90 / 90 verts en ~5 s** | `pytest tests/ -q` |
| Couverture nouveaux modules | ✅ | `test_expert_report.py`, `test_csv_exporter.py`, `test_platform_compliance.py`, `test_export_batch.py`, `test_ftp_uploader.py`, `test_ollama_facade.py` |
| Tests smoke UI | ✅ 1 test e2e | `tests/ui/test_app_v3_shell.py` ouvre les 6 modaux |
| Coverage HTML | 🟢 disponible | `htmlcov/` généré |
| Lint ruff | 🟡 **3 warnings** mineurs | imports inutilisés (`SPACE_LG`, `BytesIO`) + variable inutilisée (`rows`) — détails §8 |
| Dossier de qualification IHM (méthodologie de qualification) | ✅ complet | `test/` : matrice XLSX 49 tests + HTML interactif + 15 images Pillow + scripts run/compare/refs + rapport_qualification.md |
| Pipeline E2E | ✅ | 15 inputs → 2 CSV en 5,4 s, **cell-for-cell match** vs `outputs_reference/` |
| Build PyInstaller | ✅ | Debug + release, 24,8 Mo chacun, smoke mainloop OK |
| CI GitHub Actions | ✅ basique | lint + test sur windows-latest |
| CI release | ✅ | trigger sur tag `v*` (à vérifier en pratique) |

---

## 7. Artefacts polluants / candidats nettoyage

| Item | Statut | Reco |
|---|---|---|
| `__pycache__/` (24 dossiers) | 🟡 traînant | Déjà dans `.gitignore` ; OK |
| `htmlcov/` (3,6 Mo) | 🟡 traînant | Déjà dans `.gitignore` |
| `.benchmarks/`, `.pytest_cache/`, `.ruff_cache/` | ⚪ caches dev | Ignorés |
| `dist/` (50 Mo) | 🟡 généré | Ignoré, OK — mais à **ne pas committer** |
| `_archive/` (315 Ko) | ⚪ choix conscient | UI v1/v2/v3-predense préservées en lecture seule. **Décision à prendre Phase 2** : garder en `_archive/` ou supprimer ? |
| `audit/` (356 Ko) | 🟡 documentation | Histoire des audits + captures écran. Garder mais éventuellement déplacer hors racine (`docs/audit/`) |
| `BASELINE_TESTS.md` racine | 🟡 obsolète | Référence à 24 tests, on est à 90 — à mettre à jour ou à archiver dans `audit/` |
| `RELEASE_1.0.0.md` racine | 🟡 obsolète | Pas de note v2 — à compléter ou déplacer dans `CHANGELOG.md` |
| `RAPPORT_AUDIT.md` racine | 🟡 doublon avec `audit/` | À déplacer dans `audit/` |
| `JOURNAL_CORRECTIONS.md` racine | 🟡 trop technique pour racine | À déplacer dans `audit/` ou fusionner avec `CHANGELOG.md` |
| `tools/wcag_check.py` | 🟡 utilité ponctuelle | OK à conserver dans `tools/` |
| Secrets / mots de passe en clair | ✅ **aucun détecté** | Le seul `password` est le **champ FTP saisi à l'écran**, jamais persisté (cf. `export_batch.py:451` + `ftp_uploader.py:56`) |
| Données perso / API keys | ✅ aucun | `.env*` ignoré, pas de fichier `.env` détecté |
| Fichiers > 1 Mo dans le repo source | ✅ aucun | Plus gros : `workspace.py` 84 Ko |
| Path absolus en dur | ⚪ à scanner | Aucun détecté lors du scan rapide |

---

## 8. Code mort / duplications / faiblesses

| Item | Localisation | Reco |
|---|---|---|
| Import inutilisé `SPACE_LG` | `app/views/export_batch.py:36` | Fix ruff `--fix` |
| Import inutilisé `io.BytesIO` | `tests/test_core/test_ftp_uploader.py:9` | Fix ruff `--fix` |
| Variable inutilisée `rows` | `app/views/export_batch.py:526` | Fix ruff manuel (commenté pour future fallback, mais inutile aujourd'hui) |
| `workspace.py` 1 856 lignes | `app/views/workspace.py` | 🟡 **candidat à split en 4 panneaux** dans la Phase 2 (`workspace/sources.py`, `editor.py`, `analyze.py`, `right_column.py`) |
| `integration.py` 1 617 lignes | `src/modules/integration.py` | 🟡 grosse facade mais cohérente. Refacto si on veut découper en mixins (1 par feature) |
| Author divergence | `pyproject.toml` (`Kiriiaq`) vs `LICENSE` (`Emmanuel Grolleau`) | À uniformiser |
| Doc `RELEASE_1.0.0.md` mentionne **OpenAI API** | racine | Trompeur : v2 n'utilise que Ollama local. À corriger |
| Doc `RELEASE_1.0.0.md` mentionne `1.0.0` mais build dit `2.0.0` | racine | Mettre à jour ou migrer vers `CHANGELOG.md` |
| `_archive/legacy_ui_v3_predense/` + `legacy_ui_v3_views/` | `_archive/` | Décider : conserver pour historique ou purger (315 Ko au total, peu coûteux) |

---

## 9. Conformité dimensions auditées (préview Phase 4)

| Dimension | État | Note |
|---|---|---|
| Onboarding (<5 min) | 🟢 OK | EXE double-clic + Ollama optionnel |
| Configuration | 🟢 OK | Settings DB + `.env*` ignorés, pas de hard-coded paths critiques |
| Robustesse | 🟢 OK | Try/except sur I/O, posture lax (warnings vs errors) |
| Packaging | 🟢 OK | 2 EXE PyInstaller buildés et smoke-testés |
| Cross-platform | 🟡 | Windows-only en pratique (PATH ExifTool, `assets/icons/icone.ico`, AppUserModelID) — macOS/Linux non testé |
| Tests | 🟢 | 90 tests + suite QA IHM 49 cas |
| Docs | 🟡 | README v1 OK mais ne couvre pas Adobe + FTP + Rapport expert (v2) |
| Démo visuelle | 🟡 | Pas de GIF / vidéo. Quelques screenshots dans `audit/captures/` mais usage interne |
| Versionning | 🟢 | tag `v1.0.0` posé en pratique (CI release configurée) — `v2.0.0` à tagger |
| Sécurité | 🟢 | Pas de secret committé, password FTP non persisté |

---

## 10. Conclusion Phase 1

**Diagnostic** : projet **mature en backend et tests**, **bien architecturé** (séparation app/src, facade centrale), **récemment enrichi** (Adobe Stock, double CSV, FTP, Ollama preload). Les manques sont **principalement documentaires et cosmétiques** : README désaligné v1 vs v2, archives à arbitrer, lint mineur, pas de démo visuelle.

**Travail restant pour shipping propre** :
1. **Phase 2 (nettoyage)** : ranger `audit/`, fixer 3 warnings ruff, statuer sur `_archive/`, déplacer ou supprimer les `*.md` obsolètes racine, harmoniser author.
2. **Phase 3** : produire **README v2** alignant Adobe + Shutterstock + FTP + Ollama optionnel, `CHANGELOG.md` Keep a Changelog, `CONTRIBUTING.md`, `CLAUDE.md` (mémoire), `PROJECT_OVERVIEW.html` (dashboard).
3. **Phase 4** : combler les 🟡 (démo visuelle GIF, doc multi-plateforme).
4. **Phase 5+** : voies de monétisation à étudier (open-source + dons / freemium / Gumroad one-shot / lead magnet portfolio).

**Volume travail estimé global pour finir la valorisation** : `~2 j-h` (Phases 2+3) puis `~1 j-h` (Phases 5-7).

---

## 11. Phase 4 — Gaps prioritized (post-Phases 1-3)

Tableau de conformité réévalué après l'application des Phases 2 et 3.
Référence colorée : 🟢 OK / 🟡 à compléter / 🔴 bloquant.

| # | Dimension | État | Détail | Priorité | Effort |
|---|---|---|---|---|---|
| D-01 | **Onboarding** (<5 min) | 🟢 | EXE double-clic. Ollama optionnel. Doc README clair. | — | — |
| D-02 | **Configuration** | 🟢 | Settings SQLite, pas de hard-coded, `.env*` ignoré, FTP password non persisté. | — | — |
| D-03 | **Robustesse** | 🟢 | Try/except sur I/O, posture lax (warnings vs errors), 90 tests verts. | — | — |
| D-04 | **Packaging Windows** | 🟢 | 2 EXE PyInstaller 24,8 Mo, smoke mainloop OK. | — | — |
| D-05 | **Cross-platform** | 🟡 | Code portable mais EXE Windows-only. ICO + AppUserModelID Windows-specific. | P2 | 1-2 j |
| D-06 | **Tests automatisés** | 🟢 | 90 tests unit + 1 smoke UI + 49 tests qualif IHM. | — | — |
| D-07 | **Docs internes** | 🟢 | README v2, CHANGELOG, CONTRIBUTING, SECURITY, CLAUDE.md, PROJECT_OVERVIEW.html, AUDIT.md. | — | — |
| D-08 | **Démo visuelle** | 🟡 | Pas de GIF, pas de vidéo, pas de screenshot README. Specs prêtes dans `docs/MEDIA.md`. | **P0** | 2 h |
| D-09 | **Versionning** | 🟡 | `v1.0.0` taggé, `v2.0.0` **pas encore taggé**. CI release-on-tag prête. | **P0** | 5 min |
| D-10 | **GitHub Release v2** | 🟡 | Aucune release publiée pour v2.0.0. Workflow `release.yml` configuré. | **P0** | 15 min |
| D-11 | **Sécurité** | 🟢 | Pas de secret committé, mots de passe non persistés, FTPS par défaut, threat model documenté. | — | — |
| D-12 | **Dépendances CVE** | 🟡 | Pas d'audit `pip-audit` automatisé en CI. Liste courte (4 deps) limite l'exposition. | P2 | 1 h |
| D-13 | **Logo + identité visuelle** | 🟡 | Icône Windows OK, pas de logo SVG vectoriel ni de bannière. | P1 | 1 h |
| D-14 | **README badges live** | 🟡 | Badges statiques actuellement. Brancher sur GitHub Actions (build status, coverage). | P2 | 30 min |
| D-15 | **Page produit / landing** | 🟡 | Pas de page externe (à hoster sur GitHub Pages ou Gumroad). | P1 | 4 h |
| D-16 | **Stratégie pro / freemium** | 🟡 | Voie choisie (freemium dual-license) mais non détaillée. | **P0** | 1 h (Phase 5) |
| D-17 | **Workspace.py 1856 LOC** | 🟡 | Gros fichier monolithique. Split en 4 panneaux refacto safe. | P2 | 4-6 h |
| D-18 | **macOS / Linux build** | 🟡 | Code portable, builds non testés. | P3 | 1-2 j |
| D-19 | **Drag & drop dans Sources** | 🟡 | Roadmap mais non implémenté. | P2 | 2 h |
| D-20 | **Pipeline pro (batch > 50 + FTP scheduling)** | 🟡 | Frontière freemium identifiée mais non packagée. | P1 | 1-2 j |

### Plan d'attaque immédiat (≤ 1 journée)

| Ordre | Action | Effort | Bloquant pour |
|---|---|---|---|
| 1 | Tag `git tag v2.0.0 && git push --tags` | 5 min | release page, communication |
| 2 | Capture GIF hero (cf. storyboard `docs/MEDIA.md`) | 1 h | README, Product Hunt, LinkedIn |
| 3 | Captures 3 screenshots (workspace, expert_report, export_batch) | 30 min | README, GitHub release |
| 4 | `gh release create v2.0.0 dist/ShutterstockAnalyzer*.exe --notes-from CHANGELOG.md` | 15 min | distribution Gumroad / téléchargements |
| 5 | Mise à jour README avec liens images réels | 15 min | qualité d'accueil page GitHub |
| 6 | `docs/MONETIZATION.md` (Phase 5) | 1 h | discussions sponsors / Gumroad |
| 7 | 3 formats LinkedIn (Phase 7) | 30 min | communication d'amorçage |

**Total chemin critique** : ~ 4 h pour passer d'un repo « propre » à un projet **présentable publiquement avec démo, release, monétisation préparée et communication prête**.

---

*Phase 4 close. Les Phases 5-7 enchaînent ci-dessous, sans intervention manuelle nécessaire avant le tag v2.0.0 et la production des assets visuels.*
