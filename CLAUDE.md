# Contexte projet pour Claude Code

> Fichier de contexte à la racine, lu en priorité par Claude Code à chaque
> nouvelle session sur ce projet. Concis et factuel. Mis à jour à chaque
> phase d'évolution.

---

## Identité du projet

- **Nom** : ShutterstockAnalyzer (alias *Shutterstock AI Metadata Generator*)
- **Pitch** : générateur local de métadonnées microstock (Adobe Stock +
  Shutterstock) avec IA optionnelle via Ollama. **Freemium** depuis
  v2.2 : **tout est gratuit** (scan, IPTC, rapport expert, IA, dual CSV,
  FTP) ; **seul l'export de données est payant** — 3 exports gratuits
  puis clé à vie **10 € (paiement unique)**.
- **Version actuelle** : `v2.2.0` (pivot monétisation = export-only, 10 € à vie)
- **Statut** : **stable, monétisation amorçable** (121 tests verts,
  EXE PyInstaller, dossier de qualification IHM, mécanique de
  licence + gate export + UI livrés). Reste avant ship public : assets
  visuels + listing Gumroad 10 €.

---

## Stack & contraintes techniques

- **Python 3.11+** (`requires-python = ">=3.11"` dans `pyproject.toml`).
- **UI** : CustomTkinter `>=5.2,<6.0` (CTk + Tkinter stdlib).
- **Images** : Pillow `>=10,<12`.
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
pytest tests/ -q                                                       # tests (120)
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
  (`customtkinter`, `Pillow`, `requests`, `urllib3`) intentionnellement
  minimale → EXE PyInstaller ≤ 25 Mo.

### Patterns à éviter
- Imports directs `src.modules.*` depuis `app/views/*` (utiliser la facade).
- Secrets en clair, chemins absolus en dur, paths Linux-only.
- `print()` (utiliser `logger`).
- Création d'un nouveau `App(CTk)` par test (Tkinter ne supporte pas).

---

## État actuel & priorités

### Fini (v2.0.0)
- Multi-plateforme Adobe + Shutterstock (CSV double export, BOM UTF-8,
  keywords séparateur virgule).
- Rapport expert heuristique (sans IA), 8 sections, 4 scores 0-10.
- IA Ollama optionnelle, avec sélection + préchargement modèle dans la
  modale, et indicateur topbar enrichi (nom modèle).
- FTP / FTPS push intégré (stdlib `ftplib`).
- Compactage UI workspace (analyse panel 3 → 2 rows, bouton `📤 Exporter…`
  dans Sources).
- Dossier qualification IHM (méthodologie de qualification) complet sous `test/`.
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

### En cours
- **Production des assets visuels** — GIF hero, 3 screenshots, vidéo
  démo. Specs prêtes dans `docs/MEDIA.md`.
- **Listing Gumroad** — texte produit prêt dans
  `LAUNCH_PROCEDURE.html` (section C.2), reste création du listing
  + workflow fulfillment.

### Fini (Phases 1-7)
- **Phase 1** ✅ — `AUDIT.md` (inventaire, stack, fonctionnel, gaps).
- **Phase 2** ✅ — nettoyage : `_archive/` supprimé, 4 .md déplacés vers
  `audit/`, ruff propre, author harmonisé.
- **Phase 3** ✅ — README v2, CHANGELOG, CONTRIBUTING, SECURITY,
  CLAUDE.md (ce fichier), PROJECT_OVERVIEW.html, docs/MEDIA.md.
- **Phase 4** ✅ — section "Gaps" dans `AUDIT.md` avec P0/P1/P2 + effort
  chiffré. Plan d'attaque ≤ 1 journée pour ship publiquement.
- **Phase 5** ✅ — `docs/MONETIZATION.md` : voie freemium dual-license
  (Pro 29 €/an, 79 € lifetime) + voie fallback (lead magnet portfolio).
- **Phase 6** ✅ — calendrier de distribution sur 4 semaines (GitHub
  Release → dev.to → Reddit → Product Hunt → LinkedIn) dans
  `docs/MONETIZATION.md`.
- **Phase 7** ✅ — `LINKEDIN_DRAFTS.md` : 3 formats prêts à publier
  (court ~ 1 000 char, carousel 8 slides, storytelling ~ 1 900 char) +
  templates de réponses aux commentaires + calendrier.
- **Pivot 2026-05-27** ✅ — réalignement Pro sur l'évaluation qualité,
  3 nouvelles features gated, frontière Community/Pro refondue dans
  les docs (README, MONETIZATION, LAUNCH_PROCEDURE, LINKEDIN_DRAFTS,
  PROJECT_OVERVIEW).

### Chemin critique restant (~ 3,5 j-h avant ship public)
1. `git tag v2.1.0 && git push --tags` (5 min)
2. Capture GIF hero (1 h, storyboard dans `docs/MEDIA.md`)
3. 3 screenshots (workspace, expert_report avec quota, export_batch avec gates) (30 min)
4. `gh release create v2.1.0 dist/*.exe` (15 min)
5. Création listing Gumroad « ShutterstockAnalyzer Pro » (1 h)
6. Mettre à jour README avec liens images réels + URL Gumroad (15 min)
7. Publier le post technique court LinkedIn (5 min)

### Bugs connus
- Aucun bloquant. Voir `audit/JOURNAL_CORRECTIONS.md` pour l'historique
  des correctifs.

---

## Décisions techniques actées

- **MIT license** — usage libre, simple à comprendre, base commune des
  bibliothèques utilisées. Compat freemium (l'API publique reste MIT, la
  version pro packagera des features supplémentaires sous licence
  commerciale distincte).
- **CustomTkinter** plutôt que Qt/PySide — bundling 25 Mo PyInstaller vs
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
  a déjà produit ses CSV. Voir `docs/MONETIZATION.md` § 2.
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
- Les fichiers de Phase 1+ déjà signés : `AUDIT.md`, `audit/RAPPORT_*.md`
  (lecture seule sauf si phase explicite).
- La méthodologie de qualification sous `test/` (matrice + HTML) sans régénérer via les
  scripts `_make_*.py`.

### Fichiers vivants à mettre à jour à chaque phase

- **`CLAUDE.md`** (ce fichier) — sections « État actuel » + « Décisions
  techniques » + « Instructions » à mettre à jour à chaque ajout majeur.
- **`PROJECT_OVERVIEW.html`** — features ✅🟡🔴 + roadmap + métriques LOC.
- **`CHANGELOG.md`** — ajouter une section `## [Unreleased]` puis figer en
  version sémantique au moment du release.
- **`AUDIT.md`** — pas à toucher (figé à Phase 1, sert de baseline).

---

*Dernière mise à jour : 2026-05-29 (v2.2.0 — pivot monétisation = export-only, 10 € à vie).*
