# JOURNAL CORRECTIONS — audit/20260428

Une ligne par modification, groupée par commit. Format : `fichier:zone — quoi (pourquoi)`.

## `9d0bc36` — chore: snapshot pre-audit v2.0 refactor

Snapshot working-tree complet (60 fichiers) comme cible de rollback. Pas de modif logique.

## `827a62a` — test(smoke): establish baseline safety net for audit refactor

- `tests/conftest.py:30` — `from src.core.database` → `from src.modules.storage.database` (module n'existe pas au premier chemin)
- `tests/smoke/test_smoke.py` (nouveau) — 13 tests baseline DB+IPTC+workers+validators+facade pour piéger les régressions Phase E
- `BASELINE_TESTS.md` (nouveau) — état + sentinelles xfail

## `970f4f3` — fix(db): align integration with actual database API

- `src/modules/integration.py:×5` — `database.add_audit_log(file_path, action, …)` → `database.log_action(action_type, file_path=…, …)` (la première méthode n'a jamais existé)
- `src/modules/integration.py:×2` — `database.update_file_status(path, has_X=True)` → `database.set_file_flags(path, has_X=True)` (la première signature exigeait `file_hash`/`file_size`/`last_modified`)
- `src/modules/storage/database.py` — nouveau `set_file_flags(file_path, has_metadata=None, has_ai_analysis=None)` qui INSERT-OR-IGNORE puis UPDATE partiel
- `tests/smoke/test_smoke.py` — `test_database_set_file_flags` regression guard

## `82d3d34` — fix(db): align batch lifecycle calls with database signatures

- `src/modules/integration.py:1300` — `create_batch(batch_id, total)` → `create_batch(batch_id=…, source_folder="", total_files=…)` (`source_folder` est NOT NULL en schéma)
- `src/modules/integration.py:1335` — `update_batch_progress(batch_id, total_done)` → `update_batch_progress(batch_id, processed=completed+skipped, failed=failed)` (comptes séparés)
- `src/modules/integration.py:1348` — `complete_batch(batch_id, completed, failed)` → `complete_batch(batch_id, status="completed")` (la signature DB n'attend pas de comptes)

## `8b44db8` — fix(models): roundtrip list fields in IPTCFields.from_dict

- `src/modules/models/metadata_models.py:6` — ajout `from dataclasses import fields as dataclass_fields`
- `src/modules/models/metadata_models.py:113-125` — réécriture `IPTCFields.from_dict` avec `{f.name for f in dataclass_fields(cls)}` au lieu de `hasattr(cls, key)` (ce dernier retourne False pour les `field(default_factory=list)` → `keywords` et `supplemental_categories` étaient silencieusement perdus)
- `tests/smoke/test_smoke.py` — sentinelle xfail-strict B-16 promue en regression guard normale

## `4ba4ed1` — fix(ui): wire WritePage with the real database, reader, and writer

- `main.py:355` — `WritePage(tab, self.api)` → `WritePage(tab, database=self.api.database, metadata_reader=self.api.metadata_reader, metadata_writer=self.api.metadata_writer)` (le 1er passait la façade `ShutterstockAIv2` comme paramètre `database`, ce qui rendait le tab Metadata Editor non fonctionnel)

## `e87a623` — chore(release): align v2.0.0 identity across runtime and build

- `main.py` — `resource_path()` helper `sys._MEIPASS`-aware
- `main.py:App.__init__` — `self.iconbitmap(resource_path("assets/icons/icone.ico"))`
- `main.py:App.__init__` — title `ShutterstockAnalyzer v2.0.0 - AI Metadata Generator for Stock Photography`
- `main.py:14` — `AppUserModelID` `ShutterstockAI.v2.0` → `ShutterstockAnalyzer.v2.0`
- `pyproject.toml:7` — `version` `1.0.1` → `2.0.0`
- `pyproject.toml:33-34` — suppression `[project.scripts]` cassé (`shutterstock_ai_v2.main:main` n'existe pas)
- `pyproject.toml:36-37` — drop `shutterstock_ai_v2*` du `packages.find` (n'existe pas)
- `build.py:17` — `VERSION` `1.0.0` → `2.0.0`

## `3bfabeb` — chore(arch): archive legacy UI v1 (~1100 LOC orphan code)

- `_archive/legacy_ui_v1/` (nouveau dossier)
- 9 fichiers déplacés via `git mv` :
  - `src/ui/main_window.py` → `_archive/legacy_ui_v1/main_window.py`
  - `src/ui/components/{advanced_window,sidebar}.py` → `_archive/legacy_ui_v1/components/`
  - `src/ui/pages/page_{analyze,journal,model,source,upload,validation}.py` → `_archive/legacy_ui_v1/pages/`
- `src/ui/__init__.py` — drop `from .main_window import ShutterstockApp` (orphelin)
- `src/ui/components/__init__.py` — drop `Sidebar` export (orphelin)
- `_archive/legacy_ui_v1/README.md` — provenance + procédure de restauration

## `db09c5a` — fix: tighten error handling, honest stub buttons, and CSV export

- `src/modules/ai/ollama_client.py:197` — `except:` (bare) → `except (requests.RequestException, ValueError):`
- `src/ui/pages/write_page.py:188-193` — bouton `Write to All Files` ajout `state="disabled"`, label `"(coming soon)"`, fg gray
- `src/ui/pages/settings_page.py:395-400` — bouton `Test Connection` (FTPS) idem
- `src/ui/pages/settings_page.py:454-458` — bouton `Create New Template` idem
- `src/ui/pages/scan_page.py` — `import csv` au top
- `src/ui/pages/scan_page.py:_export_list` — concat manuel CSV → `csv.writer(f).writerow([...])` (gestion correcte des virgules/quotes/unicode dans les paths) ; resserre `except Exception` → `except OSError`

## `33fa554` — chore(lint): apply ruff auto-fix and ruff format

- 33 fichiers : 87 fixes auto-appliqués
  - 51 imports inutilisés retirés
  - 38 import groups triés (PEP 8 stdlib / third-party / local)
  - 4 f-strings sans placeholders corrigés
  - 3 variables inutilisées retirées
- `ruff format` sur 29 fichiers (line length 120, py311 target)
- Aucun changement de comportement

## `f87a150` — fix(ui+lint): hoist main.py imports, fix lambda late-binding, clear ruff

- `main.py` — réécriture complète de l'organisation de fichier :
  - Imports hoist au module-level (`customtkinter`, page classes, `ShutterstockAIv2`, `OllamaStatus`, `messagebox`, `SplashScreen`)
  - `class App(ctk.CTk)` extraite de `main()` vers module-level (B-17 critique : avant cela le `class App` ne pouvait jamais voir `ShutterstockAIv2` etc., qui étaient locaux à `load_modules()`)
  - Suppression de `sys.path.insert` (Python ajoute déjà le dir du script à `sys.path[0]` ; PyInstaller utilise `sys._MEIPASS`)
  - Suppression de `load_modules()` thread (UX splash préservée par updates progressifs simples)
  - `_start_ai_processing` lambda capture `e` via default-arg (B-18)
- `src/ui/pages/audit_page.py:228` — `[l for l in logs ...]` → `[log for log in logs ...]` (E741)
- `src/ui/pages/audit_page.py:233-234` — `except Exception:` → `except Exception as e:` + lambda `err=str(e)`
- `src/ui/pages/scan_page.py:376` — lambda `err=str(e)` capture
- `src/modules/ai/ollama_client.py:367` — drop `response =` non utilisé
- `src/ui/pages/write_page.py:600-606` — drop boucle morte `files = []` (le corps était déjà un no-op derrière "coming soon")
- `src/modules/engines/metadata_writer.py:385` — `for field, tag in xmp_mapping.items():` → `for xmp_field, tag in ...:` (F402 — `field` shadow l'import dataclasses.field)

## `7457822` — test(ui): headless App lifecycle smoke

- `tests/ui/__init__.py` (nouveau, vide)
- `tests/ui/test_app_smoke.py` (nouveau) — `test_app_full_lifecycle` : instanciation App, vérification title + 6 onglets + facade attribute, fermeture via `on_closing`. Mock `requests.get/post` pour ne pas attendre 5 s sur Ollama down. Test consolidé en un seul plutôt que deux pour éviter pollution Tk inter-test.

## `fa33981` — build: rewrite with debug/release/all/clean subcommands and trim deps

- `build.py` — réécriture complète :
  - CLI argparse avec sous-commandes `debug | release | all | clean`
  - Profil debug : `--console --debug=imports --noupx`
  - Profil release : `--windowed --noconsole --noupx`
  - Communs : `--onefile --icon --add-data assets;assets --add-data src;src --noupx`
  - HIDDEN_IMPORTS trimés à `customtkinter`, `darkdetect`, `PIL` (le reste était fantôme)
  - EXCLUDE_MODULES étendu : drops `pydantic`, `ollama`, `piexif`, `CTkToolTip`, `oletools`, etc.
  - Fonction `_smoke()` post-build (`subprocess.Popen` timeout 8 s)
- `pyproject.toml:dependencies` — drop `CTkToolTip`, `piexif`, `ollama`, `pydantic` (jamais importés dans `src/`)
- `requirements.txt` — synchronisé : `customtkinter`, `Pillow`, `requests`, `urllib3` uniquement
- `BUILD_REPORT.md` (nouveau) — pré-scan + résultats + acceptance check

Build : 24.4 MB par profil. Smoke : both stay alive past timeout.
