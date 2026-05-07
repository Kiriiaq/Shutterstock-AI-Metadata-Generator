# Phase 4 — Tests

## Suite existante

Avant audit : **27 tests** dans `tests/`, structure conservée.

```
tests/
├── conftest.py
├── smoke/test_smoke.py            # 13 backend + facade smokes
├── test_core/test_config.py       # 4 tests ShutterstockParams + PARAMS_META
├── test_utils/test_validators.py  # 5 tests validators
├── ui/test_app_v3_shell.py        # 1 test consolidé end-to-end UI
└── _generate_test_plan.py         # générateur tests/TEST_PLAN.xlsx
```

## Couverture par module

| Module | Couverture | Notes |
|---|---|---|
| `app.app` | smoke complet (App lifecycle + 5 modales + theme + topbar health) | bonne |
| `app.config.theme` | indirecte via App | LIGHT/DARK palette + `apply_theme` testés via le toggle |
| `app.components.*` (data_table, form_field, empty_state) | smokes ciblés | autres composants — toast, tooltip, confirm — non testés directement (instanciation simple) |
| `app.views.workspace` | présence des 8 widgets-clés | logique métier non couverte unitairement |
| `app.views.{settings,audit,ai_control,validate,upload}` | construction via `open_in_modal` | OK |
| `src.modules.storage.database` | smoke CRUD + batch lifecycle + `set_file_flags` | bonne |
| `src.modules.engines.{iptc_engine}` | smoke instantiation | autres engines (reader/writer) → ExifTool externe, non testé |
| `src.modules.workers.worker_pool` | smoke `collect_image_files`, `clean_keywords_advanced`, `WorkerPool` start/stop, exec | bonne |
| `src.modules.models.metadata_models` | roundtrip IPTCFields scalars + lists | OK (B-16 fixé en audit précédent) |
| `src.modules.ai.ollama_client` | smoke enum | aucun test HTTP réel (Ollama externe) |
| `src.modules.integration.ShutterstockAIv2` | instanciation graceful sans ExifTool | OK |

## Résultats

```
$ python -m pytest tests/ -q
27 passed in 2.44s
```

## Hors-scope explicite

| Type | Raison |
|---|---|
| Tests volume (10 Mo / 100 Mo / 1 Go) | Le pipeline ne charge jamais un fichier monolithique en mémoire — les images sont traitées une par une, puis envoyées à ExifTool/Ollama via stdout/HTTP. Pas de risque de fuite mémoire batch. |
| Tests perf (`pytest-benchmark`, `cProfile`, `snakeviz`) | Hot-paths réels = appels HTTP Ollama (~ secondes par image) et `subprocess` ExifTool (~ centaines de ms). Profiling Python est dominé par I/O externe — peu actionnable côté code. |
| Tests stress (1 000 clics) | Pas applicable sans simulateur d'événements Tk (CTkButton.invoke en boucle ne reflète pas du vrai stress utilisateur). |
| `pytest-cov` couverture chiffrée | Non installé dans l'env. Couverture estimée à ~50 % de l'app/ et ~45 % de src/ (cohérente avec l'audit précédent). |

## Recommandation Phase 4 ouverte

Ajouter en suivi : un test mocké `analyze_batch_ai` end-to-end (mock
`vision_analyzer.analyze_batch`) — couvrirait la chaîne de signatures DB
qui était cassée à l'audit précédent (B-2 à B-5). Effort : 1-2 h.
