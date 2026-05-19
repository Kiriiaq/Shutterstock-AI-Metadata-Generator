# BASELINE_TESTS — audit/20260428

État de la suite de tests **avant** les refactors de la Phase E. Sert de filet
de sécurité : toute régression sur ces tests pendant E/F doit être traitée
avant de poursuivre.

## Commande

```bash
python -m pytest tests/ -q
```

## Résultat actuel

```
24 passed, 1 xfailed in 0.57s
```

## Couverture du filet

| Module | Surface couverte | Test |
|---|---|---|
| `src.core.params` | Defaults + `to_dict`/`from_dict` | `tests/test_core/test_config.py`, `tests/smoke/test_smoke.py::test_shutterstock_params_serialization` |
| `src.core.config_manager` | Import-time | `test_src_package_imports` |
| `src.utils.validators` | `validate_image_dimensions`, `validate_metadata_completeness` | `tests/test_utils/test_validators.py`, `test_validators_dimensions` |
| `src.modules.storage.database` | `set_setting`/`get_setting`, `log_action`/`get_audit_logs`, `create_batch`/`update_batch_progress`/`complete_batch`, `get_statistics` | `test_database_crud`, `test_database_batch_lifecycle` |
| `src.modules.engines.iptc_engine` | `list_templates` instantiation | `test_iptc_engine_templates` |
| `src.modules.models.metadata_models` | `IPTCFields` scalar roundtrip (lists bug B-16 → xfail) | `test_iptc_fields_roundtrip_scalars`, `test_iptc_fields_roundtrip_lists_known_broken` |
| `src.modules.workers.worker_pool` | `collect_image_files` (recursive + flat), `clean_keywords_advanced`, `WorkerPool` start/stop + handler exec | `test_collect_image_files`, `test_clean_keywords_basic`, `test_worker_pool_start_stop`, `test_worker_pool_executes_handler` |
| `src.modules.ai.ollama_client` | `OllamaStatus` enum | `test_ollama_status_enum` |
| `src.modules.integration` | `ShutterstockAIv2` instantiation (graceful sans ExifTool) | `test_shutterstock_ai_v2_instantiates` |
| Package | Importabilité 15 modules clés | `test_src_package_imports` |

## Sentinelles xfail (bugs documentés, fix programmé Phase E)

| Bug | Test | Détail |
|---|---|---|
| **B-16** | `test_iptc_fields_roundtrip_lists_known_broken` | `IPTCFields.from_dict` filtre via `hasattr(cls, key)` qui retourne `False` pour les champs `field(default_factory=list)` → `keywords` et `supplemental_categories` perdus en désérialisation. `xfail(strict=True)` : passera en xpass dès le fix → alerte. |

## Ce que la baseline ne couvre PAS (et pourquoi)

| Surface | Pourquoi pas | Quand |
|---|---|---|
| Pipeline `analyze_batch_ai` | Bugs B-2/B-3/B-4/B-5 (signatures DB cassées) → impossible à exercer en l'état | Couverture ajoutée en Phase F après fix |
| `WritePage` interactions | Bug B-1 (câblage `WritePage(tab, self.api)` cassé) | Phase F après fix + tests smoke UI headless |
| `MetadataReader`/`MetadataWriter` | Dépend d'ExifTool externe non bundlé | Phase F : test conditionnel `pytest.importorskip` ou marker `requires_exiftool` |
| Onglets UI live | Pas de display dans CI ; instanciation only en F.4 | Phase F.4 |
| Pages orphelines `page_*.py` | Code mort confirmé (Phase B), à archiver en E.5 | Pas couvert (sera supprimé) |

## Conditions de sortie Phase C

- [x] `pytest tests/` ≥ 100% passants (xfail comptés OK)
- [x] `tests/conftest.py` charge sans erreur (import `Database` corrigé)
- [x] Bugs déjà identifiés ont une sentinelle xfail OU sont déclarés hors-scope baseline
- [x] Aucune dépendance externe non documentée (ExifTool, Ollama serveur) requise par les smoke tests
