# RAPPORT AUDIT — ShutterstockAnalyzer v2.0.0

Branche `audit/20260428` à partir du tag `pre-audit-20260428`.
13 commits atomiques. 17 bugs fermés. Suite verte. EXE livrés.

## 1. RÉSUMÉ EXÉCUTIF

### Verdict livrable

**OUI**, sous réserve des résiduels listés en §7. Avant la campagne, l'app v2.0 ne pouvait littéralement pas démarrer (B-17 : la classe `App` était définie dans `main()` mais référençait des imports locaux à une fonction imbriquée). Le tag `pre-audit-20260428` capture cet état. Aujourd'hui :

- `dist/ShutterstockAnalyzer.exe` (24.4 MB) lance et reste vivant
- 27 tests verts (15 baseline + 13 smoke métier + 1 smoke UI headless)
- Ruff propre (0 warning)

### Métriques avant / après

| Axe | Avant | Après | Cible | Δ |
|---|---|---|---|---|
| Démarrage applicatif | **NameError immédiat** | Mainloop atteint | OK | ✅ |
| Tests passants | 11 (avec fixture cassée silencieusement) | 27 | 100 % smoke | ✅ |
| Bugs câblage UI/DB | 6 bloquants + 7 majeurs/mineurs | 0 connus | 0 | ✅ |
| Imports inutilisés | 51 | 0 | 0 | ✅ |
| Imports désordonnés | 38 | 0 | 0 | ✅ |
| Deps déclarées non utilisées | 4 (`CTkToolTip`, `piexif`, `ollama`, `pydantic`) | 0 | 0 | ✅ |
| LOC code mort actif (UI orpheline) | ~1100 (importée nulle part) | 0 (archivée) | 0 | ✅ |
| Couverture cœur métier | ~0 (aucun test métier) | 43 % | ≥ 60 % | ⚠️ |
| `except Exception:` larges | 67 | ~64 (B-15 partiel) | resserrés | ⚠️ |
| Taille EXE release | n/a (jamais buildé) | 24.4 MB | ≤ 100 MB | ✅ |
| Identité fenêtre + icône | défaut CTk | `ShutterstockAnalyzer v2.0.0…` + icône custom | conforme | ✅ |

### Top 5 risques résiduels

1. **Couverture 43 %** des modules cœur (cible 60 %). `MetadataReader` / `MetadataWriter` (15-17 %), `integration.py` (15 %) restent peu couverts car dépendent d'ExifTool / Ollama externes.
2. **B-15 partiel** : 64 `except Exception:` toujours en place. Risque d'avaler des erreurs métier silencieusement. Doit être resserré site par site selon l'intention.
3. **`pip-audit` / `bandit` / `mypy --strict`** non joués — pas de scan CVE ni typage strict.
4. **Pas de tests `analyze_batch_ai` end-to-end** : le pipeline IA est correct par construction (signatures alignées), mais aucun test ne mocke `VisionAnalyzer.analyze_batch` pour exercer la chaîne `init_ai → DB → run`.
5. **Vérification icône / barre de tâches en environnement réel** : automatisée best-effort (build flag + AppUserModelID). Validation visuelle Windows desktop nécessaire avant release publique.

## 2. AJUSTEMENTS DU PLAN (vs prompt original)

| Section prompt | Adaptation | Justification |
|---|---|---|
| `DOMAIN: Excel/Word/PDF/Dossiers/mixte` | Étendu à "Stock photo metadata IA" (JPG/PNG + EXIF/IPTC/XMP + Ollama) | Domaine réel du code |
| Phase D séparée (diagnostic) | Fusionnée dans la Phase B (matrice + anomalies) | Évite la duplication ; B.4 contient déjà la liste |
| `[project.scripts]` rebranchage | Suppression (pas de réimplémentation) | `python main.py` + EXE sont les vrais entry points |
| Stubs UI "coming soon" | (a) désactivés via `state="disabled"` + label explicite | Préserver UX honnête sans coût d'implémentation |
| `--light` build profile (existant v1) | Supprimé au profit de `release` (--windowed --noconsole) | Conformité spec audit ; un seul profil release suffit |
| Coverage > 60 % | Documenté à 43 % comme dette | Investissement test-engine réel (ExifTool/Ollama mocks) hors scope |

## 3. SYNTHÈSE PAR PHASE

| Phase | Livrable | Statut | Commit principal |
|---|---|---|---|
| A — Pré-audit | Branche + tag + baseline.json (mental) | ✓ | `9d0bc36` |
| B — Cartographie | Matrice UI→Backend (16 anomalies) | ✓ | (en chat) |
| C — Tests baseline | `BASELINE_TESTS.md` + 13 smoke + xfail B-16 | ✓ | `827a62a` |
| D — Diagnostic | Fusionné dans B | ✓ | (idem) |
| E — Correction | 17 bugs fermés (B-17/B-18 découverts pendant E) | ✓ | 7 commits `970f4f3` → `f87a150` |
| F — Validation | UI smoke headless + coverage HTML | ✓ | `7457822` |
| G — Packaging | `build.py` debug/release/all/clean + `BUILD_REPORT.md` + 2 EXE | ✓ | `fa33981` |
| H — Documentation | `TEST_PLAN.xlsx` + `CHECKLIST_IHM.html` | ✓ | (ce commit) |
| I — Boucle continue | Hors scope (pas de CI distant configuré) | ⚠️ | (recommandé) |

## 4. CHANGEMENTS APPLIQUÉS

13 commits, listés dans `JOURNAL_CORRECTIONS.md`. Refactors majeurs :

- **`main.py`** : `class App` sortie de `main()` vers le module-level. Imports hoist. Suppression du `load_modules()` thread (qui scellait les imports en scope local). Lambda B-18 corrigée par capture default-arg. Helper `resource_path` pour `sys._MEIPASS`. Title format conforme. AppUserModelID + iconbitmap branchés.
- **`src/modules/integration.py`** : 5 appels `add_audit_log` → `log_action`. 2 appels `update_file_status` → nouveau `set_file_flags`. `create_batch` reçoit `source_folder=""`. `complete_batch(batch_id, status="completed")`.
- **`src/modules/storage/database.py`** : nouveau `set_file_flags(file_path, has_metadata=None, has_ai_analysis=None)` (UPDATE partiel sans recopier hash/size).
- **`src/modules/models/metadata_models.py`** : `IPTCFields.from_dict` utilise `dataclasses.fields(cls)` au lieu de `hasattr` (le second rejetait les champs `default_factory=...`).
- **Archive `_archive/legacy_ui_v1/`** : `main_window.py` + 6 `page_*.py` + `advanced_window.py` + `sidebar.py` (~1100 LOC orphelins). README sur la procédure de réhabilitation.
- **`build.py`** : réécriture complète avec subcommands `debug | release | all | clean`. Trim hidden imports. Trim deps pyproject (4 deps fantômes retirées).

## 5. CHANGEMENTS RECOMMANDÉS NON APPLIQUÉS

| # | Recommandation | Effort | Impact |
|---|---|---|---|
| R1 | Couvrir `analyze_batch_ai` end-to-end avec mocks Ollama | 1-2 h | Sécurité majeure du pipeline IA |
| R2 | Resserrer les 64 `except Exception:` site par site | 2-3 h | Visibilité erreurs métier |
| R3 | Lancer `pip-audit` + `bandit -r .` + remédier HIGH/MEDIUM | 30 min | Posture sécurité |
| R4 | `mypy --strict` sur API publique de `integration.py` + `database.py` | 2-3 h | Détection bugs typage |
| R5 | Hooks pré-commit (`.pre-commit-config.yaml` : ruff, smoke pytest) | 30 min | Régression bloquée à la source |
| R6 | CI minimale GitHub Actions (`.github/workflows/ci.yml`) | 30 min | Régression bloquée en remote |
| R7 | Implémenter les 3 stubs (FTPS test, template editor, batch write) | 2-4 h chacun | Complétude fonctionnelle |
| R8 | `vulture` + nettoyage dead code restant après archivage | 30 min | Hygiène |
| R9 | Tester le dépouillement des modèles `vision_analyzer` (3 modèles supportés) | 1-2 h | Robustesse multi-modèle |
| R10 | Internationalisation FR (interface tutoiement) | 4-8 h | UX fr-FR si cible |

## 6. QUESTIONS OUVERTES

| Q | Choix par défaut appliqué | Confirmer ? |
|---|---|---|
| Working tree avant audit (commit ou stash ?) | Commit `chore: snapshot pre-audit v2.0 refactor` | ✓ |
| `PROJECT_NAME` final | `ShutterstockAnalyzer` (build.py) plutôt que `ShutterstockAI` (main.py-old) | ✓ |
| Version cible | `v2.0.0` partout | ✓ |
| `ALLOW_DELETE` | false (archivage `_archive/`) | ✓ |
| `ALLOW_RENAME` | true | ✓ |
| `ALLOW_UI_REFACTOR` | true (extraction `App` à module-level) | ✓ |
| FTPS upload — vraie implémentation ou stub | Stub désactivé | À implémenter en R7 |

## 7. DETTE TECHNIQUE RÉSIDUELLE (priorité décroissante)

1. **R1** Tests intégration `analyze_batch_ai`
2. **R3** `pip-audit` (CVE deps)
3. **R5 + R6** Pre-commit + CI
4. **R2** Reserrement `except`
5. **R7** Stubs IHM (FTPS test, template editor, batch write)
6. **R4** `mypy --strict`
7. **R9** Multi-modèles vision

## 8. ANNEXES

- [`BASELINE_TESTS.md`](BASELINE_TESTS.md)
- [`BUILD_REPORT.md`](BUILD_REPORT.md)
- [`JOURNAL_CORRECTIONS.md`](JOURNAL_CORRECTIONS.md)
- [`tests/TEST_PLAN.xlsx`](tests/TEST_PLAN.xlsx)
- [`tests/CHECKLIST_IHM.html`](tests/CHECKLIST_IHM.html)
- [`_archive/legacy_ui_v1/README.md`](_archive/legacy_ui_v1/README.md)
- Commits : `git log pre-audit-20260428..audit/20260428 --oneline`
