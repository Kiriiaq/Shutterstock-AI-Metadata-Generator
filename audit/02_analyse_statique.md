# Phase 2 — Analyse statique

État avant corrections : ruff `--select=E,F,W,B,S` reportait **159 anomalies**.

## Résultats par catégorie

| Code | Compte | Sévérité | Action |
|---|---|---|---|
| **B904** | 9 | Majeur | Chaînage d'exception manquant (`raise X(...) from e`). **Corrigé** Phase 3. |
| **B033** | 1 | Mineur | Doublon `.rw2` dans `RAW_EXTENSIONS` (Panasonic + Leica). **Corrigé** Phase 3. |
| **B023** | 1 | Majeur | Closure `stage_progress` capture la variable de boucle `stage_name`. **Corrigé** Phase 3 (default-arg trick). |
| **B007** | 1 | Mineur | Loop var `handler` non utilisée → renommée `_handler`. **Corrigé** Phase 3. |
| **S101** | 99 | Acceptée | `assert` dans tests — légitime, pas de correction. |
| **S110** | 14 | Variable | `try/except/pass` — la moitié dans `system_panel.py` (archivé en Phase 3) ; le reste sur des `widget.configure()` où ignorer une `tk.TclError` est volontaire (widget en cours de destruction). Pas de correction systématique. |
| **S108** | 11 | Acceptée | Faux positifs : `/tmp/foo` apparaît comme exemple dans des tests. |
| **S603** | 10 | Acceptée | `subprocess.run([list, ...])` — appel sans `shell=True`, sécurisé. |
| **E501** | 11 | Cosmétique | Lignes > 120 — toutes dans des f-strings de log. Tolérable. |
| **B112 / S608** | 2 | Mineur | `try/except/continue` × 1 + 1 SQL en f-string (LIMIT/OFFSET) — accepté car valeurs sont int. |

État après Phase 3 : `ruff check app/ src/ main.py build.py` → **clean** (les S101 / S108 / S603 / S110 résiduels sont volontaires et documentés).

## py_compile

`python -m compileall -q app/ src/ tests/ main.py build.py` → **0 erreur de syntaxe**.

## Recherches manuelles

Recherche `TODO|FIXME|XXX|NotImplementedError|coming soon` :
- `app/` : **0 occurrence** dans le code actif.
- `src/` : 8 occurrences `pass` toutes dans des **classes d'exception** (corps vide,
  `class OllamaTimeoutError(OllamaError): pass`) — légitimes, pas des stubs.
- `coming soon` : 0 occurrence dans le code actif (les 3 stubs « bientôt » de
  l'audit précédent ont disparu après le swap UI v3).

## Code mort

`vulture` non installé dans l'env audit, mais inspection manuelle des imports :

| Fichier | Statut | Action |
|---|---|---|
| `app/components/sidebar.py` | Aucun import dans `app/` ou `src/` actifs | Archivé Phase 3 → `_archive/legacy_ui_v3_predense/` |
| `app/components/system_panel.py` | idem | idem |
| `app/components/command_palette.py` | idem | idem |
| `app/components/context_panel.py` | idem | idem |

## Sécurité (bandit-équivalent)

- **Pas de secret hardcodé** (vérifié à l'audit précédent, comportement préservé).
- **Pas de `eval` / `exec`** dans `app/` ni `src/`.
- **Subprocess** utilisé pour ExifTool : args en liste, jamais `shell=True`. ✓
- **SQL** : un seul appel `f"... LIMIT ? OFFSET ?"` dans `database.get_audit_logs` — les paramètres sont liés via `?`, l'injection n'est pas possible.

## Verdict avant corrections

| Niveau | Compte | Statut |
|---|---|---|
| BLOQUANT | 0 | — |
| MAJEUR | 11 (9 B904 + B023 + 1 stub UI virtuel pour Esc/cancel) | Corrigés Phase 3 |
| MINEUR | 14 (S110 légitime, B033, B007, E501, B112) | Tolérés ou corrigés |
| COSMÉTIQUE | 11 (E501) | Tolérés (logging) |
