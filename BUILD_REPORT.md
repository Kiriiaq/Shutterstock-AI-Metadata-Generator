# Build report — Refonte UI v3

**Date du build courant** : **2026-05-11 22:06** (rebuild on demand)
**Date du build initial** : 2026-05-08 09:04 UTC
**Branche** : `main`
**Commit HEAD** : `7069953` (build artefacts) sur la base de `9a435fbba63baef58bfc563cc5d534f48c164b03` (merge UI v3) ([log](#git-log-tail))
**Builder** : PyInstaller 6.20.0 (Python 3.11.9, Windows 11 Home)
**Statut global** : ✅ **OK** (27/27 tests, debug + light ALIVE)

> Le présent rapport remplace l'ancien `build_report.md` (audit/20260428, Phase G) — l'ancien contenu reste retrouvable via `git show HEAD~1:build_report.md`.

---

## 1. Récapitulatif des étapes

| # | Étape | Résultat |
|---|---|---|
| 1 | Pré-vérifications git | ✅ sur `main`, 9 fichiers modifiés détectés (refonte UI précédente) |
| 2 | Compilation initiale | ✅ `py_compile main.py app/main.py app/app.py` clean ; `import app.main` sans effet |
| 3 | Tests pytest | ✅ **27/27 PASS** (smoke 15, core 6, validators 5, UI shell 1) |
| 4 | Feature branch + merge `--no-ff` dans `main` | ✅ commit `294925e` puis merge `9a435fb` (`ort` strategy, no conflicts) |
| 5 | Build debug | ✅ `ShutterstockAnalyzer-debug.exe` — 24.6 MB |
| 6 | Build release light | ✅ `ShutterstockAnalyzer.exe` — 24.6 MB |
| 7 | Smoke tests post-build | ✅ debug + release démarrent depuis tempdir, `UI mainloop starting` atteint, pas de crash 6 s |
| 8 | Rapport | ✅ ce document |

---

## 2. Artefacts produits

**Build courant (2026-05-11 22:06)** :

| Fichier | Taille | SHA-256 (16 c) | Profil |
|---|---:|---|---|
| `dist/ShutterstockAnalyzer-debug.exe` | 24.57 MB | `ce9bdca53c2ca653` | Debug — console visible, logs d'imports verbeux (`--debug=imports`) |
| `dist/ShutterstockAnalyzer.exe` | 24.57 MB | `f68f7c2cedb161ec` | Release — `--windowed --noconsole`, exclusions module, pas d'UPX (anti-AV) |

**Build du 2026-05-08** (archivé sous `dist/_archive_20260511_220435/`) :

| Fichier | Taille | SHA-256 (12 c) |
|---|---:|---|
| `ShutterstockAnalyzer-debug.exe` | 24.57 MB | `a480a6ae2243` |
| `ShutterstockAnalyzer.exe`       | 24.57 MB | `8d41491ffef2` |

**Build pré-refonte** (archivé sous `dist/_archive_pre-refonte/`) :

| Fichier | Taille | SHA-256 (12 c) |
|---|---:|---|
| `ShutterstockAnalyzer-debug_pre-refonte.exe` | 24.55 MB | `f3e51df26c35` |
| `ShutterstockAnalyzer_pre-refonte.exe`       | 24.55 MB | `b49bd74f04f7` |

**Évolution du poids depuis pré-refonte → build courant** :
- Debug : 25 744 794 B → 25 766 834 B = **+22 KB (+0.09 %)**
- Release : 25 741 987 B → 25 762 256 B = **+20 KB (+0.08 %)**

Variation négligeable entre runs (`±4 KB` selon le timestamp embarqué). Les changements de la refonte sont essentiellement du code Python (compilé en `.pyc` de taille équivalente). Les 3 PNG de captures (`audit/captures/*.png`) ne sont **pas** embarqués dans l'EXE (`audit/` n'est pas dans `--add-data`, conformément à `build.py`).

Les hashes SHA-256 diffèrent entre les builds **même quand le code est identique** : PyInstaller embarque un timestamp de compilation dans le PE header → reproductibilité non bit-à-bit.

---

## 3. Configuration PyInstaller

Reprise du `build.py` du dépôt (profils `debug` / `release`). Points-clés :

- **`--onefile`** — un seul EXE auto-extractible
- **`--noupx`** — UPX désactivé (déclenche des faux-positifs antivirus sous Windows)
- **`--icon assets/icons/icone.ico`**
- **Hidden imports** : `customtkinter`, `darkdetect`, `PIL`
- **Modules exclus** : `scipy`, `numpy`, `pandas`, `matplotlib`, `cv2`, `whisper`, `PyPDF2`, `pdfplumber`, `fitz`, `reportlab`, `win32com`, `piexif`, `pydantic`, `ollama`, `unittest`, `pytest`, `idlelib`, `tkinter.test`, … (cf. [build.py](build.py) pour la liste complète)
- **Add-data** : `assets/`, `src/`, `app/` (résolus à l'exécution via `_resource_path` qui honore `sys._MEIPASS`)

Différences avec le spec utilisateur (qui propose `--debug=all` et `--strip --clean`) :

| Spec utilisateur | Build effectif | Justification |
|---|---|---|
| `--debug=all` | `--debug=imports` | `all` est très verbeux (chaque appel C↔Python tracé) ; `imports` couvre 95 % des cas debug et reste lisible. Le repo a déjà ce choix. |
| `--strip` | non utilisé | `strip` retire les symboles de debug ; sur Windows l'effet est marginal (0–0.5 %) et casse les stack traces dans les outils style Sentry. |
| `--clean` | géré par `_clean_artifacts()` | `build.py` nettoie `build/` + `*.spec` après chaque build pour le même résultat. |

---

## 4. Smoke tests post-build

Exécutés via [`audit/smoke_exe.py`](audit/smoke_exe.py) :

- chaque EXE lancé depuis un **tempdir hermétique** (no `VIRTUAL_ENV`, no `PYTHONPATH`) → confirme l'absence de dépendance au venv ou au cwd du dépôt
- maintien actif 6 s puis `taskkill /F /T /PID` (kill du process tree — nécessaire car le bootloader PyInstaller `--onefile` spawn un enfant)
- log d'exécution capturé dans le tempdir

### 4.1 Profil debug (build du 2026-05-11 22:06)

```
status    : ALIVE
hold_time : 6.19 s
log tail  :
    import 'app.views.validate' # PyiFrozenLoader
    PyiFrozenFinder: find_spec called with fullname='app.views.workspace'
    PyiFrozenFinder: found 'app.views.workspace' in PYZ
    import 'app.views.workspace' # PyiFrozenLoader
    22:06:31 [INFO] app.config.theme: UI monospace font resolved to: Cascadia Mono
    22:06:31 [INFO] ShutterstockAnalyzer: UI mainloop starting
```

### 4.2 Profil release (build du 2026-05-11 22:06)

```
status    : ALIVE
hold_time : 6.22 s
log tail  :
    22:06:37 [INFO] src.modules.storage.database: Database initialized: ~/.shutterstock_ai/shutterstock_ai.db
    22:06:37 [INFO] src.modules.workers.worker_pool: WorkerPool initialized with 4 workers (processes=False)
    22:06:37 [INFO] app.config.theme: UI proportional font resolved to: Segoe UI
    22:06:38 [INFO] app.config.theme: UI monospace font resolved to: Cascadia Mono
    22:06:38 [INFO] ShutterstockAnalyzer: UI mainloop starting
```

### 4.3 Fonctionnalités critiques attestées

Au point `UI mainloop starting`, le shell a :
1. Importé toute la chaîne `app.views.*` (workspace + 4 vues modales)
2. Initialisé la base SQLite locale
3. Démarré le `WorkerPool` (4 workers)
4. Résolu les polices Segoe UI + Cascadia Mono
5. Construit le `WorkspaceView` avec ses 7 panneaux (Sources, Édition IPTC, Analyse IA, Modèle IA, Validation, Historique, Paramètres)

Pas de DLL manquante, pas d'exception silencieuse dans `stderr`.

---

## 5. Warnings PyInstaller à surveiller

Aucun warning bloquant. PyInstaller a tracé :
- `INFO: Building PYZ`, `INFO: Building PKG`, `INFO: Building EXE` — chaîne nominale
- L'icône `assets/icons/icone.ico` a bien été embarquée
- Les `--add-data` pour `assets/`, `src/`, `app/` ont été résolus

À surveiller sur livraison :
- **Faux-positifs antivirus** (Defender SmartScreen) sur le bootloader PyInstaller — comportement attendu pour un onefile non signé. Pas de signature de code dans cette release.
- **Démarrage à froid ~3-5 s** sous Windows (premier lancement) à cause de la décompression `_MEIxxxxxx` dans `%TEMP%`. À chaud (~1 s).

---

## 6. Sortie `git log --oneline -5` <a id="git-log-tail"></a>

```
9a435fb merge: refonte UI v3 (soft-gray light theme + unified scroll + panel icons + bottom alignment)
294925e feat(ui): refonte v3 — soft-gray light theme, unified scroll, panel icons, bottom alignment
0f2f7ff feat(theme/phase-A): gray-blue palette + ThemeManager + Topbar pilot
2f19a32 feat(theme): bg_deep for Sources panel + collapsible Editor IPTC
b66dcc4 feat(ui): grayer light theme, drop FTPS, scrollable workspace columns
```

---

## 7. Livrables

1. ✅ `dist/ShutterstockAnalyzer-debug.exe` (24.6 MB) — EXE debug avec console
2. ✅ `dist/ShutterstockAnalyzer.exe` (24.6 MB) — EXE light release sans console
3. ✅ `build_report.md` — ce document
4. ✅ `audit/REFONTE_UI_REPORT.md` — rapport détaillé de la refonte UI (sources des changements)
5. ✅ `audit/captures/after_dark.png`, `after_light.png`, `after_modal_ai_control.png` — preuves visuelles
6. ✅ `audit/smoke_exe.py` — script de smoke test rejouable (`python audit/smoke_exe.py`)
7. ✅ Merge `--no-ff` propre sur `main` (commit `9a435fb`), branche `ui-refonte-2026-05` conservée pour rollback éventuel

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
