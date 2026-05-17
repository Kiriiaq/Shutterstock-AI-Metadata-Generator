# Build report — Phase G (UI tweaks 2026-05-16)

**Date du build courant** : **2026-05-16 21:40** (rebuild après UI tweaks Phase G)
**Date du build précédent** : 2026-05-14 19:23 (Phase F, archivé)
**Date du build initial v3** : 2026-05-08 09:04 UTC
**Branche** : `main`
**Commit HEAD** : `2d60f20` (fix(ui): Récursif inline + marges fenêtre + icône app sur toutes les modales)
**Builder** : PyInstaller 6.20.0 (Python 3.11.9, Windows 11 Home)
**Statut global** : ✅ **OK** (30/30 tests, debug + release ALIVE)

> Ce rapport remplace la version 2026-05-14 (commit `895bd84`). Le contenu précédent reste accessible via `git show 895bd84:BUILD_REPORT.md`.

---

## 1. Récapitulatif des étapes

| # | Étape | Résultat |
|---|---|---|
| 1 | Pré-vérifications git | ✅ sur `main`, HEAD = `2d60f20`, working tree clean après commit Phase G |
| 2 | Lint ruff (check + format) | ✅ `All checks passed!` · `41 files already formatted` |
| 3 | Tests pytest | ✅ **30/30 PASS** (inchangé : les changements Phase G sont purement visuels — Récursif inline, marges, icône modale) |
| 4 | Cleanup build/ + *.spec | ✅ `rm -rf build/ *.spec` avant chaque profil (bug PyInstaller 6.20 connu) |
| 5 | Build debug | ✅ `ShutterstockAnalyzer-debug.exe` — 24.5 MB |
| 6 | Build release | ✅ `ShutterstockAnalyzer.exe` — 24.5 MB |
| 7 | Smoke tests post-build (`audit/smoke_exe.py`) | ✅ debug + release **ALIVE** (~6 s), `UI mainloop starting` atteint |
| 8 | Rapport | ✅ ce document |

### Changements Phase G intégrés

1. **Sources panel** — la checkbox `Récursif` est désormais sur la
   même ligne que `Scanner` (au lieu d'une rangée dédiée). Gain de
   place vertical, les 3 contrôles qui pilotent le scan dossier sont
   alignés sur le même axe horizontal.
2. **Marges externes** — `SPACE_MD` → `SPACE_LG` sur les `padx`/`pady`
   des deux colonnes du workspace. Le contenu n'est plus collé au
   chrome de la fenêtre Windows (effet « scotché » quand
   l'Explorateur est ouvert à côté).
3. **Icône d'application sur toutes les modales** — `iconbitmap` posé
   via le nouveau helper `App._apply_modal_icon(modal)` sur les 4
   sites d'ouverture de `CTkToplevel` : `show_details`,
   `open_in_modal` (Configurer / Détail / Tout voir / Modifier),
   `_open_help` (F1), et `confirm()` dans `confirm_dialog.py`. Avant :
   icône Tcl par défaut (cadre blanc + point). Maintenant : `icone.ico`
   ShutterstockAnalyzer.

---

## 2. Artefacts produits

**Build courant (2026-05-16 21:40)** :

| Fichier | Taille | SHA-256 | Profil |
|---|---:|---|---|
| `dist/ShutterstockAnalyzer-debug.exe` | 25 725 712 B (24.53 MB) | `757ffc312f0c7cd21eefe00b650b46f26f8bac2de575aeaefd0c2fcf66acb4aa` | Debug — console visible, logs d'imports verbeux (`--debug=imports`) |
| `dist/ShutterstockAnalyzer.exe` | 25 721 509 B (24.53 MB) | `81be8b04cab5351bb667c08139961a43cc9c30ffcfc9c9a5b9b671fc61b13c8f` | Release / light — `--windowed --noconsole`, exclusions module, pas d'UPX |

### Hashes du build précédent (Phase F, 2026-05-14)
| Fichier | SHA-256 |
|---|---|
| `ShutterstockAnalyzer-debug.exe` | `4e6bb7b8945a9ce56a24c2702baba37308f626cebbef20c5a84b7f3ea8d00a5e` |
| `ShutterstockAnalyzer.exe` | `17ca720cfdcfb8ccdae8cff1f766282cda0a15847c37ffc9066bc61d082ec596` |

Les hashes diffèrent même quand le code est identique : PyInstaller embarque un timestamp dans le PE header → reproductibilité non bit-à-bit.

**Évolution du poids vs Phase F (2026-05-14)** :
- Debug : 25 720 151 B → 25 725 712 B = **+5 561 B (+0.02 %)** — overhead du helper `_apply_modal_icon` + commentaires Phase G
- Release : 25 716 108 B → 25 721 509 B = **+5 401 B (+0.02 %)** — idem

Négligeable, dans l'épaisseur du timestamp.

**Builds antérieurs disponibles via git :**

| Commit | Date | Notes |
|---|---|---|
| `2b55460` | 2026-05-11 22:06 | Rebuild on demand après correctifs UI |
| `7069953` | 2026-05-11 ~21 | Refonte UI v3 — premier build officiel |

**Évolution du poids vs précédent build (2026-05-11)** :
- Debug : 25 766 834 B → 25 720 151 B = **−46 KB (−0.18 %)** (Lot E retrait des 4 vues mortes)
- Release : 25 762 256 B → 25 716 108 B = **−46 KB (−0.18 %)**

Les 4 vues archivées (`home.py`, `editor.py`, `sources.py`, `analyze.py`, ~887 lignes au total) sont sorties du `--add-data app/`. La poche Phase F (Lots A à D, +500 lignes dans `workspace.py` / `app.py`) compense partiellement le retrait, d'où le delta net légèrement négatif.

Les hashes SHA-256 diffèrent entre les builds **même quand le code est identique** : PyInstaller embarque un timestamp dans le PE header → reproductibilité non bit-à-bit.

---

## 3. Configuration PyInstaller

Identique au build précédent — voir `build.py` :

- **`--onefile`** — un seul EXE auto-extractible
- **`--noupx`** — UPX désactivé (déclenche des faux-positifs antivirus sous Windows)
- **`--icon assets/icons/icone.ico`**
- **Hidden imports** : `customtkinter`, `darkdetect`, `PIL`
- **Modules exclus** : `scipy`, `numpy`, `pandas`, `matplotlib`, `cv2`, `whisper`, `PyPDF2`, `pdfplumber`, `fitz`, `reportlab`, `win32com`, `piexif`, `pydantic`, `ollama`, `unittest`, `pytest`, `idlelib`, `tkinter.test`, … (liste complète dans [build.py](build.py))
- **Add-data** : `assets/`, `src/`, `app/` (résolus à l'exécution via `_resource_path` qui honore `sys._MEIPASS`)
- **Profil debug** : `--console --debug=imports`
- **Profil release** : `--windowed --noconsole`

---

## 4. Smoke tests post-build (`audit/smoke_exe.py`)

Chaque EXE est lancé depuis un **tempdir hermétique** (no `VIRTUAL_ENV`, no `PYTHONPATH`), maintenu actif 6 s puis `taskkill /F /T /PID` (kill du process tree — nécessaire car le bootloader PyInstaller `--onefile` spawn un enfant).

### 4.1 Profil debug

```
status    : ALIVE
hold_time : 6.11 s
log tail  :
    import 'app.views.validate' # <pyimod02_importers.PyiFrozenLoader…>
    PyiFrozenFinder(…\app\views): find_spec: called with fullname='app.views.workspace', target='app.views.workspace'
    PyiFrozenFinder(…\app\views): find_spec: found 'app.views.workspace' in PYZ as 'app.views.workspace', typecode=0
    import 'app.views.workspace' # <pyimod02_importers.PyiFrozenLoader…>
    21:40:44 [INFO] app.config.theme: UI monospace font resolved to: Cascadia Mono
    21:40:44 [INFO] ShutterstockAnalyzer: UI mainloop starting
```

### 4.2 Profil release / light

```
status    : ALIVE
hold_time : 6.11 s
log tail  :
    21:40:50 [INFO] src.modules.storage.database: Database initialized: C:\Users\Emmanuel Grolleau\.shutterstock_ai\shutterstock_ai.db
    21:40:50 [INFO] src.modules.workers.worker_pool: WorkerPool initialized with 4 workers (processes=False)
    21:40:50 [INFO] app.config.theme: UI proportional font resolved to: Segoe UI
    21:40:51 [INFO] app.config.theme: UI monospace font resolved to: Cascadia Mono
    21:40:51 [INFO] ShutterstockAnalyzer: UI mainloop starting
```

### 4.3 Fonctionnalités critiques attestées

Au point `UI mainloop starting`, le shell a :

1. Importé toute la chaîne `app.views.*` — **workspace** (avec les correctifs Phase F : nouveaux boutons Sources, `_refresh_action_states`, barre de progression titrée, `focus_panel`, garde-fou double-clic, `_sync_sources_state`) + 4 vues modales (settings, audit, ai_control, validate).
2. Initialisé la base SQLite locale.
3. Démarré le `WorkerPool` (4 workers).
4. Résolu les polices Segoe UI + Cascadia Mono.
5. Construit le `WorkspaceView` avec ses 7 panneaux (Sources & tri, Édition IPTC, Analyse IA, Modèle IA, Validation, Historique, Paramètres).
6. **Lié les Ctrl+1..5** via `bind_all` (Lot A) — disponibles depuis la pression de la première touche.

Pas de DLL manquante, pas d'exception silencieuse dans `stderr`.

---

## 5. Correctifs Phase F intégrés à ce build

Reprise synthétique des cinq lots commité dans `e8e8fe3` :

| Lot | Anomalies tests couvertes | Surface |
|---|---|---|
| **A** | T-016..T-020 (Ctrl+1..5) | `app/config/shortcuts.py`, `app/app.py`, `app/i18n/fr.py` |
| **B** | T-030..T-033 (modèle Sources) | `app/views/workspace.py` — panneau Sources (4 nouveaux boutons + compteur permanent + modèle incrémental) |
| **C** | T-034, T-035 (visibilité boutons + Démarrer state initial) | `app/views/workspace.py`, `app/components/topbar.py` |
| **D** | T-036 (progression), T-023 (Escape), T-221 (double-clic) | `app/views/workspace.py`, `app/app.py` |
| **E** | D-02, D-03 (code mort + i18n) | `_archive/legacy_ui_v3_views/` (4 fichiers archivés), `app/i18n/fr.py` |

Pour le détail des changements et la corrélation avec chaque ID de test, voir le `RAPPORT_AUDIT.md` ou le commit `e8e8fe3`.

---

## 6. Warnings PyInstaller à surveiller

Aucun warning bloquant. PyInstaller a tracé :
- `INFO: Building PYZ`, `INFO: Building PKG`, `INFO: Building EXE` — chaîne nominale.
- L'icône `assets/icons/icone.ico` a bien été embarquée.
- Les `--add-data` pour `assets/`, `src/`, `app/` ont été résolus.

À surveiller sur livraison :
- **Faux-positifs antivirus** (Defender SmartScreen) sur le bootloader PyInstaller — comportement attendu pour un onefile non signé. Pas de signature de code dans cette release.
- **Démarrage à froid ~3-5 s** sous Windows (premier lancement) à cause de la décompression `_MEIxxxxxx` dans `%TEMP%`. À chaud (~1 s).

---

## 7. Sortie `git log --oneline -5`

```
2d60f20 fix(ui): Récursif inline + marges fenêtre + icône app sur toutes les modales
895bd84 build: rebuild debug + light EXEs après correctifs Phase F (2026-05-14 19:23)
e8e8fe3 fix(audit): correctifs Phase F (lots A à E) — bugs T-016..T-036 + T-023
2b55460 build: rebuild debug + light EXEs (2026-05-11 22:06)
7069953 build: refonte UI v3 — debug + release EXEs + build_report
```

---

## 8. Livrables

1. ✅ `dist/ShutterstockAnalyzer-debug.exe` (24.5 MB) — EXE debug avec console
2. ✅ `dist/ShutterstockAnalyzer.exe` (24.5 MB) — EXE light release sans console
3. ✅ `BUILD_REPORT.md` — ce document
4. ✅ Commit `e8e8fe3` sur `main` — correctifs Phase F + tests + archivage code mort
5. ✅ `audit/smoke_exe.py` — script de smoke test rejouable (`python audit/smoke_exe.py`)
6. ✅ 30/30 tests pytest passants, `ruff check + format` propres

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
