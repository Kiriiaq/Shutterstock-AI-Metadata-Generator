# Build report — Phase G+3 (bouton « Vider l'historique » 2026-05-19)

**Date du build courant** : **2026-05-19 13:26** (rebuild après ajout bouton Vider l'historique)
**Date du build précédent** : 2026-05-19 12:32 (Phase G+2, archivé)
**Date du build initial v3** : 2026-05-08 09:04 UTC
**Branche** : `main`
**Commit HEAD** : `94bf74a` (feat(history): bouton « Vider » à droite d'« Exporter »)
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

### Changements Phase G intégrés (cumulés sur 2 commits)

**Commit `2d60f20` (2026-05-16)** :

1. **Sources panel** — la checkbox `Récursif` est désormais sur la
   même ligne que `Scanner` (au lieu d'une rangée dédiée). Gain de
   place vertical, les 3 contrôles qui pilotent le scan dossier sont
   alignés sur le même axe horizontal.
2. **Marges externes du Workspace** — `SPACE_MD` → `SPACE_LG` sur les
   `padx`/`pady` des deux colonnes. Le contenu n'est plus collé au
   chrome de la fenêtre Windows (effet « scotché » quand
   l'Explorateur est ouvert à côté).
3. **Icône d'application sur toutes les modales** — `iconbitmap` posé
   via le nouveau helper `App._apply_modal_icon(modal)` sur les 4
   sites d'ouverture de `CTkToplevel` : `show_details`,
   `open_in_modal` (Configurer / Détail / Tout voir / Modifier),
   `_open_help` (F1), et `confirm()` dans `confirm_dialog.py`. Avant :
   icône Tcl par défaut (cadre blanc + point). Maintenant : `icone.ico`
   ShutterstockAnalyzer.

**Commit `a636b3c` (2026-05-17)** :

4. **Topbar — marges uniformes 16 px sur les 4 côtés**. Avant : 16 px
   à gauche du titre, 12 px à droite des boutons, et seulement ~6 px
   en haut/bas (asymétrie visible). Après : `HEIGHT` passe de 44 à
   64 px, `pady=SPACE_LG` (16 px) sur les 3 éléments grid (titre,
   strip santé, boutons d'action), `padx` extérieur harmonisé à
   `SPACE_LG` (16) gauche et droite, et `grid_rowconfigure(0, weight=1)`
   ajouté pour permettre le centrage vertical naturel. Cleanup :
   retrait de l'import `SPACE_MD` devenu inutilisé.

**Commit `9a255d3` (2026-05-18) — Phase G+1** :

5. **Topbar — 3ᵉ chip santé "Ollama"** (En ligne / Hors ligne /
   Non init. / —). État caché dans `App._ollama_health`, mis à jour
   par `WorkspaceView._refresh_dynamic_worker` qui tourne déjà toutes
   les 5 s en background → pas d'appel HTTP synchrone dans
   `topbar.refresh_health()`. Nouvelle méthode publique
   `App.set_ollama_health(label, kind)`.

6. **Panneau MODÈLE IA — bouton "▶ Démarrer Ollama"**. Lance
   `subprocess.Popen([ollama, "serve"])` en process détaché
   (`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` sous Windows pour
   que le serveur survive à la fermeture de l'app). Recherche l'exe
   dans le PATH puis dans `%LOCALAPPDATA%\\Programs\\Ollama\\` et
   `C:/Program Files/Ollama/`. Toast info au démarrage, refresh
   dynamique programmé à 2,5 s pour basculer la chip topbar sur
   "En ligne". Toast d'erreur si l'exe est introuvable.

7. **Fix bug rétrécissement colonne droite à chaque toggle thème**.
   CTk `set_appearance_mode` recalcule le DPI scaling et grignote
   quelques pixels du `CTkScrollableFrame` interne à chaque appel
   (effet cumulatif visible après quelques cycles light/dark/system).
   Workaround : `App._toggle_theme` capture `self.geometry()` avant
   le toggle et la restaure ~50 ms après (`self.after(50, …)`).

8. **Panneau Sources — compteur déplacé inline avec les boutons
   d'action**. La rangée "opts" (qui ne contenait que le compteur)
   est supprimée ; `_sources_status` est désormais packed
   `side="right"` dans la même rangée que `+ Fichiers / + Dossier /
   Supprimer / Vider`. Le format change pour matcher la demande
   utilisateur : « nombre de fichiers : N » (avec
   « · M sélectionné(s) » quand une sélection est active). Le
   `DataTable` remonte d'une rangée (row 4 → row 3).

**Commit `582c5ec` (2026-05-19) — Phase G+2** :

9. **Panneau MODÈLE IA — zone de feedback sortie sur sa propre
   rangée** sous les boutons. Avant : `_model_test_msg` était
   packed `side="right"` à droite de `Démarrer Ollama / Tester /
   Configurer`, ce qui tronquait les messages longs sur les
   petites fenêtres. Maintenant : grille row=3 dédiée alignée à
   gauche avec `wraplength=380` pour les messages multi-lignes.

10. **Toggle thème — bug du rétrécissement cumulatif enfin
    corrigé**. Le workaround Phase G+1 (1 seul `after(50, …)`) ne
    suffisait pas : CTk continue à recalculer plusieurs frames
    après `set_appearance_mode`. Nouvelle approche multi-passes :
    `geometry()` restaurée à 3 instants (50, 150, 350 ms),
    `grid_columnconfigure` ré-appliquées sur le workspace
    (weight 3 / 2 + minsize 320), `update_idletasks()` final.

11. **ExifTool — plus de console qui flashe à chaque scan**.
    Nouveau helper `src/utils/subprocess_helper.py` qui expose
    `SUBPROCESS_NO_WINDOW = {"creationflags": CREATE_NO_WINDOW}`
    sur Windows, no-op sur POSIX. Splaté dans les 8 appels
    `subprocess.run` du pipeline metadata (4 reader + 4 writer).
    Plus aucune fenêtre `exiftool.exe` qui apparaît pendant un
    scan dossier ou une écriture batch.

**Commit `94bf74a` (2026-05-19) — Phase G+3** :

12. **Bouton « Vider l'historique » à droite d'« Exporter… »**.
    Présent dans les 2 vues qui exposent un bouton Exporter :
    - Panneau HISTORIQUE du workspace (visible par défaut)
    - Modale Audit (« Tout voir… »)
    Couleurs error (rouge) pour signaler l'effet destructif,
    `confirm_destructive` obligatoire avant tout DELETE en base
    (texte de confirmation : « Cette action supprime DÉFINITIVEMENT
    toutes les entrées du journal… »). Backend : nouvelle méthode
    `Database.clear_audit_log()` qui retourne le compte avant DELETE
    + commit. Toast feedback « N entrée(s) supprimée(s). » + refresh
    immédiat de l'aperçu (pas d'attente du poll 5 s).

---

## 2. Artefacts produits

**Build courant (2026-05-19 13:26)** :

| Fichier | Taille | SHA-256 | Profil |
|---|---:|---|---|
| `dist/ShutterstockAnalyzer-debug.exe` | 25 759 719 B (24.56 MB) | `f975595ab278ab26a34568d05c7c58e455f7d5d9f9d6e6a803e0a501fb29ef2d` | Debug — console visible, logs d'imports verbeux (`--debug=imports`) |
| `dist/ShutterstockAnalyzer.exe` | 25 755 005 B (24.56 MB) | `7573befcb0de34d34f998130ef0827b0dcf6b30b37189596e4909acb004edbf5` | Release / light — `--windowed --noconsole`, exclusions module, pas d'UPX |

### Hashes du build précédent (Phase G+2, 2026-05-19 12:32)
| Fichier | SHA-256 |
|---|---|
| `ShutterstockAnalyzer-debug.exe` | `c2d10532ce45ab1bbddbd29409eadf1a3c1b96449a0c8319517f8a9cd91e2220` |
| `ShutterstockAnalyzer.exe` | `580419b69c3e988e273da62b764cd7d56d44701003174fded91626f91d2a2519` |

**Évolution du poids vs build du 2026-05-19 12:32 (Phase G+2)** :
- Debug : 25 754 277 B → 25 759 719 B = **+5 442 B (+0.02 %)** — backend `clear_audit_log()` + bouton + handler `_history_clear/_clear` dans 2 vues
- Release : 25 749 561 B → 25 755 005 B = **+5 444 B (+0.02 %)** — idem

Variation négligeable.

### Hashes des builds précédents
| Build | Fichier | SHA-256 |
|---|---|---|
| Phase G initial (2026-05-16) | `ShutterstockAnalyzer-debug.exe` | `757ffc312f0c7cd21eefe00b650b46f26f8bac2de575aeaefd0c2fcf66acb4aa` |
| Phase G initial (2026-05-16) | `ShutterstockAnalyzer.exe` | `81be8b04cab5351bb667c08139961a43cc9c30ffcfc9c9a5b9b671fc61b13c8f` |
| Phase F (2026-05-14) | `ShutterstockAnalyzer-debug.exe` | `4e6bb7b8945a9ce56a24c2702baba37308f626cebbef20c5a84b7f3ea8d00a5e` |
| Phase F (2026-05-14) | `ShutterstockAnalyzer.exe` | `17ca720cfdcfb8ccdae8cff1f766282cda0a15847c37ffc9066bc61d082ec596` |

Les hashes diffèrent même quand le code est identique : PyInstaller embarque un timestamp dans le PE header → reproductibilité non bit-à-bit.

**Évolution du poids vs Phase G initial (2026-05-16)** :
- Debug : 25 725 712 B → 25 724 550 B = **−1 162 B (−0.005 %)** — variation négligeable (timestamp + retrait import SPACE_MD)
- Release : 25 721 509 B → 25 723 268 B = **+1 759 B (+0.007 %)** — idem

Dans l'épaisseur du timestamp PE.

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
log tail  : (bootloader Tk / PyiFrozenFinder, démarrage propre)
```

### 4.2 Profil release / light

```
status    : ALIVE
hold_time : 6.02 s
log tail  :
    13:26:38 [INFO] src.modules.storage.database: Database initialized
    13:26:38 [INFO] src.modules.workers.worker_pool: WorkerPool initialized with 4 workers
    13:26:38 [INFO] app.config.theme: UI proportional font resolved to: Segoe UI
    13:26:39 [INFO] app.config.theme: UI monospace font resolved to: Cascadia Mono
    13:26:39 [INFO] ShutterstockAnalyzer: UI mainloop starting
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

## 7. Sortie `git log --oneline -7`

```
582c5ec fix(ui): statut sous boutons + toggle thème robuste + ExifTool sans console
9a255d3 feat(ui+ollama): chip topbar Ollama + bouton démarrer + fix toggle thème + compteur inline
c5849d0 build: rebuild debug + release EXEs après marges topbar (2026-05-17 21:48)
a636b3c fix(ui): topbar — marges uniformes 16 px sur les 4 côtés (Phase G suite)
d19249d build: rebuild debug + release EXEs après UI tweaks Phase G (2026-05-16 21:40)
2d60f20 fix(ui): Récursif inline + marges fenêtre + icône app sur toutes les modales
895bd84 build: rebuild debug + light EXEs après correctifs Phase F (2026-05-14 19:23)
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
