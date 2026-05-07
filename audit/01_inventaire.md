# Phase 1 — Inventaire & cartographie

Date : 2026-05-07. Branche : `main`. HEAD avant audit : `88b33d1`.

## Stack

| Couche | Technologie | Notes |
|---|---|---|
| Langage | Python 3.11+ | `requires-python = ">=3.11"` |
| UI | customtkinter 5.2 + tkinter/ttk + Pillow | aucune autre dépendance UI |
| Backend | SQLite (stdlib) + ExifTool (binaire externe) + Ollama HTTP | |
| Threading | `threading` stdlib + `concurrent.futures` (workers) | |
| Persistance prefs | JSON sous `%APPDATA%/ShutterstockAnalyzer/` | |
| Lint / format | ruff 0.4+ | configuré dans `pyproject.toml` |
| Tests | pytest 9.0+ | dossier `tests/` |
| Build EXE | PyInstaller via `build.py` (subcommands debug / release / all / clean) | onefile |

## Arborescence active

```
ShutterstockAnalyzer/
├── main.py                  # 5 lignes → app.main:main
├── build.py                 # PyInstaller wrapper
├── pyproject.toml           # ruff + pytest + deps
├── README.md docs/          # documentation
├── app/                     # UI v3 — CTk + stdlib
│   ├── main.py app.py
│   ├── config/ {theme, shortcuts}.py
│   ├── core/ {events, state, navigation}.py
│   ├── components/          # 6 composants actifs (palette/sidebar/system_panel/context_panel archivés)
│   │   {confirm_dialog, data_table, empty_state, form_field, toast, tooltip, topbar}
│   ├── views/               # 9 vues : workspace + 8 vues métier
│   ├── i18n/fr.py
│   └── utils/formatters.py
├── src/                     # Backend (intouché par la couche UI)
│   ├── core/ {params, config_manager, logger}.py
│   ├── modules/             # ai, engines, models, storage, workers
│   │   └── integration.py   # façade ShutterstockAIv2
│   └── utils/ {validators, file_utils, splash_screen}.py
├── tests/                   # 27 tests (smoke, unit, ui)
└── _archive/                # legacy_ui_v1/, legacy_ui_v2/, legacy_ui_v3_predense/
```

72 fichiers `.py` actifs (76 - 4 archivés en Phase 3 ci-dessous).

## Point d'entrée

```
main.py (root)  →  app.main.main()
                       │
                       ├── apply_theme(load_theme_pref())   # défaut: light
                       ├── ShutterstockAIv2()               # façade backend
                       └── App(api=…).mainloop()            # ctk.CTk
```

`build.py` produit `dist/ShutterstockAnalyzer.exe` (release, 24.5 MB) et
`dist/ShutterstockAnalyzer-debug.exe` (debug, 24.5 MB) avec `main.py`
comme entry-point PyInstaller.

## Cartographie UI : matrice Panneau × Fonctionnalité × Statut

### Atelier (vue principale, `app/views/workspace.py`)

Layout : 2 colonnes — gauche (production, 3 panneaux) + droite (système, 5 panneaux).

| # | Panneau | Indicateurs visibles | Actions | Détail (modal) | Statut |
|---|---|---|---|---|---|
| 1 | **Sources & tri** | dossier · récursif · compteur images / dossier | Parcourir · Scanner · Sélection multi-clic | — | OK |
| 2 | **Édition IPTC** | nom fichier sélectionné | 5 champs (titre / description / mots-clés / auteur / copyright) + Lire / Écrire / Effacer + statut | — | OK |
| 3 | **Analyse IA** | n images sélectionnées · barre progression · status (n/total) · live tail | Démarrer · Arrêter (Esc aussi) | — | OK |
| 4 | **Modèle IA** | URL · ● statut Ollama · modèle courant · message test | Tester | `Configurer…` → `AIControlView` | OK |
| 5 | **Validation** | n images · n conformes · n à corriger · première anomalie | Lancer | `Détail…` → `ValidateView` | OK |
| 6 | **Historique** | n ops 24h · n erreurs · 5 dernières lignes | Exporter (JSON / CSV) | `Tout voir…` → `AuditView` | OK |
| 7 | **Paramètres** | 6 valeurs clés : workers / batch / modèle / backup / IPTC / XMP | — | `Modifier…` → `SettingsView` | OK |
| 8 | **Téléversement** | host configuré · ⚠ stub | — | `Détail…` → `UploadView` | Stub volontaire |

Topbar : titre app · chips santé live (`● Backend · ● ExifTool`) · bouton thème · bouton aide.

Auto-refresh des 5 panneaux droits toutes les 5 s (status IA, audit tail, settings chips).

### Vues détail (modales `App.open_in_modal(view_id)`)

| view_id | Fichier | Statut |
|---|---|---|
| `settings` | `app/views/settings.py` | OK |
| `audit` | `app/views/audit.py` | OK |
| `ai_control` | `app/views/ai_control.py` | OK |
| `validate` | `app/views/validate.py` | OK |
| `upload` | `app/views/upload.py` | Stub volontaire |

`App.show_details(title, builder)` ouvre une mini-modale à la demande
pour les détails d'une ligne d'audit ou d'une anomalie de validation
(remplace l'ancien ContextPanel).

## Dépendances externes

| Paquet | Version | Usage |
|---|---|---|
| `customtkinter` | ≥ 5.2 | Toute l'IHM |
| `Pillow` | ≥ 10 | Lecture dimensions image |
| `requests` + `urllib3` | ≥ 2 | HTTP vers Ollama |
| ExifTool | externe (PATH) | Lecture/écriture EXIF/IPTC/XMP |
| Ollama serveur | externe (HTTP) | Analyse vision |
| `pytest` (dev) | ≥ 7 | Tests |
| `ruff` (dev) | ≥ 0.4 | Lint + format |
| `pyinstaller` (dev) | ≥ 6 | Build EXE |

`pyproject.toml` ne déclare pas `pydantic`, `piexif`, `ollama`, `CTkToolTip` — ces
deps phantôme avaient été retirées en Phase 6 de l'audit précédent.
