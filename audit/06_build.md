# Phase 6 — Build des exécutables

## Commandes

```bash
python build.py debug      # console + import trace
python build.py release    # windowed, no console
python build.py all        # both
python build.py clean      # purge build/, dist/, *.spec
```

`build.py` est un wrapper Python autour de PyInstaller, pas une ligne
shell brute — les flags `--exclude-module`, `--add-data`, hidden imports
sont centralisés.

## Profil DEBUG (`dist/ShutterstockAnalyzer-debug.exe`)

| Réglage | Valeur |
|---|---|
| Mode | `--onefile --console --debug=imports --noupx` |
| Hidden imports | `customtkinter`, `darkdetect`, `PIL` |
| Data dirs bundlés | `assets/`, `src/`, `app/` |
| Logging | DEBUG sur stdout (visible dans la console) |
| Taille (ce build) | 24.5 MB |

Cas d'usage : reproduire un crash, observer les imports manqués, lire les
tracebacks complets sans les copier-coller depuis la fenêtre Tk.

## Profil RELEASE (`dist/ShutterstockAnalyzer.exe`)

| Réglage | Valeur |
|---|---|
| Mode | `--onefile --windowed --noconsole --noupx` |
| Hidden imports | `customtkinter`, `darkdetect`, `PIL` |
| Data dirs bundlés | `assets/`, `src/`, `app/` |
| Excludes étendus | `pytest`, `IPython`, `notebook`, `matplotlib.tests`, etc. |
| Logging | INFO vers `logging` (pas de console) |
| Taille (ce build) | 24.5 MB |

Cas d'usage : utilisateur final.

## Smoke test post-build

Le wrapper `build.py` lance l'EXE en `subprocess.Popen` après le build,
attend 8 s, kill propre. Critère : exit 0 et pas de traceback.

Run de cette campagne (release) :

```
09:07:??  Database initialized: %APPDATA%/.shutterstock_ai/shutterstock_ai.db
09:07:??  WorkerPool initialized with 4 workers (processes=False)
09:07:??  UI proportional font resolved to: Segoe UI
09:07:??  UI monospace font resolved to: Cascadia Mono
09:07:??  UI mainloop starting     ← Atelier dense affiché en mode clair
…         (smoke kill timeout 8 s)
09:07:??  WorkerPool stopped
09:07:??  UI mainloop ended
exit code 0
```

Pas de DLL manquante, pas d'asset introuvable (vérifié à chaque build de
l'audit précédent et reconfirmé).

## Acceptance check Phase 6

- [x] `build.py release` produit un EXE windowed.
- [x] `build.py debug` produit un EXE console+imports.
- [x] EXE release démarre dans un env propre (vérifié multiples fois).
- [x] Pas de console au lancement release.
- [x] Icône custom (`assets/icons/icone.ico`) embarquée.
- [x] Title fenêtre = `ShutterstockAnalyzer v2.0.0 — Atelier`.
- [x] Taille ≤ 100 MB cible.
