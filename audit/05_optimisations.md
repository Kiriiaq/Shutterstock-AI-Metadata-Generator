# Phase 5 — Optimisations

## Code mort retiré

4 fichiers archivés en Phase 3 (sidebar.py, system_panel.py,
command_palette.py, context_panel.py) — **~700 LOC** supprimés du
graphe d'imports actif. PyInstaller ne les bundle plus dans l'EXE car
ils ne sont plus dans le module path après le `git mv`.

## Imports différés déjà en place

Auditées dans l'audit précédent :

- `app/views/__init__.py` ne ré-exporte rien — chaque vue est importée à
  la demande dans `App._register_views()`.
- `src/modules/integration.py:init_ai()` charge `OllamaClient` et
  `VisionAnalyzer` au premier appel, pas à l'import du module.
- `app/views/workspace.py` charge PIL et `collect_image_files` dans le
  worker thread, pas au chargement de la vue.

## Pas d'optimisation runtime applicable

Le profil temps réel est dominé par :
- HTTP Ollama (5-30 s par image selon le modèle)
- subprocess ExifTool (200-800 ms par fichier)
- I/O disque pour le scan de dossier (variable)

Aucune de ces 3 sources n'est dans du code Python que nous contrôlons —
le code Python est toujours en attente. Optimiser le glue Python autour
ne donnerait pas de gain perceptible à l'utilisateur.

## Taille EXE

PyInstaller release reste à **24.5 MB** — seul le graphe d'imports
compte (les 4 fichiers archivés étaient déjà petits). Pas de gain
mesurable côté binaire.

## Métriques avant / après

| Mesure | Avant | Après | Δ |
|---|---|---|---|
| Fichiers `.py` actifs | 76 | 72 | -4 |
| Anomalies ruff B-rules | 12 | 0 | -100 % |
| Tests passants | 27 | 27 | = |
| Taille `dist/ShutterstockAnalyzer.exe` | 24.5 MB | 24.5 MB | = |
| Temps de démarrage EXE (smoke EXE médiane sur 3 runs précédents) | ~ 1.0 s avant mainloop | ~ 1.0 s | = |

L'optimisation utile portait sur la dette de qualité (chaînage
d'exception, code mort, B-rules) plutôt que sur la performance.
