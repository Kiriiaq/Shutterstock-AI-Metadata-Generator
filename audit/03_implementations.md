# Phase 3 — Combler les trous

Toutes les corrections appliquées dans le même commit. Aucune régression
détectée par la suite de tests.

## 1. Code mort archivé (4 fichiers)

`git mv` vers `_archive/legacy_ui_v3_predense/` (préserve l'historique git
sur chaque fichier) :

- `app/components/sidebar.py`
- `app/components/system_panel.py`
- `app/components/command_palette.py`
- `app/components/context_panel.py`

Ces 4 composants ont vécu pendant les itérations v3 et ont été remplacés
par les panneaux du Workspace dense + `App.open_in_modal()` +
`App.show_details()`. Aucun import résiduel.

`_archive/legacy_ui_v3_predense/README.md` créé pour documenter la
provenance et la procédure de réhabilitation.

## 2. Chaînage d'exception (9 sites B904)

Convention : `raise NewError(...) from <orig>` pour préserver le contexte
de l'exception originale dans les tracebacks.

| Fichier | Ligne | Avant | Après |
|---|---|---|---|
| `src/modules/ai/ollama_client.py` | 422 | `except requests.exceptions.Timeout:` | `... as exc` + `raise … from exc` |
| `src/modules/ai/ollama_client.py` | 425 | `raise OllamaError(...)` | `raise … from e` |
| `src/modules/engines/metadata_reader.py` | 230 | `except subprocess.TimeoutExpired:` | `... as exc` + `raise … from exc` |
| `src/modules/engines/metadata_reader.py` | 232 | `raise MetadataReadError(...)` | `raise … from e` |
| `src/modules/engines/metadata_reader.py` | 251 | `except subprocess.TimeoutExpired:` | `... as exc` + `raise … from exc` |
| `src/modules/engines/metadata_reader.py` | 253 | `raise MetadataReadError(...)` | `raise … from e` |
| `src/modules/engines/metadata_writer.py` | 442 | `except subprocess.TimeoutExpired:` | `... as exc` + `raise … from exc` |
| `src/modules/engines/metadata_writer.py` | 444 | `raise MetadataWriteError(...)` | `raise … from e` |
| `src/modules/engines/metadata_writer.py` | 486 | `raise MetadataWriteError(...)` | `raise … from e` |

## 3. Bugs B033 / B023 / B007

- **B033** `metadata_writer.py:537` : `RAW_EXTENSIONS` set contenait `.rw2`
  deux fois (Panasonic + Leica). Suppression de la seconde occurrence
  + commentaire explicatif. Pas d'effet runtime (un set dédoublonne) —
  était trompeur à la lecture.
- **B023** `worker_pool.py:480` : la closure `stage_progress` capturait
  la variable de boucle `stage_name`, ce qui renvoyait toujours le nom
  du **dernier** stage si l'appel était différé. Correction par
  default-arg : `def stage_progress(..., _stage_name=stage_name)`.
- **B007** `worker_pool.py:468` : la variable `handler` du `for stage_idx,
  (stage_name, handler) in enumerate(...)` n'était jamais utilisée dans
  le corps (le handler est récupéré via `self._pools[stage_name]`).
  Renommée `_handler`.

## 4. Comportement Esc — chaîne fermeture-modale → cancel-processing

Réécriture de `App._close_top_modal` :

1. Purge la liste `_open_modals` des Toplevels déjà détruits (X-clic ou
   Esc-dans-modale) — la liste pouvait grossir indéfiniment.
2. S'il reste un modal vivant, ferme-le.
3. Sinon, si le Workspace est en `_processing`, appelle son
   `_analyze_stop()` — Esc et le bouton « Arrêter » du panneau Analyse IA
   pointent maintenant vers la même logique d'annulation.

Match l'attente courante d'un utilisateur (Esc = annuler), tout en
conservant la priorité Esc-ferme-le-modal-d'abord.

## 5. Décisions documentées

- **Pas de `Ctrl+1..5`** : la sidebar a été supprimée en Phase 8 de l'audit
  précédent, à la demande explicite de l'utilisateur (« pas de duplication »).
  Ajouter Ctrl+1..5 maintenant rétablirait une couche de navigation
  duplique de ce qui est déjà visible. Hors-scope.
- **Pas de re-implémentation FTPS** : panneau Téléversement reste un stub
  conscient avec EmptyState explicite. Décision préservée de l'audit
  précédent (R7).
- **S110 résiduels conservés** : sur `widget.configure()` autour de
  changements de thème, `tk.TclError` est attendue lorsque le widget vient
  d'être détruit (theme switch déclenche un rebuild). Logger ces erreurs
  ferait du bruit pour aucun bénéfice opérationnel.
