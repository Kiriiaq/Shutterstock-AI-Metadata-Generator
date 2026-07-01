# Changelog

All notable changes to this project are documented here.

Format inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.2.1] — 2026-07-01

> **Audit complet 2026-06-12** (branche `audit/2026-06-12`) — scan
> intégral du code, 2 bugs majeurs + 7 mineurs corrigés, ~2 800 lignes
> de code mort v1 retirées. Aucun changement de comportement voulu en
> dehors des correctifs listés.

### Fixed
- **« Ignorer si méta » (Analyse IA batch) fonctionne enfin** : la
  facade pré-filtre les fichiers qui portent déjà un bloc IPTC
  exploitable (headline + keywords) avant d'appeler le modèle. Avant,
  l'option était un no-op et toutes les images repassaient à l'IA
  (B-01). 3 nouveaux tests facade.
- **Colonne « Méta » des Sources honnête** : sonde le vrai bloc IPTC
  (`has_metadata`) au lieu d'afficher « Oui » pour tout fichier lisible
  (B-02).
- Bouton « Aller à Sources » de la modale Validation (route inexistante
  → ferme désormais la modale) (B-03).
- La chip « Édition » de la topbar se rafraîchit immédiatement après
  activation/retrait d'une licence (B-04).
- L'export CSV multi-fichiers du Rapport expert ne gèle plus l'UI
  (construction des rapports déportée en thread) (B-06).
- Dédoublonnage des ajouts Sources concurrents (B-07) ; textbox des
  détails d'audit étirable (B-08).
- Modale « Modèle IA » : passe par la facade, persiste l'URL testée et
  le modèle choisi, bouton « ⬇ Charger » (le combo n'avait aucun effet)
  (B-09).

### Removed
- Code mort v1 (~2 800 lignes, grep-vérifié sans appelant, récupérable
  via git) : surface inutilisée de la facade (process_folder, diff/
  comparaison IA, templates wrappers, get_statistics…), handler
  `ai_analyze` cassé (B-05), `ProcessingPipeline`, suite sidecar XMP du
  writer, méthodes orphelines IPTCEngine/Database/OllamaClient/
  VisionAnalyzer, historique back/forward du Router, table SQLite
  `processing_queue` jamais écrite, et les modules v1 `src/core/` +
  `src/utils/file_utils|validators` (testés mais jamais utilisés par
  l'app — tests associés retirés avec eux).

### Docs
- Docstrings mises à jour (licence 10 € à vie, dataclasses stdlib),
  commentaire `urllib3` (dépendance transitive épinglée).

## [2.2.0] — 2026-05-29

> **Monetization pivot → single 10 € lifetime, export-only paywall.**
> The previous Pro tier (29/89 €/an, 79 € lifetime) gated quality
> evaluation. This release drops all of that: **everything is free** —
> scan, IPTC, the full expert report, AI enrichment, dual CSV, FTP —
> **except the data export**, which Community runs 3 times for free,
> then unlocks forever with a **10 € one-shot lifetime key**.

### Changed
- **Single paid tier.** `Tier` collapses to `COMMUNITY` + `LIFETIME`
  (dropped `PRO_SOLO` / `PRO_STUDIO`). `PRO_FEATURES = {"data_export"}`.
- **The paywall is the export run, not the analysis.** The expert
  report, AI enrichment and dual-platform CSV are now free and unlimited.
- **Quota moved from the expert report to the export.**
  `COMMUNITY_EXPORT_QUOTA = 3`; facade exposes `export_quota_remaining()`
  / `consume_export_quota()` / `reset_export_quota()` (settings key
  `community_exports_used`). Both export entry points (Export Batch and
  the expert report's CSV button) share the counter.
- **UI.** Export Batch shows a live export-quota banner + upsell at
  Start; all "🔒 Pro" labels on dual CSV / AI / batch are gone.
- **Pricing → 10 € lifetime** across README, settings, keygen and docs.

### Removed
- `Tier.PRO_SOLO`, `Tier.PRO_STUDIO`, the multi-feature `PRO_FEATURES`
  set (expert_report / dual_csv_export / ai_enrichment / batch_unlimited
  / roadmap flags) and the Community batch cap (50).

### Tests
- `test_licensing.py` + `test_freemium_journey.py` rewritten around the
  export quota and the single lifetime tier. Suite green, ruff clean.

---

## [2.1.0] — 2026-05-27

> **Pivot Pro = évaluation qualité.** The previous Pro tier targeted
> batch/scheduling add-ons that hadn't been built yet ; this release
> repositions Pro around **what the app actually delivers today** —
> the multi-section expert report, the dual CSV export, and the AI
> enrichment. Community keeps a fully working metadata workflow
> (scan, IPTC edit, single-platform CSV, FTP push) plus 2 teaser
> rapport-expert slots.

### Changed
- **Pro/Community frontier reframed**. The headline value the app
  delivers (quality scoring + multi-platform export + AI overlay)
  is now Pro ; the basic metadata pipeline stays free.

| Feature | Community | Pro |
|---|---|---|
| Scan folder, multi-select | ✅ | ✅ |
| IPTC read/write (manual editor) | ✅ | ✅ |
| Single-platform CSV export (Adobe **or** Shutterstock) | ✅ | ✅ |
| Basic validation, history, FTP push | ✅ | ✅ |
| **Expert report** (4 scores, risks, improvements, marketing uses) | 🎁 2 teasers | ✅ unlimited |
| **Dual CSV export** (Adobe + Shutterstock side-by-side) | 🔒 | ✅ |
| **AI enrichment** (Ollama vision overlay) | 🔒 | ✅ |
| **Anti-stuffing** (brand/keyword filters via expert report) | 🔒 | ✅ |
| **Batch > 50 images** | 🔒 | ✅ |

### Added
- **Three new Pro features** registered in
  `src/modules/licensing/license.py` :
  `expert_report`, `dual_csv_export`, `ai_enrichment` — alongside
  the six features carried over from the previous groundwork.
- **Community quota**: `COMMUNITY_EXPERT_REPORT_QUOTA = 2`. The
  expert report modal renders normally for the first 2 images, then
  swaps to a Pro pitch screen on the 3rd. Counter persisted to the
  SQLite `settings` table (`community_expert_reports_used`).
- **Facade API** (`ShutterstockAIv2`) gained:
  `.expert_report_quota_remaining()`,
  `.consume_expert_report_quota()`,
  `.reset_expert_report_quota()`.
- **UI — Expert Report modal**: yellow banner showing remaining
  teaser slots, AI checkbox locked with "🔒 Pro" label in Community,
  dedicated **upsell screen** (benefits list + Acheter Pro + J'ai
  déjà une clé) when the quota hits zero.
- **UI — Export Batch modal**: "Les deux 🔒 Pro" indicator on the
  platform radio (default forced to Adobe in Community), "🔒 IA Pro"
  label on the enrichment checkbox, Pro gates at Start with
  explicit toast + status badge.

### Tests
- **9 new tests** in `tests/test_core/test_licensing.py`:
  `TestPivotFeatures` (4) pins the new Pro features in the registry,
  `TestCommunityExpertReportQuota` (5) covers the persisted counter
  (initial value, consume → zero clamp, persistence across facade
  restart, Pro=infinite, reset). **Total suite: 120 passing.**
- Ruff: still **0 errors** across `app/`, `src/`, `tests/`.

### Documentation
- `docs/MONETIZATION.md` — section 2.1 (frontier) and 2.2 (pricing)
  rewritten around the pivot.
- `README.md` — features table now flags Community/Pro per row.
- `LAUNCH_PROCEDURE.html` — section C.2 (Gumroad description) and
  C.4 (email template) reflect the new pitch.
- `LINKEDIN_DRAFTS.md` — three pitch formats updated to lead with
  "quality evaluation" rather than "batch automation".
- `CLAUDE.md` — sections « Fini », « État actuel », « Décisions
  techniques » updated.

### Security notes
- HMAC stays honour-system (documented in
  `src/modules/licensing/license.py`). The ed25519 hardening path
  remains on the v2.2.0 roadmap.

### Fixed
- **Single source of truth for the version.** `src/__init__.py` still
  declared `__version__ = "2.0.0"` while `pyproject.toml`, `build.py`
  and both UI title bars read `2.1.0`. The package `__version__` is now
  canonical — `build.py` and `app/i18n/fr.py` derive from it, and
  `tests/test_core/test_version.py` fails the suite if they ever drift.
- **Licence test isolation.** An autouse fixture redirects
  `DEFAULT_LICENSE_PATH` to a tmp file, so running the suite can no
  longer overwrite or delete a real `~/.shutterstock_ai/license.json`.

### Removed
- **Dead `src/utils/splash_screen.py`** — never wired into the app
  (`main.py` opens the window directly). Dropped with its smoke-test
  import line.

### Tests (release finalization)
- `test_freemium_journey.py` — end-to-end Community → quota exhausted →
  activate Pro → unlock → deactivate, plus a tampered-key rejection
  path. `test_version.py` — version-consistency guard. **Suite: 126.**

---

## [2.0.0] — 2026-05-19

Major release — pipeline becomes **multi-platform** (Adobe Stock + Shutterstock)
and the **AI step is now optional**.

### Added
- **Adobe Stock CSV export** (5 columns: `Filename, Title, Keywords, Category, Releases`).
- **Double CSV export** (Adobe + Shutterstock side by side in one click).
- **Expert microstock report** (`src.modules.analysis.expert_report`) — 8-section dashboard:
  scores (commercial / technical / SEO / rejection risk), dual titles, top-10 keywords,
  categories, rejection risks, improvements, marketing uses, buyer profiles, trends.
- **Heuristic-first mode** — full report builder runs **without Ollama** on any PC.
- **Platform compliance helper** (`src.modules.analysis.platform_compliance`) —
  Adobe (4–100 MP, 45 MB, sRGB) + Shutterstock (4 MP min, 50 MB) checks as
  non-blocking warnings.
- **FTP / FTPS push** (`src.modules.export.ftp_uploader`) — direct upload to
  contributor portal after CSV export. Stdlib `ftplib`, no extra dependency.
- **Batch export orchestrator** (`src.modules.export.batch.run_export_batch`) —
  end-to-end pipeline: reports → CSV → optional IPTC write-back → optional FTP push.
- **UI: `ExportBatchView` modal** — compact dashboard with platform radio,
  IPTC + AI checkboxes, file table with live status badges (⏸ → ⏳ → ✎ → ✅/❌),
  FTP credentials reveal, progress bar, log.
- **UI: `ExpertReportView` modal** — 8-section report, exportable.
- **Ollama model management** — `list_vision_models()`, `preload_model()`,
  `get_current_model()` on the facade, surfaced as a dropdown + test + load
  button + status chip inside the Export Batch modal.
- **Topbar Ollama chip enriched** — shows the loaded model name (e.g.
  `llama3.2-vision`) when warm, `En ligne (vide)` when server is up without a
  model, `Hors ligne` otherwise.
- **Keyword anti-stuffing** — silent filters for brand names (Apple, Nike,
  Coca-Cola, BMW…) and stuffing terms (`stock`, `image`, `photo`, `wallpaper`…
  preserved only when present in the title).
- **Qualification dossier** (`test/`) following the Edvance methodology:
  `matrice_tests.xlsx` (49 tests, 8 categories, with formulas), interactive
  `validation_ihm.html` (sections, 1-click OK/NOK/NA, micro-description per
  test, « Tout OK » per section, localStorage persistence, JSON + Markdown
  export), `inputs/` with 15 realistic Pillow images, `outputs_reference/`
  + `run_tests.py` + `compare_outputs.py` for cell-for-cell regression.
- **`AUDIT.md`** at the root — Phase 1 inventory (stack, code map, functional
  inventory, gaps, code mort).
- **`CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`, `CLAUDE.md`,
  `PROJECT_OVERVIEW.html`** — standard repo files (Phase 3).

### Changed
- **Workspace UI compacted** — analyse panel collapsed from 3 rows to 2
  (checkboxes + buttons + status on a single row, progress bar + summary
  on the second).
- **Sources panel** — new `📤 Exporter…` button (accent style), enabled
  when selection ≥ 1.
- **Editor IPTC** — new `Rapport expert…` button next to Lire/Écrire/Effacer.
- **Facade `ShutterstockAIv2`** — gained `build_expert_report`,
  `build_expert_reports_batch`, `export_double_csv`, `export_batch`,
  `test_ftp_connection`, `list_vision_models`, `preload_model`,
  `get_current_model`.
- **Author harmonised** to `Emmanuel Grolleau` in `pyproject.toml`
  (was inconsistently `Kiriiaq` vs the LICENSE).
- **README.md** completely rewritten for v2 (multi-platform, AI optional,
  FTP, expert report).

### Fixed
- **P0 — CSV Shutterstock keywords separator** : `ShutterstockMetadata.to_csv_row()`
  was joining keywords with spaces (one giant keyword on import). Now uses
  comma separator, matching the contributor portal template.
  (`src/modules/models/metadata_models.py:349`)
- **`ftplib.all_errors` nested tuple** — `except (ftplib.all_errors, OSError)`
  raised `TypeError: catching classes that do not inherit from BaseException`.
  Flattened to a module-level tuple `_FTP_ERRORS` at load time.

### Removed
- **`_archive/`** (legacy UIs v1, v2, v3-predense, v3-views, 315 KB total).
  Recoverable via `git show HEAD~:_archive/...` if ever needed.

### Tests
- Suite grew **24 → 90 tests** (~5 s wall time).
- New modules: `test_expert_report.py` (16), `test_csv_exporter.py` (4),
  `test_platform_compliance.py` (10), `test_export_batch.py` (10),
  `test_ftp_uploader.py` (8), `test_ollama_facade.py` (9).
- UI smoke covers the new `expert_report` and `export_batch` modals.
- Ruff: **0 errors** across `app/`, `src/`, `tests/`.

---

## [1.0.1] — 2026-04-29

### Added
- Initial PyInstaller build pipeline (`build.py debug | release | all | clean`).
- Audit campaign 20260428 → baseline test net (24 tests).
- Active runtime requirements trimmed to 4 dependencies (`customtkinter`,
  `Pillow`, `requests`, `urllib3`); v1 inheritance (`CTkToolTip`, `piexif`,
  `ollama`, `pydantic`) removed.

### Changed
- Architecture split — `app/` (UI v3) + `src/` (backend) with single facade.
- IPTC reader/writer refactored around ExifTool subprocess calls.

### Fixed
- `IPTCFields.from_dict` losing list fields (`keywords`,
  `supplemental_categories`) because `hasattr(cls, key)` returns False for
  `field(default_factory=list)`. Switched to
  `{f.name for f in dataclass_fields(cls)}`.
- Database API mismatches (`add_audit_log` vs `log_action`,
  `update_file_status` vs `set_file_flags`, batch lifecycle calls).
- `WritePage` wiring received the facade as `database` param — now passes
  the real database, reader, writer.

---

## [1.0.0] — 2025-02-06

### Added
- First public release.
- Ollama vision pipeline (LLaMA 3.2 Vision, LLaVA, Moondream).
- Shutterstock CSV export.
- IPTC metadata read + write via ExifTool.
- Batch processing (organised into 50-image batches matching the Shutterstock
  upload limit).
- Built-in Ollama server start/stop + auto-repair (zombie process killer,
  port 11434 freed).
- Pre-filtering (validates Shutterstock requirements: ≥ 4 MP, correct format).
- Checklist validator + FTPS upload to Shutterstock servers.

[2.1.0]: https://github.com/Kiriiaq/Shutterstock-AI-Metadata-Generator/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/Kiriiaq/Shutterstock-AI-Metadata-Generator/compare/v1.0.1...v2.0.0
[1.0.1]: https://github.com/Kiriiaq/Shutterstock-AI-Metadata-Generator/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/Kiriiaq/Shutterstock-AI-Metadata-Generator/releases/tag/v1.0.0
