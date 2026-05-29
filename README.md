# ShutterstockAnalyzer

> **Generate Adobe Stock & Shutterstock metadata locally — AI optional, FTP push built-in.**

[![Version](https://img.shields.io/badge/version-2.1.0-blue.svg)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.11+-brightgreen.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-120%20passing-success.svg)](tests/)
[![Build](https://img.shields.io/badge/build-PyInstaller-orange.svg)](build.py)

> _Demo GIF placeholder — see [`docs/MEDIA.md`](docs/MEDIA.md) for the asset list to produce._

---

## What it does

Stock photographers spend **5–10 minutes per image** writing titles, descriptions,
keywords and categories before upload. ShutterstockAnalyzer collapses that
loop to **a few seconds per image**, fully offline:

- **Heuristic metadata builder** — generates an 8-section expert report
  (scores, keywords, risks, marketing uses) from the **existing IPTC + image
  properties only**. No AI required — runs instantly on any PC.
- **Optional AI enrichment** via local [Ollama](https://ollama.ai) vision
  models (LLaMA 3.2 Vision, LLaVA, Moondream). Off by default.
- **Dual-platform CSV export** — Adobe Stock (`Filename,Title,Keywords,
  Category,Releases`) and Shutterstock (`Filename,Description,Keywords,
  Categories,Editorial,Mature,Illustration`) with UTF-8 BOM, ready for
  the contributor portals.
- **Direct FTP / FTPS push** to the contributor portal after export.
- **IPTC write-back** (opt-in) — files unchanged if the checkbox stays off.

No cloud API costs · no telemetry · everything stays on your machine.

---

## Quick start

### Option 1 — Download the EXE (recommended)

1. Grab `ShutterstockAnalyzer.exe` from the [Releases](https://github.com/Kiriiaq/Shutterstock-AI-Metadata-Generator/releases) page (~25 MB).
2. Double-click. Done.

### Option 2 — Run from source

```bash
git clone https://github.com/Kiriiaq/Shutterstock-AI-Metadata-Generator.git
cd Shutterstock-AI-Metadata-Generator
pip install -r requirements.txt
python main.py
```

---

## Workflow

```
Photos folder  ──►  Scan + select  ──►  Expert report  ──►  Export CSV  ──►  FTP push
                       │                     │              (Adobe       (optional)
                       │                     │               +/or SH)
                       │                     │
                       ▼                     ▼
                   IPTC editor          AI enrichment
                   (manual)             (optional, opt-in)
```

1. **Sources & tri**: scan a folder, add files incrementally, multi-select.
2. **Rapport expert…**: 8-section report (4 scores, dual titles, keywords with
   top-10 highlighted, categories Adobe + Shutterstock, rejection risks,
   improvements, marketing uses, buyer profiles, trends).
3. **📤 Exporter…**:
   - Pick platform: Adobe / Shutterstock / both.
   - Toggle « Écrire IPTC dans le fichier » (off by default — files stay
     untouched, results are in the CSV).
   - Toggle « Enrichir avec IA » (off by default — heuristic only).
   - Pick output folder and basename.
   - Toggle « Pousser en FTP » with credentials + test button.
4. CSV files dropped in your output folder, optionally pushed to FTP.

---

## Optional AI setup (Ollama)

The default heuristic mode works **without Ollama**. To enable AI enrichment:

1. Install [Ollama](https://ollama.ai/download).
2. Pull a vision model:
   ```bash
   ollama pull llama3.2-vision:11b      # 7 GB VRAM, recommended
   ollama pull llava:7b                 # 4 GB VRAM, faster
   ollama pull moondream:1.8b           # 2 GB VRAM, CPU OK
   ```
3. In the **Export Batch** modal, check « Enrichir avec IA » → use the
   « 🔌 Tester » button to probe the server, pick a model in the dropdown,
   click « ⬇ Charger » to warm it up.
4. The topbar chip turns green and shows the loaded model name.

Without Ollama, the « Enrichir avec IA » checkbox is harmless: the pipeline
falls back to the heuristic builder transparently.

---

## Editions

The app ships as a freemium build unlocked by a key. **Community is free
forever** and includes the *entire* analysis workflow — scan, IPTC
editor, the multi-section expert report, AI enrichment, the
dual-platform CSV layout, FTP push. The only metered thing is the
**data export**: Community gets **3 free export runs**, then the **Pro
key (10 € — lifetime, one-shot)** unlocks unlimited exports.

| Feature | Community | Pro (10 € à vie) |
|---|:---:|:---:|
| Scan folder + multi-select | ✅ | ✅ |
| IPTC read / write (manual editor) | ✅ | ✅ |
| Expert report — 4 scores + risks + improvements + marketing uses | ✅ | ✅ |
| AI enrichment via local Ollama vision models | ✅ | ✅ |
| Dual-platform CSV layout (Adobe + Shutterstock) | ✅ | ✅ |
| Pre-upload validation, history, theme, FTP / FTPS push | ✅ | ✅ |
| Cross-platform compliance hints (Adobe 4–100 MP, Shutterstock 4 MP+ / 50 MB) | ✅ | ✅ |
| **Data export** (write the CSV your stock platforms ingest) | 🎁 3 free runs | ✅ unlimited |

Pro is activated by pasting a JSON licence key into **Settings → Licence**.
One-shot **10 €** payment, works offline forever, no subscription.
See [`docs/MONETIZATION.md`](docs/MONETIZATION.md) for the rationale.

---

## Features (technical detail)

| Feature | Detail |
|---|---|
| **Multi-platform export** | Adobe Stock + Shutterstock CSVs, UTF-8 BOM, comma-separated keywords (Shutterstock-compliant) |
| **Expert report** | 4 scores (commercial / technique / SEO / risque rejet), dual titles, top-10 keywords, rejection risks, marketing uses |
| **Heuristic-first** | All scores + reports work without AI. Runs in ~3 s on 15 images. |
| **AI optional (Pro)** | Ollama vision models for enrichment. Falls back gracefully when Ollama is absent. |
| **FTP / FTPS push** | Direct upload to contributor portal after export. Credentials never persisted by default. |
| **IPTC editor** | Read/write IPTC headline, caption, keywords, byline, copyright. |
| **Validation pre-upload** | Per-image checks (resolution, format, file size, keyword count). |
| **Theme** | Light / dark / system, persisted between sessions. |
| **History** | All operations logged to local SQLite, filterable + exportable. |
| **Cross-platform compliance** | Adobe (4–100 MP, 45 MB, sRGB) + Shutterstock (4 MP min, 50 MB) checks as warnings, not blockers. |

---

## Anti-stuffing built in

The keyword pipeline silently filters:

- **Brand names** (`apple`, `nike`, `coca-cola`, `bmw`…) — auto-stripped.
- **Stuffing keywords** (`stock`, `image`, `photo`, `wallpaper`…) — stripped
  unless they appear in the title (where they describe the image).
- **Duplicates** + lowercase normalization.
- Hard cap at **50 keywords** (Adobe + Shutterstock limit).
- **Top 10 are commercial priority** — the rest is searchable padding.

---

## Architecture

```
ShutterstockAnalyzer/
├── main.py              ← thin entry point
├── app/                 ← UI v3 (CustomTkinter)
│   ├── app.py           ← shell + router + modal manager
│   ├── components/      ← 8 reusable widgets
│   ├── config/          ← theme, shortcuts
│   ├── core/            ← events, state, navigation
│   ├── views/           ← workspace + 6 modal views
│   ├── i18n/fr.py       ← all UI strings (French)
│   └── utils/           ← formatters
└── src/                 ← Backend (UI-agnostic)
    ├── core/            ← params, config_manager, logger
    ├── modules/
    │   ├── ai/          ← Ollama client + vision analyzer
    │   ├── analysis/    ← expert_report + platform_compliance
    │   ├── engines/     ← metadata_reader + metadata_writer (ExifTool)
    │   ├── export/      ← csv_exporter + batch + ftp_uploader
    │   ├── models/      ← dataclasses (IPTC, Expert, Shutterstock)
    │   ├── storage/     ← SQLite database
    │   ├── workers/     ← worker_pool
    │   └── integration.py  ← ShutterstockAIv2 facade (UI entry point)
    └── utils/           ← validators, file helpers
```

Full cartography in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). UI and
backend talk through **one facade** (`src.modules.integration.ShutterstockAIv2`).

---

## System requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 64-bit | Windows 11 |
| Python (source) | 3.11 | 3.11 |
| RAM | 4 GB (heuristic mode) | 16 GB (AI mode) |
| GPU | None | NVIDIA RTX 3060+ (8 GB VRAM) for `llama3.2-vision:11b` |
| Storage | 100 MB (EXE only) | 20 GB (AI models) |
| ExifTool | optional (IPTC read/write) | recommended |
| Ollama | optional (AI enrichment) | optional |

macOS / Linux are not actively tested. The code itself is portable; only
the bundled EXE and the AppUserModelID are Windows-specific.

---

## Development

```bash
pip install -e ".[dev]"

# Run
python main.py

# Tests (120 tests, ~7 s)
pytest tests/ -q

# Lint
ruff check app/ src/ main.py build.py tests/
ruff format app/ src/ main.py build.py tests/

# Build EXEs (debug + release)
python build.py all
```

Builds drop to `dist/ShutterstockAnalyzer.exe` and `…-debug.exe`.

---

## Project structure

| Folder | Purpose |
|---|---|
| `app/` | UI v3 — CustomTkinter, French locale |
| `src/` | Backend — UI-agnostic, single facade entry point |
| `tests/` | Automated suite (pytest, 120 tests) |
| `test/` | Qualification dossier (Edvance methodology) — matrix XLSX, interactive HTML, Pillow inputs, run/compare scripts |
| `docs/` | ARCHITECTURE, MEDIA, MONETIZATION |
| `audit/` | Internal audit history + screenshots |
| `assets/icons/` | Windows ICO |
| `tools/` | Standalone scripts (WCAG colour checker) |
| `.github/workflows/` | CI (lint + test) + Release (tag-triggered) |

---

## Roadmap

- ✅ Adobe Stock CSV export (v2.0.0)
- ✅ FTP / FTPS push (v2.0.0)
- ✅ Ollama model selection + preload (v2.0.0)
- ✅ Heuristic-only expert report (no AI required) (v2.0.0)
- ✅ Pro tier — quality evaluation, dual CSV, AI enrichment, batch > 50 (v2.1.0)
- 🟡 FTP scheduling (Pro) — background recurring push
- 🟡 Multi-account FTP (Pro) — Adobe + Shutterstock simultaneous
- 🟡 Custom IPTC templates (Pro)
- 🟡 macOS / Linux EXE
- 🟡 Built-in demo GIF + screencast
- 🟡 Drag & drop into Sources panel
- ⚪ Custom prompt templates per category

---

## Security

Credentials (FTP password) are **never persisted** by default. See
[`SECURITY.md`](SECURITY.md) for the full handling policy.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). This is a personal project; PRs
are welcome but not the primary distribution channel — for major
changes, open an issue first.

---

## License

[MIT](LICENSE) © 2024-2026 Emmanuel Grolleau.

The Community edition (everything described above as ✅) is free for
personal and commercial use under MIT.

The **Pro edition** unlocks **unlimited data export** (Community runs the
export 3 times for free). Everything else — scan, IPTC, expert report,
AI enrichment, dual-platform CSV, FTP — is free. Sold via Gumroad as a
**10 € one-shot lifetime key**, activated locally with HMAC signature
verification.

See [`docs/MONETIZATION.md`](docs/MONETIZATION.md) for the full
breakdown.

*Not affiliated with Shutterstock, Inc. or Adobe Inc.*

---

## Acknowledgments

- [Ollama](https://ollama.ai) — local AI model server
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) — modern GUI framework
- [Pillow](https://python-pillow.org/) — image processing
- [ExifTool](https://exiftool.org/) — IPTC read/write engine

---

## Support

If this tool saves you time, consider supporting the project:

[![Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/kiriiaq)

**Star the repo** if it helps your stock photography workflow.
