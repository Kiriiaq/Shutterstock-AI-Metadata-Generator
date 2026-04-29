# Shutterstock AI Metadata Generator

> **Automate your stock photography workflow with AI-powered image analysis and metadata generation for Shutterstock**

[![Release](https://img.shields.io/badge/release-v1.0.1-blue.svg)](https://github.com/Kiriiaq/Shutterstock-AI-Metadata-Generator/releases)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/AI-Ollama%20Vision-orange.svg)](https://ollama.ai)
[![Ko-fi](https://img.shields.io/badge/Ko--fi-Support%20Me-ff5e5b?logo=ko-fi)](https://ko-fi.com/kiriiaq)

---

## The Problem

Stock photographers spend **hours manually writing descriptions, keywords, and categories** for each image before uploading to Shutterstock. This repetitive task is:
- Time-consuming (5-10 minutes per image)
- Error-prone (inconsistent keywords, missing categories)
- Frustrating (takes time away from actual photography)

## The Solution

**Shutterstock AI Metadata Generator** uses local AI vision models (via Ollama) to automatically:
- **Analyze your images** and understand their content
- **Generate SEO-optimized descriptions** (max 200 characters)
- **Create relevant keywords** (7-50 per image)
- **Assign Shutterstock categories** automatically
- **Detect editorial/illustration flags**
- **Export everything to CSV** ready for Shutterstock upload

**No cloud API costs** - Everything runs locally on your computer!

---

## Features

### AI-Powered Analysis
- Uses Ollama vision models (LLaMA 3.2 Vision, LLaVA, Moondream)
- Automatic GPU detection and optimization (NVIDIA CUDA)
- Generates Shutterstock-compliant metadata in seconds

### Smart Image Management
- **Pre-filtering**: Validates images meet Shutterstock requirements (4+ MP, correct format)
- **Batch processing**: Organizes images into folders of 50 (Shutterstock limit)
- **Resume capability**: Continue interrupted processing sessions
- **Duplicate detection**: Avoids reprocessing already analyzed images

### Built-in Ollama Management
- One-click Ollama server start/stop
- **Auto-repair**: Fixes common Ollama issues (zombie processes, port conflicts)
- Model download and loading from the GUI
- Real-time GPU/VRAM status display

### Validation & Upload
- Checklist validator (photos vs metadata matching)
- Metadata completeness verification
- FTPS upload to Shutterstock servers with progress tracking

---

## Quick Start

### Prerequisites
1. **Windows 10/11** (64-bit)
2. **[Ollama](https://ollama.ai/download)** installed
3. **GPU recommended** (NVIDIA with 4GB+ VRAM)

### Installation

#### Option 1: Download Executable (Recommended)
1. Download from [Releases](https://github.com/Kiriiaq/Shutterstock-AI-Metadata-Generator/releases):
   - `ShutterstockAI-MetadataGenerator-v1.0.1.exe` (Release)
   - `ShutterstockAI-MetadataGenerator-v1.0.1_debug.exe` (Debug with console)
2. Double-click to run - no installation needed!

#### Option 2: Run from Source
```bash
git clone https://github.com/Kiriiaq/Shutterstock-AI-Metadata-Generator.git
cd Shutterstock-AI-Metadata-Generator
pip install -r requirements.txt
python main.py            # launcher (root entry point)
# ou de manière équivalente :
python -m app.main        # entrée explicite dans la couche UI v3
```

> Le code est organisé en deux couches : le backend (`src/modules/`) reste
> stable, l'UI v3 vit dans `app/` avec CustomTkinter uniquement. Voir
> `docs/architecture.md` pour la cartographie complète.

### First Run
1. **Start Ollama** - Click "Start" or let auto-repair handle it
2. **Download a model** - Select `llama3.2-vision:11b` and click "Download"
3. **Select your photo folder** - Browse to your images
4. **Click "Start Analysis"** - Watch the AI work!
5. **Find your CSV** in the `Shutterstock/` folder

---

## Workflow

```
1. Photos Folder    →  2. Pre-filter     →  3. AI Analysis
   (your images)        (Valid/Invalid)       (metadata.csv)
                                                   ↓
4. Validation       ←  5. Shutterstock/  ←  Batch folders
   (Checklist tab)       (organized)         (max 50 images)
```

### Folder Structure After Processing
```
Your_Photos/
├── Valid/              # Pre-filtered valid images
├── Invalid/            # Rejected images (too small, wrong format)
├── Shutterstock/       # Batch 1 (up to 50 images + metadata.csv)
├── Shutterstock_2/     # Batch 2 (if needed)
└── Shutterstock_3/     # And so on...
```

---

## Supported Models

| Model | VRAM | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `moondream:1.8b` | 2GB | Fast | Basic | CPU or low VRAM |
| `llava:7b` | 4GB | Fast | Good | Budget GPUs |
| `llama3.2-vision:11b` | 7GB | Medium | Excellent | **Recommended** |
| `llava:34b` | 20GB | Slow | Best | High-end GPUs |

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 64-bit | Windows 11 |
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16 GB |
| GPU | None (CPU mode) | NVIDIA RTX 3060+ (8GB VRAM) |
| Storage | 500 MB | 20 GB (for AI models) |

---

## Troubleshooting

### Ollama Won't Start
Click the **"Repair"** button - it automatically:
- Kills zombie processes
- Frees port 11434
- Cleans temp files
- Restarts the server

### "No connection" Error
1. Check if Ollama is installed: `ollama --version`
2. Try manual start: `ollama serve`
3. Check Windows Firewall settings

### Slow Performance
- Use a lighter model (`moondream:1.8b`)
- Increase cooldown time between images
- Close other GPU-intensive applications

---

## Known Limitations

- Windows only (macOS/Linux support planned)
- Requires Ollama to be installed separately
- Large images (>100MP) are automatically rejected
- Processing speed depends on GPU/model choice

---

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Emmanuel Grolleau**

## Acknowledgments

- [Ollama](https://ollama.ai) - Local AI model server
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) - Modern GUI framework
- [Pillow](https://python-pillow.org/) - Image processing

---

## Keywords

`shutterstock metadata generator` `stock photography automation` `AI image analyzer` `photo keywording tool` `Ollama vision` `image description generator` `automatic photo tagging` `Shutterstock CSV generator` `stock photo workflow`

---

## Build Executables

```bash
pip install -e ".[dev]"

python build.py debug     # debug profile (console + import trace)
python build.py release   # release profile (windowed, no console)
python build.py all       # both
python build.py clean     # purge build/, dist/, *.spec
```

Output: `dist/ShutterstockAnalyzer.exe` (release) and
`dist/ShutterstockAnalyzer-debug.exe` (debug).

## Development

```bash
pip install -e ".[dev]"
ruff check app/ src/ main.py build.py tests/
ruff format app/ src/ main.py build.py tests/
pytest tests/ -q
```

---

## Project structure

```
ShutterstockAnalyzer/
├── main.py                 # 5-line wrapper -> app.main:main
├── build.py                # PyInstaller (debug | release | all | clean)
├── pyproject.toml
├── docs/
│   └── architecture.md     # full UI v3 cartography
├── app/                    # UI v3 — CustomTkinter + stdlib only
│   ├── main.py             # bootstrap (logging, theme, backend, mainloop)
│   ├── app.py              # App(CTk) shell + Router + shortcuts wiring
│   ├── config/             # theme.py + shortcuts.py
│   ├── core/               # events.py + state.py + navigation.py
│   ├── components/         # 10 reusable widgets (sidebar, topbar, palette,
│   │                       #   data_table, form_field, empty_state, toast,
│   │                       #   tooltip, confirm_dialog, context_panel)
│   ├── views/              # 9 business views (home, sources, analyze,
│   │                       #   editor, audit, ai_control, settings,
│   │                       #   validate, upload)
│   ├── i18n/fr.py          # every visible string keyed here
│   └── utils/formatters.py # FR number / date / size / duration
├── src/                    # Backend (untouched by UI work)
│   ├── core/               # ShutterstockParams, ConfigManager, logger
│   ├── modules/            # AI client, engines, models, storage, workers
│   │                       # All accessed via the ShutterstockAIv2 facade.
│   └── utils/              # validators, file helpers, splash
├── tests/
│   ├── test_core/          # backend unit tests
│   ├── test_utils/         # backend unit tests
│   ├── smoke/              # baseline: 13 backend smokes (audit safety net)
│   └── ui/                 # 1 consolidated end-to-end UI smoke
└── _archive/               # legacy code preserved (ALLOW_DELETE=false)
    ├── legacy_ui_v1/       # ShutterstockApp + 6 page_*.py from v1
    └── legacy_ui_v2/ui/    # 5 active pages from the audit campaign
```

The UI layer (`app/`) and the backend (`src/`) talk through one facade
(`src.modules.integration.ShutterstockAIv2`). Vues never import from
`src.modules.storage.database`, `src.modules.engines.*`, etc. directly —
this is what made the v2 → v3 swap surgical.

## Coding conventions

- Python 3.11+. Type hints everywhere on public APIs; docstrings on
  classes and public methods.
- Functions ≤ 50 lines, classes ≤ 300 lines (App shell justified).
- `logging` stdlib (`logger = logging.getLogger(__name__)`); never
  `print()` in production code.
- Every visible string passes through `app.i18n.fr.t(key)` — no
  literal user-facing strings inside views/components.
- French-locale formatting via `app.utils.formatters` (NBSP separator,
  decimal comma, JJ/MM/AAAA dates).
- All widgets accessible by keyboard (Tab/Enter/Esc); no info conveyed
  by colour alone (always paired with icon or text).
- Every operation > 300 ms runs in a `threading.Thread(daemon=True)`
  with results posted to the UI via `widget.after(0, callback)`.
- Use `grid()` everywhere for layout. Mixing `pack` and `grid` inside
  the same parent will raise — the App root is grid, so child
  containers in tests must also be grid'd.

## Adding a new view

1. **Pick a slug** (e.g. `reports`) and an icon glyph.
2. **Add an `i18n` entry** in `app/i18n/fr.py`:
   `"nav.reports": "Rapports"`.
3. **Append to** `app/components/sidebar.py::NAV_ENTRIES`:
   `("reports", "📊", "nav.reports", "system")`.
4. **Create the view** in `app/views/reports.py`, subclass `BaseView`:
   ```python
   class ReportsView(BaseView):
       view_id = "reports"
       def __init__(self, master, *, app):
           super().__init__(master)
           self.app = app
           self._build()
       def _build(self): ...
       def on_enter(self, **kwargs): ...   # optional, called by Router
       def on_leave(self): ...             # optional, called by Router
   ```
5. **Register the factory** in `app/app.py::App._register_views`:
   ```python
   factories = {
       ...,
       "reports": lambda parent: ReportsView(parent, app=self),
   }
   ```
6. **(Optional) Add a smoke check** in
   `tests/ui/test_app_v3_shell.py` — the existing navigation loop
   already exercises any newly-registered view automatically; only
   add explicit assertions if the view exposes new public API worth
   guarding against regression.

---

## Support

If this tool saves you time, consider supporting the project:

[![Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/kiriiaq)

**Star this repository if it helps your stock photography workflow!**
