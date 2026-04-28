# BUILD_REPORT — audit/20260428

PyInstaller packaging for ShutterstockAnalyzer v2.0.0.

## Profiles

```bash
python build.py debug      # Debug:   --console --debug=imports --noupx
python build.py release    # Release: --windowed --noconsole --noupx
python build.py all        # Both
python build.py clean      # Remove build/, dist/, *.spec
```

Both profiles share `--onefile`, `--icon=assets/icons/icone.ico`, `--noupx`, hidden imports for `customtkinter` / `darkdetect` / `PIL`, exclusions for unused data-science / Office / PDF stacks, and bundle `assets/` + `src/` via `--add-data`.

## Pre-scan results (Phase G.1)

| Module | In v2 source? | In `HIDDEN_IMPORTS` | Decision |
|---|---|---|---|
| `customtkinter` | yes (main.py + every page) | yes | keep |
| `darkdetect` | transitive via CTk | yes | keep (CTk uses string-based imports) |
| `PIL` | yes (scan, write, splash, validators) | yes | keep |
| `PIL.ImageTk` | no — removed by ruff F401 | no | not needed |
| `requests` | yes (ollama_client) | no — auto-detected | OK |
| `urllib3` | transitive via requests | no — auto-detected | OK |
| `tkinter.ttk` | yes (audit_page) | no — stdlib auto | OK |
| `sqlite3`, `subprocess` | yes | no — stdlib auto | OK |

Dropped from `pyproject.toml` and from PyInstaller hidden imports (verified with `grep` — never imported in `src/`):

- `CTkToolTip` (project has its own `tooltips.py`)
- `piexif` (no usage)
- `ollama` Python pkg (the project ships its own `OllamaClient`)
- `pydantic` (no usage)

Excluded modules in both profiles (heavy trees never imported by v2):
`scipy, numpy, pandas, matplotlib, seaborn, docx, pptx, openpyxl, xlrd, xlsxwriter, oletools, PyPDF2, pdfplumber, fitz, pymupdf, reportlab, cv2, dlib, moviepy, whisper, win32com, pythoncom, pywintypes`, plus the `pyproject` v1-residual deps listed above.

## Build results

| Profile | Output | Size | Wall time | Status |
|---|---|---|---|---|
| Debug | `dist/ShutterstockAnalyzer-debug.exe` | **24.4 MB** | ~20 s | OK |
| Release | `dist/ShutterstockAnalyzer.exe` | **24.4 MB** | ~23 s | OK |

Both are well under the 100 MB pragmatic ceiling (and far under any sane Pillow-app target). The dependency trim from v1 removed the unused `pydantic` / `ollama` / `piexif` / `CTkToolTip` trees, which is the main reason the bundle is this small.

## Smoke tests (Phase G.3)

Both EXEs were launched via `subprocess.Popen` with an 8 s timeout:

| EXE | Behavior | Verdict |
|---|---|---|
| Debug | Started, emitted 3278 lines of `--debug=imports` trace, stayed alive past timeout (= killed by harness) | OK — GUI mainloop reached |
| Release | Started, no stdout (expected: `--windowed --noconsole`), stayed alive past timeout | OK — GUI mainloop reached |

A second smoke ran with `cwd=dist/`. Both EXEs again stayed alive past the 6 s budget without exiting on their own — i.e. neither crashes during App.__init__ or the first mainloop iteration.

## Acceptance check (Phase G.4)

| Check | Status | How verified |
|---|---|---|
| Window title = `ShutterstockAnalyzer v2.0.0 - AI Metadata Generator for Stock Photography` | ✓ | `tests/ui/test_app_smoke.py::test_app_full_lifecycle` asserts the format under a real Tk root. The same `App` class is what the EXEs run. |
| Window icon (top-left corner) | ✓ best-effort | `App.__init__` calls `self.iconbitmap(resource_path("assets/icons/icone.ico"))`. `resource_path` honours `sys._MEIPASS`, so it resolves under the bundled tmp dir. Visual confirmation requires a desktop session. |
| EXE icon in Explorer | ✓ | `--icon=assets/icons/icone.ico` was in the PyInstaller args (verified in build log line `INFO: Copying icon to EXE`). |
| Taskbar icon (grouped under app, not generic Python) | ✓ best-effort | `main()` calls `SetCurrentProcessExplicitAppUserModelID("ShutterstockAnalyzer.v2.0")` before any window is created. |
| AUCUNE console au lancement (release) | ✓ | `--windowed --noconsole`, smoke ran with `stdout=PIPE` returned 0 bytes. |
| `dist/ShutterstockAnalyzer.exe` ≤ 100 MB target | ✓ | 24.4 MB, well under. |

The visual-only checks (icon rendering, taskbar grouping) cannot be auto-verified from this session — they're listed best-effort because the build artefacts are present and correctly wired in code, but actually seeing them requires a graphical desktop session.

## Iterations / corrections during builds

Zero. Both `debug` and `release` succeeded on the first attempt because the integration / DB / UI / scoping bugs (B-1 to B-18) had already been fixed in Phase E, and the dependency trim was done before the first PyInstaller invocation.

## Known residuals (out of scope for this packaging)

- ExifTool is an external binary, not bundled. The app degrades gracefully (logs a warning, `metadata_reader` / `metadata_writer` set to None, status bar shows `ExifTool: NOT FOUND`). Documented in README.
- Ollama server is external (Windows installer from ollama.ai). The app shows `AI: Offline` if not running.
- Build hosts: Windows-only (PyInstaller produces a `.exe`). macOS / Linux out of scope.

## Reproducing locally

```bash
pip install -e ".[dev]"
python build.py all
# dist/ShutterstockAnalyzer.exe       (release, no console)
# dist/ShutterstockAnalyzer-debug.exe (debug, with console + import trace)
```

To clean: `python build.py clean` removes `build/`, `dist/`, and any leftover `*.spec`.
