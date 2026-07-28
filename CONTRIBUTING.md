# Contributing to StockMeta Pro

This is a personal project. Contributions are welcome but not the primary
distribution channel. For bigger ideas, open an issue **before** sending a PR
so we can sync on direction.

---

## Reporting bugs / requesting features

Open a GitHub issue with:

- **Bug report** — Windows version, Python version (if run from source),
  reproduction steps, expected vs actual behaviour, screenshot if UI is
  involved. If the issue is silent (no exception), attach the relevant
  excerpt from `~/.shutterstock_ai/logs/` or the debug EXE console output.
- **Feature request** — what you want, why, and a sketch of how you'd use it
  in your workflow. Microstock-specific context helps.

---

## Local development setup

```bash
git clone https://github.com/Kiriiaq/StockMeta.git
cd StockMeta
python -m venv .venv
.venv\Scripts\activate           # PowerShell / cmd
pip install -e ".[dev]"
```

Run the app:
```bash
python main.py
```

---

## Quality bar before opening a PR

| Check | Command | Required |
|---|---|---|
| Tests | `pytest tests/ -q` | All 120+ tests must pass |
| Lint | `ruff check app/ src/ main.py build.py tests/` | 0 errors |
| Format | `ruff format app/ src/ main.py build.py tests/` | Apply before commit |
| E2E pipeline | `python test/scripts/run_tests.py && python test/scripts/compare_outputs.py` | Cell-for-cell match |
| Build smoke | `python build.py release` | EXE produced, exit code 0 |

The CI runs the lint + test steps automatically on Windows for every PR.

---

## Coding conventions

- **Python 3.11+**. Type hints on public APIs. Docstrings on classes and
  public methods.
- **Functions ≤ 50 lines**, **classes ≤ 300 lines** (the `App` shell is
  the only justified exception).
- `logging` stdlib only — `logger = logging.getLogger(__name__)`. **Never**
  `print()` in production code.
- Every visible UI string passes through `app.i18n.fr.t(key)` — no literal
  user-facing strings inside views/components.
- French-locale formatting via `app.utils.formatters` (NBSP separator,
  decimal comma, JJ/MM/AAAA dates).
- Every operation > 300 ms runs in a
  `threading.Thread(daemon=True)` with results posted to the UI via
  `widget.after(0, callback)`.
- Use `grid()` everywhere for layout. The App root is grid; mixing `pack`
  inside the same parent will raise.

---

## Where to add what

| You want to add… | Put it in… |
|---|---|
| New UI view | `app/views/<slug>.py`, subclass `BaseView`, register in `app.py::_register_views` |
| New reusable widget | `app/components/<name>.py` |
| New backend feature | `src/modules/<area>/<name>.py`, expose via `src/modules/integration.py` facade |
| New export format | `src/modules/export/<name>_exporter.py` |
| New analysis heuristic | `src/modules/analysis/` |
| New test | `tests/test_core/` (backend) or `tests/ui/` (UI) |
| New asset / icon | `assets/icons/` (Windows ICO) |
| Architecture decision | Add a line in `docs/ARCHITECTURE.md` and update `CLAUDE.md` |

---

## Architectural rules of thumb

- **UI doesn't import backend modules directly** — only via
  `src.modules.integration.ShutterstockAIv2`. Keep the facade as the single
  contract.
- **The facade never imports UI**.
- **Heuristic-first** — every new feature must work without Ollama / ExifTool.
  These are *enrichers*, not requirements.
- **Lax posture on validation** — emit warnings, don't gate uploads. Adobe
  and Shutterstock reviewers are the final QA stage.
- **No new runtime dependency** without discussion. The current 4-dep list
  (`customtkinter`, `Pillow`, `requests`, `urllib3`) is intentional — keeps
  the PyInstaller bundle under 25 MB.

---

## Commit message style

Short, imperative, conventional-commits-ish:

```
feat(export): Adobe CSV writer + double export helper
fix(csv): Shutterstock keywords separator was space, must be comma
refactor(workspace): collapse analyse band 3 rows → 2
test(ftp): cover storbinary partial failure path
docs(readme): document AI-optional default mode
chore(deps): drop unused piexif import
```

---

## License

By contributing, you agree your code is released under the
[MIT License](LICENSE) of the project.
