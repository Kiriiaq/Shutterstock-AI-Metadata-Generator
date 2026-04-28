# Legacy UI v1 — archived 2026-04-28

These files were the original UI shell for `ShutterstockApp`
(`main_window.py`) plus the six page modules it composed
(`page_source.py`, `page_model.py`, `page_analyze.py`,
`page_validation.py`, `page_upload.py`, `page_journal.py`) and two of
its components (`advanced_window.py`, `sidebar.py`).

They were superseded during the v2.0 refactor: `main.py` now defines an
inline `App(ctk.CTk)` that wires the new active pages
(`ai_control_page.py`, `audit_page.py`, `scan_page.py`,
`settings_page.py`, `write_page.py`) directly. The files here were no
longer instantiated by any active code path — they were imported only
by `src/ui/__init__.py` (now cleaned) and transitively by each other.

Kept under `_archive/` rather than deleted because:
- The audit policy is `ALLOW_DELETE=false` for this campaign.
- The page modules contain UX patterns (sidebar layout, source-folder
  workflow) that may be useful when the v2 UI grows.

To rehabilitate, restore the original tree layout (or rewrite imports
to be relative to `_archive`), then re-add `from .main_window import
ShutterstockApp` to `src/ui/__init__.py`.

Imports inside these files use the original `src.ui...` paths and
will not work in-place without restoration.
