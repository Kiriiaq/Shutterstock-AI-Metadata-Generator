# Legacy UI v2 — archived 2026-04-29

These five page modules and their `tooltips.py` helper were the active UI
surface during the v2 audit campaign (`audit/20260428`). They lived
under `src/ui/` and were instantiated by an inline `App` class inside
the original root-level `main.py`.

The v3 rewrite (`app/`) replaces them entirely:

| Legacy (`src/ui/pages/`) | Replacement (`app/views/`) |
|---|---|
| `ai_control_page.py` | `ai_control.py` |
| `audit_page.py` | `audit.py` |
| `scan_page.py` | `sources.py` |
| `settings_page.py` | `settings.py` |
| `write_page.py` | `editor.py` |

`tooltips.py` is replaced by `app/components/tooltip.py`.

The legacy code is kept under `_archive/` rather than deleted because:

- `ALLOW_DELETE=false` is the campaign policy.
- These files contain UX patterns and IPTC-mapping tables that may
  inform future v3 polish.

To revive: restore `src/ui/` and adjust the import statements in
`main.py` (the v3 wrapper would need to be rolled back to the v2
inline-`App` style).
