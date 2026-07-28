"""StockMeta Pro — entry point.

Thin wrapper that delegates to ``app.main:main``. Kept at the repo root
so ``build.py`` (PyInstaller) and existing scripts that invoke
``python main.py`` continue to work unchanged.
"""

from __future__ import annotations

from app.main import main

if __name__ == "__main__":
    raise SystemExit(main())
