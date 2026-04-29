"""App bootstrap — logging, theme, backend facade, mainloop."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make project root importable when the script is run directly
# (so ``python app/main.py`` and ``python -m app.main`` both work).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _set_appusermodel_id() -> None:
    """Windows: anchor the taskbar icon to ShutterstockAnalyzer.v2.0."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ShutterstockAnalyzer.v2.0")
    except Exception:
        logging.getLogger(__name__).warning("Could not set AppUserModelID", exc_info=True)


def _instantiate_backend():
    """Build the ShutterstockAIv2 facade. Returns ``None`` on failure so the
    UI degrades gracefully (the shell still loads and shows the user a toast)."""
    try:
        from src.modules.integration import ShutterstockAIv2

        return ShutterstockAIv2()
    except Exception:
        logging.getLogger(__name__).exception("Backend init failed; running UI-only")
        return None


def main() -> int:
    _configure_logging()
    _set_appusermodel_id()
    logger = logging.getLogger("ShutterstockAnalyzer")

    # Import after sys.path setup
    from app.app import App
    from app.config.theme import apply_theme, load_theme_pref

    apply_theme(load_theme_pref())

    api = _instantiate_backend()
    app = App(api=api)
    if api is None:
        app.toasts.show(
            "Backend indisponible — mode interface seule.",
            kind="warning",
            timeout_ms=6000,
        )

    logger.info("UI mainloop starting")
    app.mainloop()
    logger.info("UI mainloop ended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
