"""UI smoke tests — headless instantiation of the main App window.

Exercises the full import chain (main → src.modules.* → src.ui.*) plus
every page constructor under a real Tk root, then closes cleanly via
on_closing(). Closest available proxy for "did the v2 refactor actually
wire up?" without a full mainloop.

Single test by design: multiple App() instances in one process share
Tk globals (default root, after() queues, ctk theme), and tearing them
down between tests is racy with AIControlPage's deferred connection
probe. One full lifecycle test is more reliable than two split ones.
"""

import os
import sys

import pytest


@pytest.mark.skipif(
    sys.platform != "win32" and not os.environ.get("DISPLAY"),
    reason="Tk requires a display server (set DISPLAY or run on Windows desktop)",
)
def test_app_full_lifecycle(monkeypatch):
    """App() builds, exposes the right identity + tabs, and on_closing
    tears down without raising.

    Mocks requests.get/post so the Ollama connection probe inside
    AIControlPage doesn't gate the test on a 5-second TCP timeout.
    """
    import requests

    def _instant_fail(*args, **kwargs):
        raise requests.exceptions.ConnectionError("test: no Ollama")

    monkeypatch.setattr(requests, "get", _instant_fail)
    monkeypatch.setattr(requests, "post", _instant_fail)

    import main

    app = main.App()
    try:
        app.update_idletasks()

        # Window identity — matches the audit acceptance check format
        # "{{PROJECT_NAME}} {{PROJECT_VERSION}} - {{PROJECT_TAGLINE}}".
        title = app.title()
        assert "ShutterstockAnalyzer v2.0.0" in title
        assert "AI Metadata Generator" in title

        # All tabs present (= every page constructor succeeded).
        for tab in ["AI Control", "Scan Images", "AI Process", "Metadata Editor", "Audit Log", "Settings"]:
            assert app.tabview.tab(tab) is not None

        # Facade attribute present (None only if ExifTool unavailable;
        # either way, the attribute must exist).
        assert hasattr(app, "api")
    finally:
        # on_closing handles api.close() + self.destroy(); calling
        # destroy() again would raise on a dead widget.
        try:
            app.on_closing()
        except Exception:
            # If on_closing fails for any reason, force-destroy so the
            # next test isn't poisoned by a dangling Tk root.
            try:
                app.destroy()
            except Exception:
                pass
            raise
