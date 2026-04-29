"""Smoke test for the new ``app/`` shell.

Headless instantiation: the App is built, the layout grid is checked,
the placeholder ``home`` view is loaded, theme toggle round-trip, then
the window is destroyed. No backend is wired (api=None branch).
"""

from __future__ import annotations

import pytest


def test_app_v3_shell_lifecycle(tmp_path, monkeypatch):
    pytest.importorskip("customtkinter")

    # Redirect prefs file to a tmpdir so the real user config isn't touched.
    from app.config import theme as theme_mod

    fake_prefs = tmp_path / "ui_prefs.json"
    monkeypatch.setattr(theme_mod, "get_prefs_path", lambda: fake_prefs)

    from app.app import App
    from app.components.sidebar import NAV_ENTRIES
    from app.config.shortcuts import GLOBAL_SHORTCUTS, display_label

    app = App(api=None)
    try:
        app.update_idletasks()

        # Title sanity
        assert "ShutterstockAnalyzer" in app.title()

        # Layout: sidebar + topbar + central must exist
        assert app.sidebar is not None
        assert app.topbar is not None
        assert app._center.winfo_exists()

        # Initial view = home
        assert app.router.current_id == "home"

        # Navigate through every registered entry without crashing
        for view_id, _icon, _label_key, _section in NAV_ENTRIES:
            app.router.navigate_to(view_id)
            app.update_idletasks()
            assert app.router.current_id == view_id

        # History: back / forward
        app.router.back()
        app.update_idletasks()
        app.router.forward()
        app.update_idletasks()

        # Sidebar collapse / expand
        original_width = app.sidebar.cget("width")
        app.sidebar.toggle_collapsed()
        app.update_idletasks()
        assert app.sidebar.cget("width") != original_width
        app.sidebar.toggle_collapsed()
        app.update_idletasks()

        # Theme toggle round-trip — must not raise and must be persisted to fake file
        before = theme_mod.load_theme_pref()
        app._toggle_theme()
        after = theme_mod.load_theme_pref()
        assert before != after
        assert fake_prefs.exists()

        # display_label: regression guard for "Alt+Left" not becoming "Alt+LefT"
        assert display_label("<Alt-Left>") == "Alt+Left"
        assert display_label("<Control-k>") == "Ctrl+K"
        assert display_label("<Control-Shift-T>") == "Ctrl+Shift+T"
        assert display_label("<F1>") == "F1"
        assert len(GLOBAL_SHORTCUTS) >= 12

    finally:
        try:
            app.destroy()
        except Exception:
            pass
