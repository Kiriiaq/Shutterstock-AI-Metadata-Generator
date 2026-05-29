"""
ShutterstockAnalyzer — single source of truth for the package version.

Professional image metadata management with AI-powered analysis. Every
other version string (``build.py``, the UI title bars in
``app/i18n/fr.py``, the ``main.py`` docstring) derives from
``__version__`` below. ``tests/test_core/test_version.py`` is the
regression guard that fails the suite if any of them drift apart.
"""

__version__ = "2.1.0"
__author__ = "Emmanuel Grolleau"
