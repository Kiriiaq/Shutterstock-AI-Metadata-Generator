"""Regression guard: the version must have a single source of truth.

``src/__init__.py.__version__`` is canonical. Every other place that
needs the version (``pyproject.toml`` packaging metadata, ``build.py``
artifact naming, the two UI title bars) must agree with it. Before
v2.1.0 these had drifted (``src`` said ``2.0.0`` while everything else
said ``2.1.0``); this test exists so that never ships again.
"""

from __future__ import annotations

import re
from pathlib import Path

import src
from app.i18n.fr import t

ROOT = Path(__file__).resolve().parents[2]

# Canonical semantic-version shape, e.g. ``2.1.0`` / ``2.1.0rc1``.
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+")


def test_canonical_version_is_wellformed():
    assert _SEMVER_RE.match(src.__version__), src.__version__


def test_pyproject_matches_package_version():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "no version field in pyproject.toml"
    assert match.group(1) == src.__version__


def test_build_version_matches_package_version():
    # build.py imports the version from src, so this is structural — it
    # also fails loudly if someone re-hardcodes a literal there.
    import build

    assert build.VERSION == src.__version__


def test_ui_titles_embed_the_canonical_version():
    version = src.__version__
    assert version in t("app.title")
    assert version in t("app.topbar_title")
