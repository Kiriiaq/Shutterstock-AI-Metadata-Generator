"""
Pytest configuration and fixtures.
"""

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_license_file(tmp_path, monkeypatch):
    """Never let a test touch the user's real ~/.shutterstock_ai/license.json.

    ``activate_license`` writes to ``DEFAULT_LICENSE_PATH`` and
    ``deactivate_license`` unlinks it. Without this redirect, running the
    suite on a machine that has a real Pro key installed would overwrite
    and then delete that key. Pointing the path at a per-test tmp file
    keeps the suite hermetic and makes ``load_license`` deterministic
    (no stray file ⇒ Community).
    """
    fake = tmp_path / "license.json"
    monkeypatch.setattr("src.modules.licensing.DEFAULT_LICENSE_PATH", fake, raising=False)
    monkeypatch.setattr("src.modules.licensing.license.DEFAULT_LICENSE_PATH", fake, raising=False)
    yield


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_image_path(temp_dir):
    """Create a sample test image path (placeholder)."""
    image_path = temp_dir / "test_image.jpg"
    # Note: Actual image creation would require pillow
    return image_path


@pytest.fixture
def mock_database(temp_dir):
    """Create a mock database for testing."""
    from src.modules.storage.database import Database

    db_path = temp_dir / "test.db"
    return Database(db_path)
