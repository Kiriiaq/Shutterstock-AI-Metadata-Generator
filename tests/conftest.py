"""
Pytest configuration and fixtures.
"""

import shutil
import tempfile
from pathlib import Path

import pytest


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
