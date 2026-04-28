"""
Tests for configuration management.
"""

from src.core.params import PARAMS_META, ShutterstockParams


class TestShutterstockParams:
    """Tests for ShutterstockParams dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = ShutterstockParams()
        assert params.source_folder == ""
        assert params.prefilter_enabled is True
        assert params.model_name == "llama3.2-vision:11b"
        assert params.min_megapixels == 4.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        params = ShutterstockParams(source_folder="/test/path")
        data = params.to_dict()
        assert isinstance(data, dict)
        assert data["source_folder"] == "/test/path"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"source_folder": "/test/path", "debug_mode": True}
        params = ShutterstockParams.from_dict(data)
        assert params.source_folder == "/test/path"
        assert params.debug_mode is True

    def test_from_dict_ignores_invalid_keys(self):
        """Test that invalid keys are ignored."""
        data = {"source_folder": "/test", "invalid_key": "value"}
        params = ShutterstockParams.from_dict(data)
        assert params.source_folder == "/test"
        assert not hasattr(params, "invalid_key")


class TestParamMeta:
    """Tests for parameter metadata."""

    def test_params_meta_exists(self):
        """Test that PARAMS_META is defined."""
        assert PARAMS_META is not None
        assert len(PARAMS_META) > 0

    def test_source_folder_meta(self):
        """Test source_folder metadata."""
        meta = PARAMS_META.get("source_folder")
        assert meta is not None
        assert meta.label == "Dossier source"
        assert meta.category == "essential"
