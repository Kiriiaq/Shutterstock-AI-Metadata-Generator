"""
Tests for validators module.
"""

from src.utils.validators import (
    validate_image_dimensions,
    validate_metadata_completeness,
)


class TestValidateImageDimensions:
    """Tests for image dimension validation."""

    def test_valid_dimensions(self):
        """Test valid image dimensions."""
        is_valid, error = validate_image_dimensions(3000, 2000, min_megapixels=4.0)
        assert is_valid is True
        assert error is None

    def test_invalid_dimensions(self):
        """Test invalid image dimensions."""
        is_valid, error = validate_image_dimensions(1000, 1000, min_megapixels=4.0)
        assert is_valid is False
        assert "too low" in error.lower()


class TestValidateMetadataCompleteness:
    """Tests for metadata completeness validation."""

    def test_complete_metadata(self):
        """Test complete metadata."""
        result = validate_metadata_completeness(
            title="Beautiful mountain landscape at sunset",
            description="A scenic view of mountains with golden sunlight",
            keywords=["mountain", "landscape", "sunset", "nature", "scenic", "outdoor", "beautiful"],
            categories=["Nature", "Landscape"],
        )
        assert result.completeness_score == 100

    def test_missing_title(self):
        """Test missing title."""
        result = validate_metadata_completeness(
            title=None, keywords=["test1", "test2", "test3", "test4", "test5", "test6", "test7"]
        )
        assert result.is_valid is False
        assert any("title" in e.lower() for e in result.errors)

    def test_insufficient_keywords(self):
        """Test insufficient keywords."""
        result = validate_metadata_completeness(title="Test Title", keywords=["keyword1", "keyword2"])
        assert result.is_valid is False
        assert any("keyword" in e.lower() for e in result.errors)
