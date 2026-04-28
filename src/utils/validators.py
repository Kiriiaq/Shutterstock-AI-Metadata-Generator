"""
Validators for image and metadata validation.
Includes robust file validation with corruption detection.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Magic bytes for image format detection
IMAGE_SIGNATURES = {
    'jpeg': [b'\xFF\xD8\xFF'],
    'png': [b'\x89PNG\r\n\x1a\n'],
    'tiff_le': [b'II*\x00'],  # Little-endian TIFF
    'tiff_be': [b'MM\x00*'],  # Big-endian TIFF
    'gif': [b'GIF87a', b'GIF89a'],
    'bmp': [b'BM'],
    'webp': [b'RIFF'],  # WEBP (need to check for WEBP after RIFF)
}


def validate_image_file(
    file_path: Path,
    check_content: bool = True,
    min_size_bytes: int = 1024,
    max_size_mb: float = 100.0
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Comprehensive image file validation with corruption detection.

    Args:
        file_path: Path to image file
        check_content: Verify magic bytes and try opening with PIL
        min_size_bytes: Minimum file size (default 1KB, too small likely corrupt)
        max_size_mb: Maximum file size in MB

    Returns:
        Tuple of (is_valid, error_message, detected_format)
    """
    file_path = Path(file_path)

    # Check existence
    if not file_path.exists():
        return False, f"File not found: {file_path}", None

    if not file_path.is_file():
        return False, f"Not a file: {file_path}", None

    # Check read permission
    if not os.access(file_path, os.R_OK):
        return False, f"Permission denied: {file_path}", None

    # Check file size
    try:
        size = file_path.stat().st_size
    except Exception as e:
        return False, f"Cannot read file: {e}", None

    if size < min_size_bytes:
        return False, f"File too small ({size} bytes) - likely corrupted", None

    size_mb = size / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File too large: {size_mb:.1f} MB (max {max_size_mb} MB)", None

    # Check extension
    valid_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.gif', '.bmp', '.webp', '.eps']
    ext = file_path.suffix.lower()
    if ext not in valid_extensions:
        return False, f"Unsupported image format: {ext}", None

    if not check_content:
        return True, None, ext[1:]

    # Check magic bytes
    try:
        with open(file_path, 'rb') as f:
            header = f.read(16)

        detected_format = None

        # Check against known signatures
        for fmt, signatures in IMAGE_SIGNATURES.items():
            for sig in signatures:
                if header.startswith(sig):
                    detected_format = fmt
                    break
            if detected_format:
                break

        # Special check for WEBP (RIFF....WEBP)
        if header.startswith(b'RIFF') and b'WEBP' in header:
            detected_format = 'webp'

        # EPS files
        if header.startswith(b'%!PS') or header.startswith(b'\xC5\xD0\xD3\xC6'):
            detected_format = 'eps'

        if detected_format is None and ext not in ['.eps']:
            return False, "Unknown or corrupted image format (invalid header)", None

    except PermissionError:
        return False, f"Permission denied reading: {file_path}", None
    except Exception as e:
        return False, f"Error reading file header: {e}", None

    # Try opening with PIL for deeper validation
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            img.verify()  # Verify image integrity
    except ImportError:
        # PIL not available, skip deep validation
        logger.warning("PIL not available for deep image validation")
    except Exception as e:
        return False, f"Image corrupted or unreadable: {str(e)}", detected_format

    return True, None, detected_format


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool = True
    errors: List[str] = None
    warnings: List[str] = None
    completeness_score: float = 0.0
    quality_score: float = 0.0
    seo_score: float = 0.0

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


def validate_image_dimensions(
    width: int,
    height: int,
    min_megapixels: float = 4.0
) -> Tuple[bool, Optional[str]]:
    """
    Validate image dimensions meet minimum requirements.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        min_megapixels: Minimum megapixels required

    Returns:
        Tuple of (is_valid, error_message)
    """
    megapixels = (width * height) / 1_000_000

    if megapixels < min_megapixels:
        return False, f"Resolution too low: {megapixels:.1f} MP (min {min_megapixels} MP)"

    return True, None


def validate_file_size(
    file_path: Path,
    max_size_mb: float = 50.0
) -> Tuple[bool, Optional[str]]:
    """
    Validate file size is within limits.

    Args:
        file_path: Path to file
        max_size_mb: Maximum file size in MB

    Returns:
        Tuple of (is_valid, error_message)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return False, f"File not found: {file_path}"

    size_mb = file_path.stat().st_size / (1024 * 1024)

    if size_mb > max_size_mb:
        return False, f"File too large: {size_mb:.1f} MB (max {max_size_mb} MB)"

    return True, None


def validate_metadata_completeness(
    title: Optional[str] = None,
    description: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    categories: Optional[List[str]] = None
) -> ValidationResult:
    """
    Validate metadata completeness for Shutterstock submission.

    Args:
        title: Image title
        description: Image description
        keywords: List of keywords
        categories: List of categories

    Returns:
        ValidationResult with scores and any issues
    """
    result = ValidationResult()

    # Check required fields
    if not title:
        result.is_valid = False
        result.errors.append("Title is required")
    elif len(title) < 5:
        result.warnings.append("Title is very short")
    elif len(title) > 200:
        result.errors.append("Title exceeds 200 characters")

    if not description:
        result.warnings.append("Description is recommended")
    elif len(description) < 20:
        result.warnings.append("Description is very short")

    # Keywords validation
    if not keywords:
        result.is_valid = False
        result.errors.append("Keywords are required (minimum 7)")
    elif len(keywords) < 7:
        result.is_valid = False
        result.errors.append(f"Need at least 7 keywords, have {len(keywords)}")
    elif len(keywords) > 50:
        result.warnings.append(f"Too many keywords ({len(keywords)}), maximum is 50")

    # Check for keyword quality
    if keywords:
        short_keywords = [k for k in keywords if len(k) < 3]
        if short_keywords:
            result.warnings.append(f"Very short keywords: {', '.join(short_keywords)}")

        # Check for duplicates
        if len(set(keywords)) != len(keywords):
            result.warnings.append("Duplicate keywords found")

    # Calculate completeness score
    score = 0
    if title:
        score += 25
    if description:
        score += 25
    if keywords and len(keywords) >= 7:
        score += 30
    if categories:
        score += 20
    result.completeness_score = score

    # Calculate quality score
    quality = 50
    if title and 30 <= len(title) <= 150:
        quality += 15
    if title and not title.isupper():
        quality += 5
    if keywords and len(keywords) >= 20:
        quality += 15
    if keywords and len(set(keywords)) == len(keywords):
        quality += 10
    if description and len(description) >= 50:
        quality += 5
    result.quality_score = min(100, quality)

    # Calculate SEO score
    seo = 50
    if keywords:
        kw_count = len(keywords)
        if kw_count >= 30:
            seo += 20
        elif kw_count >= 20:
            seo += 15
        elif kw_count >= 10:
            seo += 10
    if categories and len(categories) == 2:
        seo += 10
    if title and keywords:
        title_lower = title.lower()
        matches = sum(1 for kw in keywords if kw.lower() in title_lower)
        if matches >= 2:
            seo += 20
        elif matches >= 1:
            seo += 10
    result.seo_score = min(100, seo)

    return result


def validate_shutterstock_requirements(
    file_path: Path,
    title: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> ValidationResult:
    """
    Validate all Shutterstock submission requirements.

    Args:
        file_path: Path to image file
        title: Image title
        keywords: List of keywords
        width: Image width
        height: Image height

    Returns:
        ValidationResult with all validation checks
    """
    result = ValidationResult()

    # File size validation
    is_valid, error = validate_file_size(file_path, max_size_mb=50.0)
    if not is_valid:
        result.is_valid = False
        result.errors.append(error)

    # Dimension validation
    if width and height:
        is_valid, error = validate_image_dimensions(width, height, min_megapixels=4.0)
        if not is_valid:
            result.is_valid = False
            result.errors.append(error)

    # Title validation
    if not title:
        result.is_valid = False
        result.errors.append("Title is required")
    elif len(title) > 200:
        result.errors.append("Title exceeds 200 characters")

    # Keywords validation
    if not keywords:
        result.is_valid = False
        result.errors.append("Keywords are required")
    elif len(keywords) < 7:
        result.is_valid = False
        result.errors.append(f"Minimum 7 keywords required, have {len(keywords)}")
    elif len(keywords) > 50:
        result.warnings.append("Maximum 50 keywords allowed")

    return result
