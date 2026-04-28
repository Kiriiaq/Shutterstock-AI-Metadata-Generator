"""
File utilities for image processing.
"""

import fnmatch
import hashlib
import logging
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ValidationErrorCode(Enum):
    """Standardized validation error codes."""

    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    PERMISSION_DENIED_READ = "PERMISSION_DENIED_READ"
    PERMISSION_DENIED_WRITE = "PERMISSION_DENIED_WRITE"
    FILE_EMPTY = "FILE_EMPTY"
    FILE_CORRUPTED = "FILE_CORRUPTED"
    INVALID_FORMAT = "INVALID_FORMAT"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INSUFFICIENT_SPACE = "INSUFFICIENT_SPACE"
    DIRECTORY_NOT_FOUND = "DIRECTORY_NOT_FOUND"
    ENCODING_ERROR = "ENCODING_ERROR"


@dataclass
class FileValidationResult:
    """Structured file validation result."""

    is_valid: bool
    error_code: Optional[ValidationErrorCode] = None
    error_message: Optional[str] = None
    warnings: List[str] = None
    file_info: Dict[str, Any] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.file_info is None:
            self.file_info = {}


# Supported image extensions
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".tif", ".tiff", ".png", ".eps"]


def is_valid_image_extension(file_path: Path) -> bool:
    """
    Check if file has a valid image extension.

    Args:
        file_path: Path to file

    Returns:
        True if extension is valid
    """
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    return file_path.stat().st_size / (1024 * 1024)


def compute_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """
    Compute SHA-256 hash of a file.

    Args:
        file_path: Path to file
        chunk_size: Read chunk size

    Returns:
        Hex string of file hash
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)

    return sha256.hexdigest()


def collect_image_files(
    directory: Path,
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
    exclude_extensions: Optional[List[str]] = None,
    exclude_folders: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[Path]:
    """
    Collect all image files from a directory with filtering options.

    Args:
        directory: Source directory
        recursive: Search subdirectories
        extensions: File extensions to include (default: common image formats)
        exclude_extensions: File extensions to exclude
        exclude_folders: Folder names to exclude (e.g., ['_backup', 'thumbs', 'cache'])
        exclude_patterns: Glob patterns to exclude (e.g., ['*_thumb.*', '*.bak'])

    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = IMAGE_EXTENSIONS

    if exclude_extensions is None:
        exclude_extensions = []

    if exclude_folders is None:
        exclude_folders = []

    if exclude_patterns is None:
        exclude_patterns = []

    # Normalize exclude folders to lowercase for comparison
    exclude_folders_lower = [f.lower() for f in exclude_folders]

    directory = Path(directory)
    files = []

    pattern = "**/*" if recursive else "*"

    for ext in extensions:
        # Skip if in exclude list
        if ext.lower() in [e.lower() for e in exclude_extensions]:
            continue

        for file_path in directory.glob(f"{pattern}{ext}"):
            files.append(file_path)
        for file_path in directory.glob(f"{pattern}{ext.upper()}"):
            files.append(file_path)

    # Filter out excluded folders
    if exclude_folders_lower:
        filtered_files = []
        for file_path in files:
            # Check if any parent folder is in exclude list
            skip = False
            for parent in file_path.parents:
                if parent.name.lower() in exclude_folders_lower:
                    skip = True
                    break
            if not skip:
                filtered_files.append(file_path)
        files = filtered_files

    # Filter out excluded patterns
    if exclude_patterns:
        filtered_files = []
        for file_path in files:
            skip = False
            for pat in exclude_patterns:
                if fnmatch.fnmatch(file_path.name.lower(), pat.lower()):
                    skip = True
                    break
            if not skip:
                filtered_files.append(file_path)
        files = filtered_files

    # Remove duplicates and sort
    files = sorted(set(files))

    logger.info(f"Found {len(files)} image files in {directory}")
    return files


# Default stopwords for keyword cleaning
DEFAULT_STOPWORDS: Set[str] = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "been",
    "be",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "what",
    "which",
    "who",
    "whom",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "every",
    "both",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "not",
    "only",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "also",
    "now",
    "here",
    "there",
    "then",
    "once",
    "image",
    "photo",
    "picture",
    "stock",
    "shutterstock",
    "photography",
}


def clean_keywords(
    keywords: List[str],
    stopwords: Optional[Set[str]] = None,
    blacklist: Optional[Set[str]] = None,
    min_length: int = 2,
    max_length: int = 64,
    max_keywords: int = 50,
    remove_duplicates: bool = True,
    lowercase: bool = True,
) -> List[str]:
    """
    Clean and filter keywords list.

    Args:
        keywords: List of keywords to clean
        stopwords: Set of words to remove (default: DEFAULT_STOPWORDS)
        blacklist: Set of forbidden words to remove
        min_length: Minimum keyword length
        max_length: Maximum keyword length
        max_keywords: Maximum number of keywords to return
        remove_duplicates: Remove duplicate keywords
        lowercase: Convert to lowercase

    Returns:
        Cleaned list of keywords
    """
    import re

    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS

    if blacklist is None:
        blacklist = set()

    # Combine stopwords and blacklist
    excluded_words = stopwords | blacklist

    cleaned = []
    seen: Set[str] = set()

    for kw in keywords:
        # Normalize
        if lowercase:
            kw = kw.lower()
        kw = kw.strip()

        # Remove special characters except hyphen and space
        kw = re.sub(r"[^\w\s-]", "", kw)

        # Normalize whitespace
        kw = re.sub(r"\s+", " ", kw).strip()

        # Skip if too short or too long
        if len(kw) < min_length or len(kw) > max_length:
            continue

        # Skip if in stopwords or blacklist
        if kw in excluded_words:
            continue

        # Skip duplicates
        if remove_duplicates:
            if kw in seen:
                continue
            seen.add(kw)

        cleaned.append(kw)

        # Stop if we have enough
        if len(cleaned) >= max_keywords:
            break

    return cleaned


def validate_disk_space(path: str, required_mb: float = 100.0) -> FileValidationResult:
    """
    Check if there's enough disk space for operations.

    Args:
        path: Path to check
        required_mb: Required space in MB

    Returns:
        FileValidationResult with space info
    """
    try:
        path_obj = Path(path)
        check_path = path_obj if path_obj.exists() else path_obj.parent
        while not check_path.exists() and check_path.parent != check_path:
            check_path = check_path.parent

        usage = shutil.disk_usage(str(check_path))
        free_mb = usage.free / (1024 * 1024)

        if free_mb < required_mb:
            return FileValidationResult(
                is_valid=False,
                error_code=ValidationErrorCode.INSUFFICIENT_SPACE,
                error_message=f"Insufficient disk space: {free_mb:.1f} MB available, {required_mb:.1f} MB required",
                file_info={"free_mb": free_mb, "required_mb": required_mb},
            )

        return FileValidationResult(
            is_valid=True,
            file_info={
                "free_mb": free_mb,
                "total_mb": usage.total / (1024 * 1024),
                "used_mb": usage.used / (1024 * 1024),
            },
        )

    except Exception as e:
        return FileValidationResult(
            is_valid=False,
            error_code=ValidationErrorCode.PERMISSION_DENIED_READ,
            error_message=f"Cannot check disk space: {e}",
        )


def read_file_with_encoding_fallback(
    file_path: str, encodings: List[str] = None
) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Read a text file with automatic encoding detection.

    Args:
        file_path: Path to text file
        encodings: List of encodings to try

    Returns:
        Tuple of (content, detected_encoding, error_message)
    """
    if encodings is None:
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    path = Path(file_path)

    if not path.exists():
        return None, "", f"File not found: {file_path}"

    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                content = f.read()
            return content, encoding, None
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return None, "", f"Error reading file: {e}"

    return None, "", f"Cannot decode file with encodings: {encodings}"


def validate_output_directory(dir_path: str, required_space_mb: float = 100.0) -> FileValidationResult:
    """
    Validate output directory is usable.

    Args:
        dir_path: Output directory path
        required_space_mb: Required space in MB

    Returns:
        FileValidationResult
    """
    path = Path(dir_path)

    # Create if doesn't exist
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            return FileValidationResult(
                is_valid=False,
                error_code=ValidationErrorCode.PERMISSION_DENIED_WRITE,
                error_message=f"Cannot create directory: {dir_path}",
            )
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                error_code=ValidationErrorCode.DIRECTORY_NOT_FOUND,
                error_message=f"Error creating directory: {e}",
            )

    # Check it's a directory
    if not path.is_dir():
        return FileValidationResult(
            is_valid=False, error_code=ValidationErrorCode.INVALID_FORMAT, error_message=f"Not a directory: {dir_path}"
        )

    # Check write permission
    if not os.access(path, os.W_OK):
        return FileValidationResult(
            is_valid=False,
            error_code=ValidationErrorCode.PERMISSION_DENIED_WRITE,
            error_message=f"Permission denied (write): {dir_path}",
        )

    # Check disk space
    return validate_disk_space(str(path), required_space_mb)
