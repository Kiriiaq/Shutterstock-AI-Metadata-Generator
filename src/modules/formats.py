"""Single source of truth for supported image formats.

Covers the classic microstock formats (JPEG/PNG/TIFF) plus the
smartphone-era containers: HEIC/HEIF (iPhone, Samsung), AVIF, WebP and
DNG (smartphone RAW, e.g. Samsung Expert RAW).

Three orthogonal capabilities are tracked per format:

- **scan** — the app lists and analyzes the file (`SUPPORTED_EXTENSIONS`).
- **IPTC IIM write** — legacy IPTC block support. HEIC/AVIF/WebP have no
  IIM container, so writers must fall back to XMP + EXIF there.
- **AI conversion** — Ollama vision models only ingest JPEG/PNG, so
  every other format is transparently re-encoded to JPEG in memory
  before hitting the API.

Pillow cannot open HEIC/HEIF/AVIF natively; the optional
``pillow-heif`` wheel registers openers for both. ``ensure_pillow_plugins``
activates it when present and degrades silently when absent (heuristic
paths keep working through ExifTool alone — "heuristique d'abord").
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Formats the app scans and analyzes. Keep lowercase, dot-prefixed.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Classic microstock trio
        ".jpg",
        ".jpeg",
        ".jpe",
        ".jfif",
        ".png",
        ".tif",
        ".tiff",
        # Smartphone containers (iPhone / Samsung / Google)
        ".heic",
        ".heif",
        ".hif",
        ".avif",
        ".webp",
        # Smartphone RAW (Samsung Expert RAW, Pixel RAW…)
        ".dng",
    }
)

# Formats whose container accepts a legacy IPTC IIM block. Everything
# else gets metadata via XMP (+ EXIF where applicable) — ExifTool
# refuses IPTC group writes on HEIC/AVIF/WebP.
IPTC_IIM_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".tif", ".tiff", ".dng"})

# Formats Ollama's vision endpoint accepts as-is. Anything not in this
# set is converted to JPEG in memory before base64 encoding.
AI_NATIVE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".jpe", ".jfif", ".png"})

_pillow_plugins_loaded: bool | None = None


def ensure_pillow_plugins() -> bool:
    """Register the pillow-heif openers (HEIC/HEIF/AVIF) once.

    Returns True when the plugin is active, False when pillow-heif is
    not installed. Safe to call repeatedly.
    """
    global _pillow_plugins_loaded
    if _pillow_plugins_loaded is not None:
        return _pillow_plugins_loaded
    try:
        import warnings

        import pillow_heif

        pillow_heif.register_heif_opener()
        # AVIF ships as a separate opener; pillow-heif marks it
        # deprecated (moving to pillow-avif-plugin). Register it while
        # it exists, silencing the deprecation noise — it becomes a
        # no-op once the attribute is dropped.
        if hasattr(pillow_heif, "register_avif_opener"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                pillow_heif.register_avif_opener()
        _pillow_plugins_loaded = True
        logger.debug("pillow-heif openers registered (HEIC/HEIF/AVIF)")
    except ImportError:
        _pillow_plugins_loaded = False
        logger.info("pillow-heif absent — HEIC/AVIF: métadonnées via ExifTool uniquement")
    return _pillow_plugins_loaded


def is_supported(path: Path | str) -> bool:
    """True when the file extension is scannable by the app."""
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def supports_iptc_iim(path: Path | str) -> bool:
    """True when the container accepts legacy IPTC IIM writes."""
    return Path(path).suffix.lower() in IPTC_IIM_EXTENSIONS


def needs_ai_conversion(path: Path | str) -> bool:
    """True when the file must be re-encoded to JPEG for the vision API."""
    return Path(path).suffix.lower() not in AI_NATIVE_EXTENSIONS


def convert_to_jpeg_bytes(path: Path | str, *, quality: int = 90, max_side: int = 2048) -> bytes:
    """Re-encode any Pillow-openable image to JPEG bytes for the AI.

    Downscales to ``max_side`` on the long edge — vision models don't
    benefit from full resolution and smaller payloads speed up Ollama.

    Raises whatever Pillow raises when the format can't be opened
    (e.g. HEIC without pillow-heif); callers surface that as a normal
    analysis failure for the file.
    """
    import io

    from PIL import Image

    ensure_pillow_plugins()

    with Image.open(Path(path)) as img:
        img = img.convert("RGB")
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality, optimize=True)
        return buf.getvalue()
