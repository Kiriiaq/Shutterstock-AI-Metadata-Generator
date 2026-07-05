"""Platform compliance checks — Adobe Stock + Shutterstock.

Posture: **lax**. Every limit is reported as a warning, not an error.
Adobe and Shutterstock reviewers run their own QA pipeline; we don't
need to second-guess them. The goal here is to give the contributor
a heads-up before upload, not to gate the upload.

Rules implemented:

Adobe Stock
    - JPEG only (or PNG for graphic resources — we accept both)
    - sRGB color space (or unknown — we don't block)
    - 4 MP minimum, 100 MP maximum
    - 45 MB file size maximum

Shutterstock
    - JPEG only (TIFF/PNG flagged but not blocked)
    - 4 MP minimum (no documented maximum)
    - 50 MB file size maximum

Each check returns a *PlatformCompliance* with booleans and a list of
human-readable warnings (French). Empty warnings = clean.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


ADOBE_MAX_FILE_MB = 45.0
SHUTTERSTOCK_MAX_FILE_MB = 50.0
MIN_MEGAPIXELS = 4.0
ADOBE_MAX_MEGAPIXELS = 100.0


@dataclass
class PlatformCompliance:
    """Summary of compliance checks for one image, both platforms."""

    file_path: Path
    file_size_mb: float = 0.0
    megapixels: float = 0.0
    format: str = ""
    color_space: str = ""

    adobe_ready: bool = True
    shutterstock_ready: bool = True
    adobe_warnings: List[str] = field(default_factory=list)
    shutterstock_warnings: List[str] = field(default_factory=list)

    def all_warnings(self) -> List[str]:
        """Deduplicated warnings across both platforms."""
        seen, out = set(), []
        for w in self.adobe_warnings + self.shutterstock_warnings:
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out


def _probe_file(path: Path) -> Tuple[float, float, str, str]:
    """Best-effort probe of an image file.

    Returns (file_size_mb, megapixels, format, color_space). Any error
    yields zeros + empty strings — the compliance call still runs but
    every check that needs the missing field is silently skipped.

    PIL is used because ExifTool may not be installed; we avoid any
    hard dependency on the metadata_reader path. Pillow is already in
    requirements.txt.
    """
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
    except OSError as e:
        logger.debug("stat failed on %s: %s", path, e)
        size_mb = 0.0

    fmt = path.suffix.lstrip(".").lower()
    mp = 0.0
    color_space = ""

    try:
        from PIL import Image

        from ..formats import ensure_pillow_plugins

        # HEIC/HEIF/AVIF need the pillow-heif openers; no-op otherwise.
        ensure_pillow_plugins()

        with Image.open(path) as img:
            mp = (img.width * img.height) / 1_000_000
            # PIL.Image.mode gives us a coarse color space hint
            # ("RGB", "CMYK", "P", "L"...). It's not the EXIF
            # ColorSpace tag, but it's enough to flag non-sRGB.
            mode_to_space = {
                "RGB": "sRGB",
                "RGBA": "sRGB",
                "L": "Grayscale",
                "CMYK": "CMYK",
                "P": "Palette",
                "1": "Bitmap",
            }
            color_space = mode_to_space.get(img.mode, img.mode or "")
            if not fmt or fmt == "jpg":
                # PIL's format is "JPEG" not "jpg"; normalise once.
                fmt = (img.format or fmt).lower()
    except Exception as e:  # noqa: BLE001 — PIL raises a zoo of types
        logger.debug("PIL probe failed on %s: %s", path, e)

    return size_mb, mp, fmt, color_space


def check_adobe_compliance(
    path: Path,
    *,
    size_mb: Optional[float] = None,
    megapixels: Optional[float] = None,
    fmt: Optional[str] = None,
    color_space: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """Adobe Stock compliance check. Returns (ready, warnings)."""
    warnings: List[str] = []

    if megapixels is not None and megapixels > 0:
        if megapixels < MIN_MEGAPIXELS:
            warnings.append(f"Adobe : résolution {megapixels:.1f} MP < {MIN_MEGAPIXELS} MP minimum")
        elif megapixels > ADOBE_MAX_MEGAPIXELS:
            warnings.append(f"Adobe : résolution {megapixels:.1f} MP > {ADOBE_MAX_MEGAPIXELS} MP maximum")

    if size_mb is not None and size_mb > ADOBE_MAX_FILE_MB:
        warnings.append(f"Adobe : poids {size_mb:.1f} Mo > {ADOBE_MAX_FILE_MB} Mo")

    if fmt:
        fmt_norm = fmt.lower().lstrip(".")
        if fmt_norm in {"jpg"}:
            fmt_norm = "jpeg"
        if fmt_norm not in {"jpeg", "png"}:
            warnings.append(f"Adobe : format {fmt_norm or '?'} (JPEG recommandé)")

    if color_space and color_space not in {"sRGB", ""}:
        if color_space == "CMYK":
            warnings.append("Adobe : espace CMJN détecté (sRGB attendu)")
        else:
            warnings.append(f"Adobe : espace colorimétrique {color_space} (sRGB attendu)")

    # Lax posture — we never set ready=False on dimension/format alone.
    # The only blocker we hard-flag would be a corrupted file, but
    # that case is already handled upstream in validators.py. So the
    # contributor sees the warning and decides.
    return True, warnings


def check_shutterstock_compliance(
    path: Path,
    *,
    size_mb: Optional[float] = None,
    megapixels: Optional[float] = None,
    fmt: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """Shutterstock compliance check. Returns (ready, warnings)."""
    warnings: List[str] = []

    if megapixels is not None and megapixels > 0 and megapixels < MIN_MEGAPIXELS:
        warnings.append(f"Shutterstock : résolution {megapixels:.1f} MP < {MIN_MEGAPIXELS} MP minimum")

    if size_mb is not None and size_mb > SHUTTERSTOCK_MAX_FILE_MB:
        warnings.append(f"Shutterstock : poids {size_mb:.1f} Mo > {SHUTTERSTOCK_MAX_FILE_MB} Mo")

    if fmt:
        fmt_norm = fmt.lower().lstrip(".")
        if fmt_norm == "jpg":
            fmt_norm = "jpeg"
        if fmt_norm not in {"jpeg"}:
            # Shutterstock historically only accepts JPEG for photos.
            # TIFF/PNG aren't outright rejected (illustration/vector
            # paths exist), but we surface the friction.
            warnings.append(f"Shutterstock : format {fmt_norm or '?'} (JPEG recommandé)")

    return True, warnings


def check_platform_compliance(file_path: Path) -> PlatformCompliance:
    """Run both compliance checks on one file. Single PIL probe."""
    path = Path(file_path)
    size_mb, mp, fmt, cs = _probe_file(path)

    pc = PlatformCompliance(
        file_path=path,
        file_size_mb=size_mb,
        megapixels=mp,
        format=fmt,
        color_space=cs,
    )

    pc.adobe_ready, pc.adobe_warnings = check_adobe_compliance(
        path, size_mb=size_mb, megapixels=mp, fmt=fmt, color_space=cs
    )
    pc.shutterstock_ready, pc.shutterstock_warnings = check_shutterstock_compliance(
        path, size_mb=size_mb, megapixels=mp, fmt=fmt
    )

    return pc
