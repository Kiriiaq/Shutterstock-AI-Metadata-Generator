"""
MetadataWriter - Write EXIF, IPTC, and XMP metadata to image files
Uses ExifTool for comprehensive metadata writing

Two write semantics coexist:

- **Additive** (historical): only fields carrying a value are sent to
  ExifTool. Empty/None fields are left untouched in the file.
- **Authoritative** (`write_editor_fields`): the editor's field set is
  written as the new truth — an empty field emits a deletion arg
  (``-TAG=``) and the XMP + EXIF mirror tags are kept in sync, so a
  re-read always matches what the user last saw in the editor.

Format awareness: HEIC/HEIF/AVIF/WebP have no IPTC IIM container, so
IPTC group args are skipped there and XMP + EXIF carry the data.
"""

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...utils.subprocess_helper import SUBPROCESS_NO_WINDOW
from ..analysis.limits import ADOBE_TITLE_MAX, smart_truncate
from ..formats import supports_iptc_iim
from ..models.metadata_models import ImageMetadata, IPTCFields, ShutterstockMetadata

logger = logging.getLogger(__name__)


class MetadataWriteError(Exception):
    """Raised when metadata cannot be written to file"""

    pass


@dataclass
class DryRunResult:
    """Result of a dry run operation"""

    file_path: Path
    would_write: bool
    args: List[str]
    existing_metadata: Optional[Dict[str, Any]] = None
    changes: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)


class MetadataWriter:
    """
    Writes EXIF, IPTC, and XMP metadata to image files using ExifTool
    Supports backup creation, rollback, and dry run mode
    """

    def __init__(self, exiftool_path: Optional[str] = None, create_backup: bool = True, dry_run: bool = False):
        """
        Initialize MetadataWriter

        Args:
            exiftool_path: Path to ExifTool executable
            create_backup: Whether to create backup files (_original)
            dry_run: If True, simulate writes without modifying files
        """
        self.exiftool_path = exiftool_path or self._find_exiftool()
        self.create_backup = create_backup
        self.dry_run = dry_run
        self._dry_run_results: List[DryRunResult] = []

        if not self.exiftool_path:
            raise MetadataWriteError("ExifTool not found. Please install from https://exiftool.org/")

    def _find_exiftool(self) -> Optional[str]:
        """Find ExifTool executable"""
        exiftool = shutil.which("exiftool")
        if exiftool:
            return exiftool

        common_paths = [
            "C:/Program Files/exiftool/exiftool.exe",
            "C:/Program Files (x86)/exiftool/exiftool.exe",
            "C:/exiftool/exiftool.exe",
            Path.home() / "exiftool" / "exiftool.exe",
        ]

        for path in common_paths:
            if Path(path).exists():
                return str(path)

        return None

    def write_iptc(self, file_path: Path, iptc: IPTCFields, preserve_existing: bool = True) -> bool:
        """
        Write IPTC metadata to an image file

        Args:
            file_path: Path to the image file
            iptc: IPTCFields object with metadata to write
            preserve_existing: If True, only update specified fields

        Returns:
            True if successful

        Raises:
            MetadataWriteError: If writing fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise MetadataWriteError(f"File not found: {file_path}")

        args = self._build_iptc_args(iptc)
        if not args:
            logger.warning("No IPTC fields to write")
            return True

        return self._run_exiftool_write(file_path, args)

    def write_editor_fields(self, file_path: Path, iptc: IPTCFields) -> bool:
        """Authoritative write of the IPTC editor's field set.

        Every editor-managed field (title, caption, keywords, byline,
        copyright) is written as the new truth: a value → set, empty →
        **deleted** via a bare ``-TAG=`` arg. The XMP and EXIF mirror
        tags are synchronized in the same call so that whichever group
        a reader prefers, it sees the same data. This is the fix for
        "cleared fields come back on re-read".

        On containers without IPTC IIM support (HEIC/AVIF/WebP), the
        IPTC group is skipped entirely and XMP + EXIF carry the data.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise MetadataWriteError(f"File not found: {file_path}")

        args = self._build_authoritative_args(file_path, iptc)
        return self._run_exiftool_write(file_path, args)

    def _build_authoritative_args(self, file_path: Path, iptc: IPTCFields) -> List[str]:
        """Set-or-delete args across IPTC + XMP + EXIF for editor fields."""
        write_iim = supports_iptc_iim(file_path)
        args: List[str] = []

        headline = (iptc.headline or iptc.object_name or "").strip()
        caption = (iptc.caption or "").strip()
        byline = (iptc.byline or "").strip()
        copyright_notice = (iptc.copyright_notice or "").strip()
        keywords = [k.strip() for k in (iptc.keywords or []) if k and k.strip()]

        def set_or_delete(tag: str, value: str, max_len: Optional[int] = None) -> None:
            if value:
                if max_len:
                    value = smart_truncate(value, max_len)
                args.append(f"-{tag}={value}")
            else:
                args.append(f"-{tag}=")

        # --- IPTC IIM (JPEG/TIFF/PNG/DNG only) ---
        if write_iim:
            set_or_delete("IPTC:Headline", headline, 256)
            set_or_delete("IPTC:ObjectName", headline, 64)
            set_or_delete("IPTC:Caption-Abstract", caption, 2000)
            set_or_delete("IPTC:By-line", byline, 32)
            set_or_delete("IPTC:CopyrightNotice", copyright_notice, 128)
            args.append("-IPTC:Keywords=")  # always clear, then re-add
            for kw in keywords:
                args.append(f"-IPTC:Keywords={kw}")

        # --- XMP mirrors (every format) ---
        set_or_delete("XMP-dc:Title", headline)
        set_or_delete("XMP-photoshop:Headline", headline)
        set_or_delete("XMP-dc:Description", caption)
        set_or_delete("XMP-dc:Creator", byline)
        set_or_delete("XMP-dc:Rights", copyright_notice)
        args.append("-XMP-dc:Subject=")
        for kw in keywords:
            args.append(f"-XMP-dc:Subject={kw}")

        # --- EXIF mirrors (readable by Windows Explorer & portals) ---
        set_or_delete("EXIF:ImageDescription", caption or headline)
        set_or_delete("EXIF:Artist", byline)
        set_or_delete("EXIF:Copyright", copyright_notice)
        set_or_delete("EXIF:XPTitle", headline)
        set_or_delete("EXIF:XPKeywords", "; ".join(keywords))

        return args

    def write_xmp(self, file_path: Path, xmp_data: Dict[str, Any]) -> bool:
        """
        Write XMP metadata to an image file

        Args:
            file_path: Path to the image file
            xmp_data: Dictionary of XMP fields to write

        Returns:
            True if successful
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise MetadataWriteError(f"File not found: {file_path}")

        args = self._build_xmp_args(xmp_data)
        if not args:
            return True

        return self._run_exiftool_write(file_path, args)

    def write_shutterstock_metadata(
        self, file_path: Path, metadata: ShutterstockMetadata, write_iptc: bool = True, write_xmp: bool = True
    ) -> bool:
        """
        Write Shutterstock metadata to an image file
        Writes to both IPTC and XMP for maximum compatibility

        Args:
            file_path: Path to the image file
            metadata: ShutterstockMetadata object
            write_iptc: Whether to write IPTC tags
            write_xmp: Whether to write XMP tags

        Returns:
            True if successful
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise MetadataWriteError(f"File not found: {file_path}")

        args = []
        title = smart_truncate(metadata.title or "", ADOBE_TITLE_MAX)
        description = smart_truncate(metadata.description or "", 2000)
        keywords = [k.strip() for k in metadata.keywords[:50] if k and k.strip()]

        # HEIC/AVIF/WebP have no IPTC IIM container — XMP carries it all.
        if write_iptc and not supports_iptc_iim(file_path):
            logger.debug("IPTC IIM unsupported for %s — writing XMP/EXIF only", file_path.suffix)
            write_iptc = False
            write_xmp = True

        if write_iptc:
            # IPTC fields
            args.extend(
                [
                    f"-IPTC:ObjectName={title[:64]}",  # IPTC limit
                    f"-IPTC:Caption-Abstract={description}",
                    f"-IPTC:Headline={title[:256]}",
                ]
            )

            # Keywords — clear first so re-runs replace instead of
            # accumulating duplicates, then one arg per keyword.
            args.append("-IPTC:Keywords=")
            for kw in keywords:
                args.append(f"-IPTC:Keywords={kw}")

            # Categories
            if metadata.categories:
                args.append(f"-IPTC:Category={metadata.categories[0][:3]}")
                args.append("-IPTC:SupplementalCategories=")
                for cat in metadata.categories:
                    args.append(f"-IPTC:SupplementalCategories={cat}")

        if write_xmp:
            # XMP Dublin Core
            args.extend(
                [
                    f"-XMP-dc:Title={title}",
                    f"-XMP-dc:Description={description}",
                ]
            )

            # XMP Keywords — one Subject entry per keyword (a joined
            # string would create a single bogus keyword on re-read).
            args.append("-XMP-dc:Subject=")
            for kw in keywords:
                args.append(f"-XMP-dc:Subject={kw}")

            # XMP Photoshop
            args.append(f"-XMP-photoshop:Headline={title}")

        # EXIF mirrors — visible in Windows Explorer and read by both
        # portals when IPTC/XMP are absent.
        args.append(f"-EXIF:ImageDescription={description or title}")
        args.append(f"-EXIF:XPTitle={title}")
        if keywords:
            args.append(f"-EXIF:XPKeywords={'; '.join(keywords)}")

        # Editorial flag in special instructions
        if metadata.editorial:
            if write_iptc:
                args.append("-IPTC:SpecialInstructions=EDITORIAL USE ONLY")
            args.append("-XMP-photoshop:Instructions=EDITORIAL USE ONLY")

        return self._run_exiftool_write(file_path, args)

    def write_from_image_metadata(
        self, file_path: Path, metadata: ImageMetadata, fields_to_write: Optional[List[str]] = None
    ) -> bool:
        """
        Write metadata from an ImageMetadata object back to file

        Args:
            file_path: Path to the image file
            metadata: ImageMetadata object
            fields_to_write: Specific fields to write, or None for all

        Returns:
            True if successful
        """
        file_path = Path(file_path)
        args = []

        # Write IPTC if present
        if metadata.iptc:
            iptc_args = self._build_iptc_args(metadata.iptc, fields_to_write)
            args.extend(iptc_args)

        # Write XMP rating/label if present
        if metadata.xmp_rating is not None:
            args.append(f"-XMP:Rating={metadata.xmp_rating}")
        if metadata.xmp_label:
            args.append(f"-XMP:Label={metadata.xmp_label}")
        if metadata.xmp_subject:
            for subject in metadata.xmp_subject:
                args.append(f"-XMP-dc:Subject={subject}")

        if not args:
            return True

        return self._run_exiftool_write(file_path, args)

    def _build_iptc_args(self, iptc: IPTCFields, fields_filter: Optional[List[str]] = None) -> List[str]:
        """Build ExifTool arguments for IPTC fields"""
        args = []

        field_mapping = {
            "object_name": ("IPTC:ObjectName", 64),
            "headline": ("IPTC:Headline", 256),
            "caption": ("IPTC:Caption-Abstract", 2000),
            "byline": ("IPTC:By-line", 32),
            "byline_title": ("IPTC:By-lineTitle", 32),
            "credit": ("IPTC:Credit", 32),
            "source": ("IPTC:Source", 32),
            "copyright_notice": ("IPTC:CopyrightNotice", 128),
            "city": ("IPTC:City", 32),
            "sublocation": ("IPTC:Sub-location", 32),
            "province_state": ("IPTC:Province-State", 32),
            "country_code": ("IPTC:Country-PrimaryLocationCode", 3),
            "country_name": ("IPTC:Country-PrimaryLocationName", 64),
            "category": ("IPTC:Category", 3),
            "urgency": ("IPTC:Urgency", None),
            "special_instructions": ("IPTC:SpecialInstructions", 256),
            "transmission_reference": ("IPTC:OriginalTransmissionReference", 32),
        }

        for field_name, (tag, max_len) in field_mapping.items():
            if fields_filter and field_name not in fields_filter:
                continue

            value = getattr(iptc, field_name, None)
            if value is not None:
                if max_len and isinstance(value, str):
                    value = value[:max_len]
                args.append(f"-{tag}={value}")

        # Handle keywords separately (multiple values)
        if (not fields_filter or "keywords" in fields_filter) and iptc.keywords:
            # First clear existing keywords
            args.append("-IPTC:Keywords=")
            for kw in iptc.keywords:
                args.append(f"-IPTC:Keywords={kw}")

        # Handle supplemental categories
        if (not fields_filter or "supplemental_categories" in fields_filter) and iptc.supplemental_categories:
            args.append("-IPTC:SupplementalCategories=")
            for cat in iptc.supplemental_categories:
                args.append(f"-IPTC:SupplementalCategories={cat}")

        # Handle date
        if (not fields_filter or "date_created" in fields_filter) and iptc.date_created:
            date_str = iptc.date_created.strftime("%Y%m%d")
            time_str = iptc.date_created.strftime("%H%M%S")
            args.append(f"-IPTC:DateCreated={date_str}")
            args.append(f"-IPTC:TimeCreated={time_str}")

        return args

    def _build_xmp_args(self, xmp_data: Dict[str, Any]) -> List[str]:
        """Build ExifTool arguments for XMP fields"""
        args = []

        xmp_mapping = {
            "title": "XMP-dc:Title",
            "description": "XMP-dc:Description",
            "creator": "XMP-dc:Creator",
            "rights": "XMP-dc:Rights",
            "rating": "XMP:Rating",
            "label": "XMP:Label",
            "subject": "XMP-dc:Subject",
            "headline": "XMP-photoshop:Headline",
            "city": "XMP-photoshop:City",
            "state": "XMP-photoshop:State",
            "country": "XMP-photoshop:Country",
            "instructions": "XMP-photoshop:Instructions",
        }

        # Loop variable named xmp_field to avoid shadowing the dataclasses
        # `field` import at module level.
        for xmp_field, tag in xmp_mapping.items():
            if xmp_field in xmp_data and xmp_data[xmp_field] is not None:
                value = xmp_data[xmp_field]
                if isinstance(value, list):
                    # Clear first, then add each
                    args.append(f"-{tag}=")
                    for v in value:
                        args.append(f"-{tag}={v}")
                else:
                    args.append(f"-{tag}={value}")

        return args

    def _run_exiftool_write(self, file_path: Path, args: List[str]) -> bool:
        """Execute ExifTool write command"""

        # DRY RUN MODE: Simulate without writing
        if self.dry_run:
            dry_result = DryRunResult(
                file_path=file_path,
                would_write=True,
                args=args.copy(),
                changes=[f"Would set {arg}" for arg in args if "=" in arg],
            )
            self._dry_run_results.append(dry_result)
            logger.info(f"[DRY RUN] Would write to: {file_path}")
            logger.debug(f"[DRY RUN] Args: {args}")
            return True

        cmd = [self.exiftool_path]

        # Backup option
        if not self.create_backup:
            cmd.append("-overwrite_original")

        # Encoding
        cmd.extend(["-charset", "utf8", "-charset", "iptc=utf8"])

        # Add our arguments
        cmd.extend(args)

        # Add file path
        cmd.append(str(file_path))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=60,
                **SUBPROCESS_NO_WINDOW,
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip()
                if "1 image files updated" not in result.stdout:
                    raise MetadataWriteError(f"ExifTool write failed: {error_msg}")

            logger.info(f"Metadata written to: {file_path}")
            return True

        except subprocess.TimeoutExpired as exc:
            raise MetadataWriteError(f"ExifTool timeout writing to: {file_path}") from exc
        except Exception as e:
            raise MetadataWriteError(f"Failed to write metadata: {e}") from e

    def get_dry_run_results(self) -> List[DryRunResult]:
        """Get results from dry run operations"""
        return self._dry_run_results.copy()

    def clear_dry_run_results(self):
        """Clear dry run results"""
        self._dry_run_results.clear()

    def set_dry_run(self, enabled: bool):
        """Enable or disable dry run mode"""
        self.dry_run = enabled
        if enabled:
            self._dry_run_results.clear()
            logger.info("Dry run mode ENABLED - no files will be modified")

    def write_metadata_auto(
        self,
        file_path: Path,
        iptc: Optional[IPTCFields] = None,
        xmp_data: Optional[Dict[str, Any]] = None,
        write_iptc: bool = True,
        write_xmp: bool = True,
    ) -> bool:
        """
        Write IPTC and/or XMP metadata directly into the file.

        The historical RAW branch (XMP sidecar suite) was removed in the
        2026-06-12 audit: the app only ever scans JPEG/PNG/TIFF, so the
        sidecar path was unreachable. Restore from git if RAW support
        comes back.

        Args:
            file_path: Path to the image file
            iptc: IPTC fields to write
            xmp_data: XMP data to write
            write_iptc: Write IPTC tags
            write_xmp: Write XMP tags

        Returns:
            True if successful
        """
        file_path = Path(file_path)
        success = True

        if write_iptc and iptc:
            success = self.write_iptc(file_path, iptc) and success

        if write_xmp and xmp_data:
            success = self.write_xmp(file_path, xmp_data) and success

        return success
