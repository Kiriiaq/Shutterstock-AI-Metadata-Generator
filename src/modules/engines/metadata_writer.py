"""
MetadataWriter - Write EXIF, IPTC, and XMP metadata to image files
Uses ExifTool for comprehensive metadata writing
"""

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

        if write_iptc:
            # IPTC fields
            args.extend(
                [
                    f"-IPTC:ObjectName={metadata.title[:64]}",  # IPTC limit
                    f"-IPTC:Caption-Abstract={metadata.description}",
                    f"-IPTC:Headline={metadata.title[:256]}",
                ]
            )

            # Keywords (IPTC supports multiple -Keywords tags)
            for kw in metadata.keywords[:50]:
                args.append(f"-IPTC:Keywords={kw}")

            # Categories
            if metadata.categories:
                args.append(f"-IPTC:Category={metadata.categories[0][:3]}")
                for cat in metadata.categories:
                    args.append(f"-IPTC:SupplementalCategories={cat}")

        if write_xmp:
            # XMP Dublin Core
            args.extend(
                [
                    f"-XMP-dc:Title={metadata.title}",
                    f"-XMP-dc:Description={metadata.description}",
                ]
            )

            # XMP Keywords
            keywords_str = ", ".join(metadata.keywords)
            args.append(f"-XMP-dc:Subject={keywords_str}")

            # XMP Photoshop
            args.append(f"-XMP-photoshop:Headline={metadata.title}")

        # Editorial flag in special instructions
        if metadata.editorial:
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

    def copy_metadata(self, source_path: Path, dest_path: Path, metadata_types: List[str] = None) -> bool:
        """
        Copy metadata from one file to another

        Args:
            source_path: Source file path
            dest_path: Destination file path
            metadata_types: List of types to copy ('exif', 'iptc', 'xmp'), or None for all

        Returns:
            True if successful
        """
        source_path = Path(source_path)
        dest_path = Path(dest_path)

        if not source_path.exists():
            raise MetadataWriteError(f"Source file not found: {source_path}")
        if not dest_path.exists():
            raise MetadataWriteError(f"Destination file not found: {dest_path}")

        args = ["-tagsFromFile", str(source_path)]

        if metadata_types:
            for mt in metadata_types:
                if mt.lower() == "exif":
                    args.append("-EXIF:all")
                elif mt.lower() == "iptc":
                    args.append("-IPTC:all")
                elif mt.lower() == "xmp":
                    args.append("-XMP:all")
        else:
            args.append("-all:all")

        return self._run_exiftool_write(dest_path, args)

    def clear_metadata(self, file_path: Path, metadata_types: List[str] = None) -> bool:
        """
        Clear metadata from a file

        Args:
            file_path: Path to the image file
            metadata_types: Types to clear ('exif', 'iptc', 'xmp'), or None for all

        Returns:
            True if successful
        """
        file_path = Path(file_path)

        args = []
        if metadata_types:
            for mt in metadata_types:
                if mt.lower() == "exif":
                    args.append("-EXIF:all=")
                elif mt.lower() == "iptc":
                    args.append("-IPTC:all=")
                elif mt.lower() == "xmp":
                    args.append("-XMP:all=")
        else:
            args.append("-all=")

        return self._run_exiftool_write(file_path, args)

    def write_batch(self, files_metadata: List[Tuple[Path, IPTCFields]]) -> List[Tuple[Path, bool, Optional[str]]]:
        """
        Write metadata to multiple files

        Args:
            files_metadata: List of (file_path, iptc_data) tuples

        Returns:
            List of (path, success, error_message) tuples
        """
        results = []

        for file_path, iptc in files_metadata:
            try:
                success = self.write_iptc(file_path, iptc)
                results.append((file_path, success, None))
            except Exception as e:
                results.append((file_path, False, str(e)))

        return results

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
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=60)

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

    def restore_backup(self, file_path: Path) -> bool:
        """
        Restore original file from backup

        Args:
            file_path: Path to the modified file

        Returns:
            True if backup was restored
        """
        file_path = Path(file_path)
        backup_path = file_path.with_suffix(file_path.suffix + "_original")

        if not backup_path.exists():
            logger.warning(f"No backup found for: {file_path}")
            return False

        try:
            # Remove modified file
            file_path.unlink()
            # Rename backup to original
            backup_path.rename(file_path)
            logger.info(f"Restored backup for: {file_path}")
            return True
        except Exception as e:
            raise MetadataWriteError(f"Failed to restore backup: {e}") from e

    def cleanup_backups(self, directory: Path, recursive: bool = False) -> int:
        """
        Remove all _original backup files in a directory

        Args:
            directory: Directory to clean
            recursive: Search subdirectories

        Returns:
            Number of backup files removed
        """
        directory = Path(directory)
        pattern = "*_original"

        if recursive:
            backup_files = list(directory.rglob(pattern))
        else:
            backup_files = list(directory.glob(pattern))

        count = 0
        for backup_file in backup_files:
            try:
                backup_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to remove backup {backup_file}: {e}")

        logger.info(f"Removed {count} backup files from {directory}")
        return count

    # ==================== XMP Sidecar Support ====================

    # RAW file extensions that require XMP sidecar
    RAW_EXTENSIONS = {
        ".cr2",
        ".cr3",  # Canon
        ".nef",
        ".nrw",  # Nikon
        ".arw",
        ".srf",
        ".sr2",  # Sony
        ".orf",  # Olympus
        ".rw2",  # Panasonic
        ".pef",
        ".dng",  # Pentax, Adobe DNG
        ".raf",  # Fujifilm
        ".raw",
        ".rwl",  # Leica (.rw2 already listed above for Panasonic)
        ".3fr",  # Hasselblad
        ".fff",  # Imacon
        ".iiq",  # Phase One
        ".srw",  # Samsung
        ".x3f",  # Sigma
        ".kdc",
        ".dcr",  # Kodak
        ".mrw",  # Minolta
        ".erf",  # Epson
    }

    def is_raw_file(self, file_path: Path) -> bool:
        """
        Check if a file is a RAW format

        Args:
            file_path: Path to the image file

        Returns:
            True if file is a RAW format
        """
        return file_path.suffix.lower() in self.RAW_EXTENSIONS

    def get_xmp_sidecar_path(self, file_path: Path) -> Path:
        """
        Get the XMP sidecar path for a file

        Args:
            file_path: Path to the image file

        Returns:
            Path to the XMP sidecar file
        """
        return file_path.with_suffix(".xmp")

    def xmp_sidecar_exists(self, file_path: Path) -> bool:
        """
        Check if XMP sidecar exists for a file

        Args:
            file_path: Path to the image file

        Returns:
            True if sidecar exists
        """
        return self.get_xmp_sidecar_path(file_path).exists()

    def create_xmp_sidecar(
        self,
        file_path: Path,
        iptc: Optional[IPTCFields] = None,
        xmp_data: Optional[Dict[str, Any]] = None,
        copy_from_raw: bool = True,
    ) -> Path:
        """
        Create XMP sidecar file for a RAW image

        Args:
            file_path: Path to the RAW image file
            iptc: Optional IPTC fields to write
            xmp_data: Optional XMP data to write
            copy_from_raw: Copy existing metadata from RAW file

        Returns:
            Path to the created XMP sidecar

        Raises:
            MetadataWriteError: If creation fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise MetadataWriteError(f"File not found: {file_path}")

        xmp_path = self.get_xmp_sidecar_path(file_path)

        # DRY RUN MODE
        if self.dry_run:
            dry_result = DryRunResult(
                file_path=xmp_path,
                would_write=True,
                args=["create_xmp_sidecar"],
                changes=[f"Would create XMP sidecar: {xmp_path}"],
            )
            self._dry_run_results.append(dry_result)
            logger.info(f"[DRY RUN] Would create XMP sidecar: {xmp_path}")
            return xmp_path

        # Step 1: Create sidecar from RAW (copies existing metadata)
        if copy_from_raw:
            cmd = [self.exiftool_path, "-o", str(xmp_path), "-charset", "utf8", str(file_path)]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=60)

                if result.returncode != 0 and "1 output files created" not in result.stdout:
                    # Create empty XMP sidecar
                    self._create_empty_xmp_sidecar(file_path, xmp_path)

            except Exception as e:
                logger.warning(f"Failed to copy from RAW, creating empty sidecar: {e}")
                self._create_empty_xmp_sidecar(file_path, xmp_path)
        else:
            self._create_empty_xmp_sidecar(file_path, xmp_path)

        # Step 2: Write additional metadata to sidecar
        if iptc or xmp_data:
            self.write_to_xmp_sidecar(file_path, iptc, xmp_data)

        logger.info(f"Created XMP sidecar: {xmp_path}")
        return xmp_path

    def _create_empty_xmp_sidecar(self, source_path: Path, xmp_path: Path):
        """Create an empty XMP sidecar file with basic structure"""
        # Get original filename for reference
        original_filename = source_path.name

        xmp_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Shutterstock AI Metadata Generator">
    <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
        <rdf:Description rdf:about=""
            xmlns:dc="http://purl.org/dc/elements/1.1/"
            xmlns:xmp="http://ns.adobe.com/xap/1.0/"
            xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"
            xmlns:Iptc4xmpCore="http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/"
            xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/"
            xmp:CreatorTool="Shutterstock AI Metadata Generator"
            xmpMM:OriginalDocumentID="{original_filename}">
        </rdf:Description>
    </rdf:RDF>
</x:xmpmeta>
'''
        with open(xmp_path, "w", encoding="utf-8") as f:
            f.write(xmp_content)

    def write_to_xmp_sidecar(
        self, file_path: Path, iptc: Optional[IPTCFields] = None, xmp_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Write metadata to XMP sidecar file

        Args:
            file_path: Path to the original image file
            iptc: IPTC fields to write
            xmp_data: XMP data to write

        Returns:
            True if successful
        """
        xmp_path = self.get_xmp_sidecar_path(file_path)

        # Create sidecar if it doesn't exist
        if not xmp_path.exists():
            self.create_xmp_sidecar(file_path, copy_from_raw=False)

        args = []

        # Convert IPTC to XMP format
        if iptc:
            if iptc.object_name:
                args.append(f"-XMP-dc:Title={iptc.object_name}")
            if iptc.headline:
                args.append(f"-XMP-photoshop:Headline={iptc.headline}")
            if iptc.caption:
                args.append(f"-XMP-dc:Description={iptc.caption}")
            if iptc.byline:
                args.append(f"-XMP-dc:Creator={iptc.byline}")
            if iptc.copyright_notice:
                args.append(f"-XMP-dc:Rights={iptc.copyright_notice}")
            if iptc.city:
                args.append(f"-XMP-photoshop:City={iptc.city}")
            if iptc.province_state:
                args.append(f"-XMP-photoshop:State={iptc.province_state}")
            if iptc.country_name:
                args.append(f"-XMP-photoshop:Country={iptc.country_name}")
            if iptc.country_code:
                args.append(f"-XMP-iptcCore:CountryCode={iptc.country_code}")
            if iptc.special_instructions:
                args.append(f"-XMP-photoshop:Instructions={iptc.special_instructions}")
            if iptc.credit:
                args.append(f"-XMP-photoshop:Credit={iptc.credit}")
            if iptc.source:
                args.append(f"-XMP-photoshop:Source={iptc.source}")

            # Keywords
            if iptc.keywords:
                args.append("-XMP-dc:Subject=")  # Clear first
                for kw in iptc.keywords:
                    args.append(f"-XMP-dc:Subject={kw}")

            # Categories as supplemental
            if iptc.supplemental_categories:
                args.append("-XMP-photoshop:SupplementalCategories=")
                for cat in iptc.supplemental_categories:
                    args.append(f"-XMP-photoshop:SupplementalCategories={cat}")

        # Add additional XMP data
        if xmp_data:
            xmp_args = self._build_xmp_args(xmp_data)
            args.extend(xmp_args)

        if not args:
            return True

        # Write to XMP sidecar
        return self._run_exiftool_write(xmp_path, args)

    def read_xmp_sidecar(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Read metadata from XMP sidecar file

        Args:
            file_path: Path to the original image file

        Returns:
            Dictionary of XMP metadata or None if sidecar doesn't exist
        """
        xmp_path = self.get_xmp_sidecar_path(file_path)

        if not xmp_path.exists():
            return None

        cmd = [self.exiftool_path, "-json", "-charset", "utf8", "-XMP:all", str(xmp_path)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=30)

            if result.returncode == 0:
                data = json.loads(result.stdout)
                if data and len(data) > 0:
                    return data[0]

        except Exception as e:
            logger.error(f"Failed to read XMP sidecar: {e}")

        return None

    def sync_sidecar_to_raw(self, file_path: Path) -> bool:
        """
        Sync XMP sidecar metadata back to RAW file (if supported)

        Args:
            file_path: Path to the RAW file

        Returns:
            True if successful
        """
        xmp_path = self.get_xmp_sidecar_path(file_path)

        if not xmp_path.exists():
            logger.warning(f"No XMP sidecar found for: {file_path}")
            return False

        # DRY RUN MODE
        if self.dry_run:
            dry_result = DryRunResult(
                file_path=file_path,
                would_write=True,
                args=["sync_sidecar_to_raw"],
                changes=[f"Would sync {xmp_path} to {file_path}"],
            )
            self._dry_run_results.append(dry_result)
            return True

        cmd = [self.exiftool_path, "-tagsFromFile", str(xmp_path), "-XMP:all", "-charset", "utf8"]

        if not self.create_backup:
            cmd.append("-overwrite_original")

        cmd.append(str(file_path))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=60)

            if "1 image files updated" in result.stdout:
                logger.info(f"Synced sidecar to RAW: {file_path}")
                return True

        except Exception as e:
            logger.error(f"Failed to sync sidecar: {e}")

        return False

    def delete_xmp_sidecar(self, file_path: Path) -> bool:
        """
        Delete XMP sidecar file

        Args:
            file_path: Path to the original image file

        Returns:
            True if deleted or didn't exist
        """
        xmp_path = self.get_xmp_sidecar_path(file_path)

        if not xmp_path.exists():
            return True

        # DRY RUN MODE
        if self.dry_run:
            dry_result = DryRunResult(
                file_path=xmp_path,
                would_write=True,
                args=["delete_xmp_sidecar"],
                changes=[f"Would delete XMP sidecar: {xmp_path}"],
            )
            self._dry_run_results.append(dry_result)
            return True

        try:
            xmp_path.unlink()
            logger.info(f"Deleted XMP sidecar: {xmp_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete sidecar: {e}")
            return False

    def write_metadata_auto(
        self,
        file_path: Path,
        iptc: Optional[IPTCFields] = None,
        xmp_data: Optional[Dict[str, Any]] = None,
        write_iptc: bool = True,
        write_xmp: bool = True,
    ) -> bool:
        """
        Automatically write metadata to file or XMP sidecar based on file type

        For RAW files: Creates/updates XMP sidecar
        For other files: Writes directly to file

        Args:
            file_path: Path to the image file
            iptc: IPTC fields to write
            xmp_data: XMP data to write
            write_iptc: Write IPTC tags (for non-RAW files)
            write_xmp: Write XMP tags

        Returns:
            True if successful
        """
        file_path = Path(file_path)

        if self.is_raw_file(file_path):
            # RAW file: Write to XMP sidecar
            logger.info(f"RAW file detected, writing to XMP sidecar: {file_path}")
            return self.write_to_xmp_sidecar(file_path, iptc, xmp_data)
        else:
            # Regular file: Write directly
            success = True

            if write_iptc and iptc:
                success = self.write_iptc(file_path, iptc) and success

            if write_xmp and xmp_data:
                success = self.write_xmp(file_path, xmp_data) and success

            return success
