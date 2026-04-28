"""
MetadataReader - Extract EXIF, IPTC, and XMP metadata from image files
Uses ExifTool for comprehensive metadata extraction
"""

import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import logging
import re

from ..models.metadata_models import (
    ImageMetadata,
    IPTCFields,
    MetadataSource
)

logger = logging.getLogger(__name__)


class ExifToolNotFoundError(Exception):
    """Raised when ExifTool is not installed or not found"""
    pass


class MetadataReadError(Exception):
    """Raised when metadata cannot be read from file"""
    pass


class MetadataReader:
    """
    Reads EXIF, IPTC, and XMP metadata from image files using ExifTool
    """

    # ExifTool arguments for comprehensive metadata extraction
    EXIFTOOL_ARGS = [
        "-json",
        "-n",  # Numeric values
        "-G1",  # Group names
        "-charset", "utf8",
        "-api", "largefilesupport=1"
    ]

    # Mapping of EXIF tags to our fields
    EXIF_MAPPING = {
        "EXIF:Make": "camera_make",
        "EXIF:Model": "camera_model",
        "EXIF:LensModel": "lens_model",
        "EXIF:FocalLength": "focal_length",
        "EXIF:FNumber": "aperture",
        "EXIF:ExposureTime": "shutter_speed",
        "EXIF:ISO": "iso",
        "EXIF:Flash": "flash_fired",
        "EXIF:DateTimeOriginal": "date_taken",
        "EXIF:GPSLatitude": "gps_latitude",
        "EXIF:GPSLongitude": "gps_longitude",
        "EXIF:GPSAltitude": "gps_altitude",
        "EXIF:Orientation": "orientation",
        "EXIF:ImageWidth": "width",
        "EXIF:ImageHeight": "height",
        "EXIF:ColorSpace": "color_space",
        "EXIF:BitsPerSample": "bit_depth",
    }

    # IPTC tag mapping
    IPTC_MAPPING = {
        "IPTC:ObjectName": "object_name",
        "IPTC:Headline": "headline",
        "IPTC:Caption-Abstract": "caption",
        "IPTC:Keywords": "keywords",
        "IPTC:By-line": "byline",
        "IPTC:By-lineTitle": "byline_title",
        "IPTC:Credit": "credit",
        "IPTC:Source": "source",
        "IPTC:CopyrightNotice": "copyright_notice",
        "IPTC:City": "city",
        "IPTC:Sub-location": "sublocation",
        "IPTC:Province-State": "province_state",
        "IPTC:Country-PrimaryLocationCode": "country_code",
        "IPTC:Country-PrimaryLocationName": "country_name",
        "IPTC:Category": "category",
        "IPTC:SupplementalCategories": "supplemental_categories",
        "IPTC:Urgency": "urgency",
        "IPTC:DateCreated": "date_created",
        "IPTC:TimeCreated": "time_created",
        "IPTC:SpecialInstructions": "special_instructions",
        "IPTC:OriginalTransmissionReference": "transmission_reference",
    }

    # XMP tag mapping
    XMP_MAPPING = {
        "XMP:Rating": "xmp_rating",
        "XMP:Label": "xmp_label",
        "XMP:Subject": "xmp_subject",
        "XMP-dc:Title": "title",
        "XMP-dc:Description": "description",
        "XMP-dc:Subject": "keywords",
        "XMP-dc:Creator": "creator",
        "XMP-dc:Rights": "rights",
        "XMP-photoshop:City": "city",
        "XMP-photoshop:State": "province_state",
        "XMP-photoshop:Country": "country_name",
        "XMP-iptcCore:Location": "sublocation",
    }

    def __init__(self, exiftool_path: Optional[str] = None):
        """
        Initialize MetadataReader

        Args:
            exiftool_path: Path to ExifTool executable. Auto-detected if not provided.
        """
        self.exiftool_path = exiftool_path or self._find_exiftool()
        if not self.exiftool_path:
            raise ExifToolNotFoundError(
                "ExifTool not found. Please install it from https://exiftool.org/ "
                "and ensure it's in your PATH or provide the path explicitly."
            )

    def _find_exiftool(self) -> Optional[str]:
        """
        Find ExifTool in common locations

        Returns:
            Path to ExifTool executable or None if not found
        """
        # Check if in PATH
        exiftool = shutil.which("exiftool")
        if exiftool:
            return exiftool

        # Check common Windows locations
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

    def read(self, file_path: Path) -> ImageMetadata:
        """
        Read all metadata from an image file

        Args:
            file_path: Path to the image file

        Returns:
            ImageMetadata object with all extracted metadata

        Raises:
            MetadataReadError: If metadata cannot be read
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise MetadataReadError(f"File not found: {file_path}")

        # Run ExifTool
        raw_data = self._run_exiftool(file_path)

        # Parse into our structure
        metadata = self._parse_metadata(file_path, raw_data)

        return metadata

    def read_batch(self, file_paths: List[Path]) -> List[Tuple[Path, Optional[ImageMetadata], Optional[str]]]:
        """
        Read metadata from multiple files efficiently

        Args:
            file_paths: List of file paths

        Returns:
            List of tuples: (path, metadata or None, error or None)
        """
        results = []

        # ExifTool can process multiple files at once
        if not file_paths:
            return results

        try:
            raw_data_list = self._run_exiftool_batch(file_paths)

            for i, file_path in enumerate(file_paths):
                try:
                    if i < len(raw_data_list):
                        metadata = self._parse_metadata(file_path, raw_data_list[i])
                        results.append((file_path, metadata, None))
                    else:
                        results.append((file_path, None, "No data returned from ExifTool"))
                except Exception as e:
                    results.append((file_path, None, str(e)))

        except Exception as e:
            # If batch fails, fall back to individual processing
            logger.warning(f"Batch processing failed, falling back to individual: {e}")
            for file_path in file_paths:
                try:
                    metadata = self.read(file_path)
                    results.append((file_path, metadata, None))
                except Exception as ex:
                    results.append((file_path, None, str(ex)))

        return results

    def _run_exiftool(self, file_path: Path) -> Dict[str, Any]:
        """Run ExifTool on a single file"""
        cmd = [self.exiftool_path] + self.EXIFTOOL_ARGS + [str(file_path)]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=30
            )

            if result.returncode != 0 and "Warning" not in result.stderr:
                raise MetadataReadError(f"ExifTool error: {result.stderr}")

            data = json.loads(result.stdout)
            return data[0] if data else {}

        except subprocess.TimeoutExpired:
            raise MetadataReadError(f"ExifTool timeout for: {file_path}")
        except json.JSONDecodeError as e:
            raise MetadataReadError(f"Failed to parse ExifTool output: {e}")

    def _run_exiftool_batch(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """Run ExifTool on multiple files at once"""
        cmd = [self.exiftool_path] + self.EXIFTOOL_ARGS + [str(p) for p in file_paths]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=300  # Longer timeout for batch
            )

            data = json.loads(result.stdout)
            return data if isinstance(data, list) else [data]

        except subprocess.TimeoutExpired:
            raise MetadataReadError("ExifTool batch timeout")
        except json.JSONDecodeError as e:
            raise MetadataReadError(f"Failed to parse ExifTool batch output: {e}")

    def _parse_metadata(self, file_path: Path, raw_data: Dict[str, Any]) -> ImageMetadata:
        """Parse raw ExifTool output into ImageMetadata structure"""

        # Basic file info
        stat = file_path.stat()

        metadata = ImageMetadata(
            file_path=file_path,
            file_name=file_path.name,
            file_size=stat.st_size,
            raw_exif={},
            raw_iptc={},
            raw_xmp={}
        )

        sources = []

        # Parse EXIF
        exif_found = self._parse_exif(metadata, raw_data)
        if exif_found:
            sources.append(MetadataSource.EXIF)

        # Parse IPTC
        iptc_found = self._parse_iptc(metadata, raw_data)
        if iptc_found:
            sources.append(MetadataSource.IPTC)

        # Parse XMP
        xmp_found = self._parse_xmp(metadata, raw_data)
        if xmp_found:
            sources.append(MetadataSource.XMP)

        # Calculate megapixels
        if metadata.width and metadata.height:
            metadata.megapixels = round((metadata.width * metadata.height) / 1_000_000, 2)

        # Determine format from file extension and MIME
        mime = raw_data.get("File:MIMEType", "")
        if "jpeg" in mime.lower():
            metadata.format = "JPEG"
        elif "tiff" in mime.lower():
            metadata.format = "TIFF"
        elif "png" in mime.lower():
            metadata.format = "PNG"
        else:
            metadata.format = file_path.suffix.upper().replace(".", "")

        metadata.has_embedded_metadata = len(sources) > 0
        metadata.metadata_sources = sources

        return metadata

    def _parse_exif(self, metadata: ImageMetadata, raw_data: Dict[str, Any]) -> bool:
        """Parse EXIF data from raw ExifTool output"""
        found = False

        for tag, field in self.EXIF_MAPPING.items():
            value = raw_data.get(tag)
            if value is not None:
                found = True
                metadata.raw_exif[tag] = value

                # Special handling for certain fields
                if field == "date_taken":
                    metadata.date_taken = self._parse_datetime(value)
                elif field == "shutter_speed":
                    metadata.shutter_speed = self._format_shutter_speed(value)
                elif field == "flash_fired":
                    metadata.flash_fired = bool(value & 1) if isinstance(value, int) else False
                else:
                    setattr(metadata, field, value)

        # Try alternative tags for dimensions
        if not metadata.width:
            metadata.width = raw_data.get("File:ImageWidth", 0)
        if not metadata.height:
            metadata.height = raw_data.get("File:ImageHeight", 0)

        return found

    def _parse_iptc(self, metadata: ImageMetadata, raw_data: Dict[str, Any]) -> bool:
        """Parse IPTC data from raw ExifTool output"""
        found = False
        iptc = IPTCFields()

        for tag, field in self.IPTC_MAPPING.items():
            value = raw_data.get(tag)
            if value is not None:
                found = True
                metadata.raw_iptc[tag] = value

                # Handle list fields
                if field in ["keywords", "supplemental_categories"]:
                    if isinstance(value, str):
                        value = [v.strip() for v in value.split(",")]
                    setattr(iptc, field, value if isinstance(value, list) else [value])
                elif field == "date_created":
                    iptc.date_created = self._parse_iptc_date(value, raw_data.get("IPTC:TimeCreated"))
                else:
                    setattr(iptc, field, value)

        metadata.iptc = iptc
        return found

    def _parse_xmp(self, metadata: ImageMetadata, raw_data: Dict[str, Any]) -> bool:
        """Parse XMP data from raw ExifTool output"""
        found = False

        # XMP Rating and Label
        for tag, field in self.XMP_MAPPING.items():
            value = raw_data.get(tag)
            if value is not None:
                found = True
                metadata.raw_xmp[tag] = value

                if field == "xmp_subject":
                    if isinstance(value, str):
                        value = [v.strip() for v in value.split(",")]
                    metadata.xmp_subject = value if isinstance(value, list) else [value]
                elif hasattr(metadata, field):
                    setattr(metadata, field, value)

        return found

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse various datetime formats"""
        if isinstance(value, datetime):
            return value

        if not isinstance(value, str):
            return None

        # Common EXIF datetime formats
        formats = [
            "%Y:%m:%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y:%m:%d",
            "%Y-%m-%d",
        ]

        # Remove timezone if present
        value = re.sub(r'[+-]\d{2}:\d{2}$', '', value)

        for fmt in formats:
            try:
                return datetime.strptime(value.strip(), fmt)
            except ValueError:
                continue

        return None

    def _parse_iptc_date(self, date_str: Any, time_str: Any = None) -> Optional[datetime]:
        """Parse IPTC date and time fields"""
        if not date_str:
            return None

        date_str = str(date_str)

        # IPTC date format: YYYYMMDD
        if len(date_str) == 8 and date_str.isdigit():
            try:
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])

                hour, minute, second = 0, 0, 0
                if time_str:
                    time_str = str(time_str)
                    if len(time_str) >= 6:
                        hour = int(time_str[:2])
                        minute = int(time_str[2:4])
                        second = int(time_str[4:6])

                return datetime(year, month, day, hour, minute, second)
            except ValueError:
                pass

        return self._parse_datetime(date_str)

    def _format_shutter_speed(self, value: Any) -> str:
        """Format shutter speed as fraction string"""
        if isinstance(value, str):
            return value

        if isinstance(value, (int, float)):
            if value >= 1:
                return f"{value}s"
            else:
                return f"1/{int(1/value)}s"

        return str(value)

    def get_quick_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get basic image info without full metadata extraction
        Faster for validation/filtering purposes
        """
        cmd = [
            self.exiftool_path,
            "-json",
            "-n",
            "-ImageWidth",
            "-ImageHeight",
            "-FileSize",
            "-MIMEType",
            "-Orientation",
            str(file_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=10
            )

            data = json.loads(result.stdout)
            if data:
                return data[0]

        except Exception as e:
            logger.warning(f"Quick info extraction failed for {file_path}: {e}")

        return {}

    def has_metadata(self, file_path: Path) -> Dict[str, bool]:
        """
        Check which metadata types exist in the file

        Returns:
            Dict with keys 'exif', 'iptc', 'xmp' and boolean values
        """
        cmd = [
            self.exiftool_path,
            "-json",
            "-G1",
            "-EXIF:all",
            "-IPTC:all",
            "-XMP:all",
            str(file_path)
        ]

        result = {
            "exif": False,
            "iptc": False,
            "xmp": False
        }

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=10
            )

            data = json.loads(proc.stdout)
            if data:
                for key in data[0].keys():
                    if key.startswith("EXIF:"):
                        result["exif"] = True
                    elif key.startswith("IPTC:"):
                        result["iptc"] = True
                    elif key.startswith("XMP"):
                        result["xmp"] = True

        except Exception as e:
            logger.warning(f"Metadata check failed for {file_path}: {e}")

        return result
