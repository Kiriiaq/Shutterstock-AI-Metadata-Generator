"""
Pydantic models for metadata validation and data structures
"""

from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pathlib import Path
import re


class ContentType(Enum):
    """Image content type classification"""
    PHOTO = "photo"
    ILLUSTRATION = "illustration"
    VECTOR = "vector"


class ProcessingStatus(Enum):
    """Status of image processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class MetadataSource(Enum):
    """Source of metadata"""
    EXIF = "exif"
    IPTC = "iptc"
    XMP = "xmp"
    AI_GENERATED = "ai_generated"
    USER_INPUT = "user_input"
    FILENAME = "filename"


# Shutterstock official categories
SHUTTERSTOCK_CATEGORIES = [
    "Abstract", "Animals/Wildlife", "Arts", "Backgrounds/Textures",
    "Beauty/Fashion", "Buildings/Landmarks", "Business/Finance",
    "Celebrities", "Education", "Food and drink", "Healthcare/Medical",
    "Holidays", "Industrial", "Interiors", "Miscellaneous", "Nature",
    "Objects", "Parks/Outdoor", "People", "Religion", "Science",
    "Signs/Symbols", "Sports/Recreation", "Technology", "Transportation",
    "Vintage"
]


@dataclass
class IPTCFields:
    """
    Complete IPTC metadata fields mapping
    Based on IPTC Photo Metadata Standard 2021.1
    """
    # Core descriptive fields
    object_name: Optional[str] = None  # 2:05 - Title
    headline: Optional[str] = None  # 2:105
    caption: Optional[str] = None  # 2:120 - Description
    keywords: List[str] = field(default_factory=list)  # 2:25

    # Creator/Copyright
    byline: Optional[str] = None  # 2:80 - Creator/Author
    byline_title: Optional[str] = None  # 2:85 - Creator's Job Title
    credit: Optional[str] = None  # 2:110 - Credit Line
    source: Optional[str] = None  # 2:115
    copyright_notice: Optional[str] = None  # 2:116

    # Location
    city: Optional[str] = None  # 2:90
    sublocation: Optional[str] = None  # 2:92
    province_state: Optional[str] = None  # 2:95
    country_code: Optional[str] = None  # 2:100 - ISO 3166-1 alpha-3
    country_name: Optional[str] = None  # 2:101

    # Editorial
    category: Optional[str] = None  # 2:15 (deprecated but still used)
    supplemental_categories: List[str] = field(default_factory=list)  # 2:20
    urgency: Optional[int] = None  # 2:10 (1-8, 1=most urgent)

    # Dates
    date_created: Optional[datetime] = None  # 2:55
    time_created: Optional[str] = None  # 2:60

    # Administrative
    special_instructions: Optional[str] = None  # 2:40
    transmission_reference: Optional[str] = None  # 2:103 - Job ID

    # Contact info
    contact_city: Optional[str] = None
    contact_country: Optional[str] = None
    contact_address: Optional[str] = None
    contact_postal_code: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_website: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, list) and len(value) == 0:
                    continue
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                else:
                    result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IPTCFields":
        """Create from dictionary.

        Uses dataclasses.fields(cls) instead of hasattr(cls, key) so that
        list-typed fields declared via field(default_factory=list) (keywords,
        supplemental_categories) are accepted — hasattr returns False for
        those at the class level.
        """
        valid_names = {f.name for f in dataclass_fields(cls)}
        init_kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key not in valid_names:
                continue
            if key == "date_created" and isinstance(value, str):
                try:
                    init_kwargs[key] = datetime.fromisoformat(value)
                except ValueError:
                    continue
            else:
                init_kwargs[key] = value
        return cls(**init_kwargs)


@dataclass
class ImageMetadata:
    """
    Complete metadata extracted from an image file
    Combines EXIF, IPTC, and XMP data
    """
    file_path: Path
    file_name: str
    file_size: int  # bytes

    # Image properties
    width: int = 0
    height: int = 0
    megapixels: float = 0.0
    format: str = ""
    color_space: str = ""
    bit_depth: int = 0

    # EXIF data
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
    focal_length: Optional[float] = None
    aperture: Optional[float] = None
    shutter_speed: Optional[str] = None
    iso: Optional[int] = None
    flash_fired: Optional[bool] = None
    date_taken: Optional[datetime] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_altitude: Optional[float] = None
    orientation: Optional[int] = None

    # IPTC data
    iptc: IPTCFields = field(default_factory=IPTCFields)

    # XMP specific
    xmp_rating: Optional[int] = None  # 0-5 stars
    xmp_label: Optional[str] = None  # Color label
    xmp_subject: List[str] = field(default_factory=list)

    # Processing metadata
    has_embedded_metadata: bool = False
    metadata_sources: List[MetadataSource] = field(default_factory=list)
    raw_exif: Dict[str, Any] = field(default_factory=dict)
    raw_iptc: Dict[str, Any] = field(default_factory=dict)
    raw_xmp: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_gps(self) -> bool:
        """Check if GPS data is available"""
        return self.gps_latitude is not None and self.gps_longitude is not None

    @property
    def has_iptc(self) -> bool:
        """Check if IPTC metadata exists"""
        return self.iptc.caption is not None or len(self.iptc.keywords) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary"""
        result = {
            "file_path": str(self.file_path),
            "file_name": self.file_name,
            "file_size": self.file_size,
            "width": self.width,
            "height": self.height,
            "megapixels": self.megapixels,
            "format": self.format,
            "color_space": self.color_space,
            "bit_depth": self.bit_depth,
            "camera_make": self.camera_make,
            "camera_model": self.camera_model,
            "lens_model": self.lens_model,
            "focal_length": self.focal_length,
            "aperture": self.aperture,
            "shutter_speed": self.shutter_speed,
            "iso": self.iso,
            "flash_fired": self.flash_fired,
            "date_taken": self.date_taken.isoformat() if self.date_taken else None,
            "gps_latitude": self.gps_latitude,
            "gps_longitude": self.gps_longitude,
            "gps_altitude": self.gps_altitude,
            "orientation": self.orientation,
            "iptc": self.iptc.to_dict(),
            "xmp_rating": self.xmp_rating,
            "xmp_label": self.xmp_label,
            "xmp_subject": self.xmp_subject,
            "has_embedded_metadata": self.has_embedded_metadata,
            "metadata_sources": [s.value for s in self.metadata_sources]
        }
        return result


@dataclass
class ShutterstockMetadata:
    """
    Shutterstock-specific metadata for submission
    Validated according to Shutterstock guidelines
    """
    filename: str
    title: str  # max 200 chars
    description: str  # max 200 chars
    keywords: List[str]  # 7-50 keywords
    categories: List[str]  # 1-2 categories

    # Flags
    editorial: bool = False
    mature_content: bool = False
    illustration: bool = False

    # Optional
    location: Optional[str] = None

    # Validation status
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate on creation"""
        self.validate()

    def validate(self) -> bool:
        """Validate metadata according to Shutterstock guidelines"""
        self.validation_errors = []

        # Title validation
        if not self.title or len(self.title.strip()) == 0:
            self.validation_errors.append("Title is required")
        elif len(self.title) > 200:
            self.validation_errors.append(f"Title exceeds 200 characters ({len(self.title)})")

        # Description validation
        if not self.description or len(self.description.strip()) == 0:
            self.validation_errors.append("Description is required")
        elif len(self.description) > 200:
            self.validation_errors.append(f"Description exceeds 200 characters ({len(self.description)})")

        # Keywords validation
        if len(self.keywords) < 7:
            self.validation_errors.append(f"Minimum 7 keywords required ({len(self.keywords)} provided)")
        elif len(self.keywords) > 50:
            self.validation_errors.append(f"Maximum 50 keywords allowed ({len(self.keywords)} provided)")

        # Check keyword quality
        for kw in self.keywords:
            if len(kw) < 2:
                self.validation_errors.append(f"Keyword too short: '{kw}'")

        # Categories validation
        if len(self.categories) < 1:
            self.validation_errors.append("At least 1 category required")
        elif len(self.categories) > 2:
            self.validation_errors.append("Maximum 2 categories allowed")

        # Validate category names
        for cat in self.categories:
            if cat not in SHUTTERSTOCK_CATEGORIES:
                self.validation_errors.append(f"Invalid category: '{cat}'")

        self.is_valid = len(self.validation_errors) == 0
        return self.is_valid

    def clean_keywords(self) -> List[str]:
        """Clean and normalize keywords"""
        cleaned = []
        seen = set()

        for kw in self.keywords:
            # Lowercase
            kw = kw.lower().strip()

            # Remove accents and special characters
            kw = re.sub(r'[^\w\s-]', '', kw)

            # Skip if too short or duplicate
            if len(kw) >= 2 and kw not in seen:
                cleaned.append(kw)
                seen.add(kw)

        self.keywords = cleaned[:50]  # Max 50
        return self.keywords

    def to_csv_row(self) -> Dict[str, str]:
        """Convert to Shutterstock CSV format"""
        return {
            "Filename": self.filename,
            "Description": self.description,
            "Keywords": " ".join(self.keywords),
            "Categories": ",".join(self.categories),
            "Editorial": "Yes" if self.editorial else "No",
            "Mature": "Yes" if self.mature_content else "No",
            "Illustration": "Yes" if self.illustration else "No"
        }


@dataclass
class ProcessingJob:
    """
    A processing job for the worker pool
    """
    job_id: str
    file_path: Path
    operations: List[str]  # e.g., ["read_metadata", "ai_analyze", "write_metadata"]
    priority: int = 5  # 1-10, 1 = highest

    # Options
    options: Dict[str, Any] = field(default_factory=dict)

    # State
    status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    result: Optional[Any] = None
    error: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        """Get processing duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class ProcessingResult:
    """
    Result of a processing operation
    """
    job_id: str
    success: bool
    file_path: Path

    # Metadata results
    metadata_read: Optional[ImageMetadata] = None
    metadata_written: bool = False
    shutterstock_metadata: Optional[ShutterstockMetadata] = None

    # AI analysis results
    ai_title: Optional[str] = None
    ai_description: Optional[str] = None
    ai_keywords: List[str] = field(default_factory=list)
    ai_categories: List[str] = field(default_factory=list)

    # Errors and warnings
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # Timing
    processing_time: float = 0.0


@dataclass
class ValidationResult:
    """
    Result of metadata validation
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    # Scores
    completeness_score: float = 0.0  # 0-100
    quality_score: float = 0.0  # 0-100
    seo_score: float = 0.0  # 0-100

    @property
    def overall_score(self) -> float:
        """Calculate overall validation score"""
        return (self.completeness_score + self.quality_score + self.seo_score) / 3
