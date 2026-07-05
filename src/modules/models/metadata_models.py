"""
Dataclass models for metadata validation and data structures.

Stdlib ``@dataclass`` only — Pydantic was dropped on purpose to keep
the PyInstaller bundle small (see CLAUDE.md, décisions techniques).
"""

import re
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    "Abstract",
    "Animals/Wildlife",
    "Arts",
    "Backgrounds/Textures",
    "Beauty/Fashion",
    "Buildings/Landmarks",
    "Business/Finance",
    "Celebrities",
    "Education",
    "Food and drink",
    "Healthcare/Medical",
    "Holidays",
    "Industrial",
    "Interiors",
    "Miscellaneous",
    "Nature",
    "Objects",
    "Parks/Outdoor",
    "People",
    "Religion",
    "Science",
    "Signs/Symbols",
    "Sports/Recreation",
    "Technology",
    "Transportation",
    "Vintage",
]


# Adobe Stock official categories (21).
# Source: Adobe Stock Contributor portal taxonomy. The mapping below
# is intentionally one-way + best-effort — Adobe expects a single
# integer category id on import, but we ship the label so a human can
# pick the integer from the CSV preview in the Contributor portal.
ADOBE_STOCK_CATEGORIES = [
    "Animals",
    "Buildings and Architecture",
    "Business",
    "Drinks",
    "The Environment",
    "States of Mind",
    "Food",
    "Graphic Resources",
    "Hobbies and Leisure",
    "Industry",
    "Landscapes",
    "Lifestyle",
    "People",
    "Plants and Flowers",
    "Culture and Religion",
    "Science",
    "Social Issues",
    "Sports",
    "Technology",
    "Transport",
    "Travel",
]


# Best-effort mapping Shutterstock → Adobe. Used as a fallback when
# the analyzer only produced a Shutterstock category (e.g. legacy
# pipeline or non-AI mode) and we still want to fill the Adobe CSV
# column. Missing keys default to "Lifestyle" downstream.
SHUTTERSTOCK_TO_ADOBE_CATEGORY: Dict[str, str] = {
    "Abstract": "Graphic Resources",
    "Animals/Wildlife": "Animals",
    "Arts": "Graphic Resources",
    "Backgrounds/Textures": "Graphic Resources",
    "Beauty/Fashion": "Lifestyle",
    "Buildings/Landmarks": "Buildings and Architecture",
    "Business/Finance": "Business",
    "Celebrities": "People",
    "Education": "Lifestyle",
    "Food and drink": "Food",
    "Healthcare/Medical": "Science",
    "Holidays": "Culture and Religion",
    "Industrial": "Industry",
    "Interiors": "Buildings and Architecture",
    "Miscellaneous": "Lifestyle",
    "Nature": "Landscapes",
    "Objects": "Hobbies and Leisure",
    "Parks/Outdoor": "Landscapes",
    "People": "People",
    "Religion": "Culture and Religion",
    "Science": "Science",
    "Signs/Symbols": "Graphic Resources",
    "Sports/Recreation": "Sports",
    "Technology": "Technology",
    "Transportation": "Transport",
    "Vintage": "Graphic Resources",
}


def map_shutterstock_to_adobe(category: str) -> str:
    """Translate a Shutterstock category to its closest Adobe equivalent."""
    return SHUTTERSTOCK_TO_ADOBE_CATEGORY.get(category, "Lifestyle")


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
            "metadata_sources": [s.value for s in self.metadata_sources],
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
            kw = re.sub(r"[^\w\s-]", "", kw)

            # Skip if too short or duplicate
            if len(kw) >= 2 and kw not in seen:
                cleaned.append(kw)
                seen.add(kw)

        self.keywords = cleaned[:50]  # Max 50
        return self.keywords

    def to_csv_row(self) -> Dict[str, str]:
        """Convert to Shutterstock CSV format.

        Shutterstock's contributor CSV template expects keywords
        separated by commas, not spaces — the historical " ".join
        produced a single mega-keyword on import.
        """
        from ..analysis.limits import (
            SHUTTERSTOCK_DESCRIPTION_MAX,
            SHUTTERSTOCK_KEYWORDS_MAX,
            clamp_keywords,
            smart_truncate,
        )

        return {
            "Filename": self.filename,
            "Description": smart_truncate(self.description, SHUTTERSTOCK_DESCRIPTION_MAX),
            "Keywords": ", ".join(clamp_keywords(self.keywords, SHUTTERSTOCK_KEYWORDS_MAX)),
            "Categories": ", ".join(self.categories),
            "Editorial": "Yes" if self.editorial else "No",
            "Mature": "Yes" if self.mature_content else "No",
            "Illustration": "Yes" if self.illustration else "No",
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

    # Platform readiness flags. Soft signals — they do NOT mark the
    # file as invalid; downstream UI shows them as informational
    # badges. Real rejection happens at Adobe/Shutterstock review.
    adobe_ready: bool = True
    shutterstock_ready: bool = True

    @property
    def overall_score(self) -> float:
        """Calculate overall validation score"""
        return (self.completeness_score + self.quality_score + self.seo_score) / 3


# ============================================================================
# Expert microstock report — multi-platform, AI-optional
# ============================================================================


@dataclass
class TechnicalFlags:
    """Visual/technical defect flags for an image.

    Each flag is a boolean — True means the defect is suspected. In
    non-AI mode (the default on low-power machines) every flag stays
    False; only the size/format/MP checks populate the report.
    The flags are intentionally generous: we don't want to reject
    borderline images, Adobe/Shutterstock reviewers will.
    """

    noise: bool = False
    soft_focus: bool = False
    jpeg_artifacts: bool = False
    oversharpen: bool = False
    hdr_overprocessed: bool = False
    halos: bool = False
    oversaturated: bool = False
    ai_artifacts: bool = False
    bad_hands: bool = False
    unreadable_text: bool = False
    watermark: bool = False
    logo_or_brand: bool = False
    protected_building: bool = False
    needs_model_release: bool = False
    needs_property_release: bool = False

    def any_active(self) -> bool:
        """True if at least one defect flag is set."""
        return any(getattr(self, f.name) for f in dataclass_fields(self))

    def active_labels(self) -> List[str]:
        """Human-readable list of active flags (French labels)."""
        labels = {
            "noise": "bruit numérique",
            "soft_focus": "focus mou",
            "jpeg_artifacts": "artefacts JPEG",
            "oversharpen": "sur-accentuation",
            "hdr_overprocessed": "HDR excessif",
            "halos": "halos",
            "oversaturated": "saturation excessive",
            "ai_artifacts": "défauts IA",
            "bad_hands": "doigts incorrects",
            "unreadable_text": "texte illisible",
            "watermark": "filigrane",
            "logo_or_brand": "logo / marque",
            "protected_building": "bâtiment protégé",
            "needs_model_release": "model release requis",
            "needs_property_release": "property release requis",
        }
        return [labels[f.name] for f in dataclass_fields(self) if getattr(self, f.name)]

    def to_dict(self) -> Dict[str, bool]:
        return {f.name: getattr(self, f.name) for f in dataclass_fields(self)}


@dataclass
class RejectionRisk:
    """One concrete risk of rejection on a stock platform."""

    issue: str  # short label, e.g. "Résolution < 4 MP"
    cause: str  # why this triggers a rejection
    fix: str  # how to fix it
    severity: str = "warning"  # "info" | "warning" | "blocker"

    def to_dict(self) -> Dict[str, str]:
        return {"issue": self.issue, "cause": self.cause, "fix": self.fix, "severity": self.severity}


@dataclass
class ExpertScores:
    """Four 0-10 scores for the expert dashboard."""

    commercial: int = 0  # commercial potential
    technical: int = 0  # technical quality
    seo: int = 0  # SEO / discoverability
    rejection_risk: int = 0  # 0 = no risk, 10 = certain rejection

    def to_dict(self) -> Dict[str, int]:
        return {
            "commercial": self.commercial,
            "technical": self.technical,
            "seo": self.seo,
            "rejection_risk": self.rejection_risk,
        }


@dataclass
class ExpertMetadataReport:
    """Multi-platform metadata report — AI-optional.

    Produced either by the heuristic builder (no AI required, runs
    instantly on any PC) or by the AI-augmented builder when Ollama
    is available. Both paths populate the same dataclass so the UI
    and the CSV exporter don't need to know which mode ran.

    Design notes:
    - All fields are lax by default. An empty `rejection_risks` list
      doesn't mean "perfect", it means "we didn't detect anything
      blocking on our side, let the reviewer decide".
    - `keywords` is already ordered from most to least commercial —
      the first 10 are the ones Adobe/Shutterstock weight the most.
    - `title_adobe` and `title_shutterstock` can be identical when
      no AI rewrite is available; the heuristic builder does that.
    """

    # Source
    file_path: Path
    source: str = "heuristic"  # "heuristic" | "ai" | "hybrid"

    # Scores
    scores: ExpertScores = field(default_factory=ExpertScores)

    # Texts
    title_adobe: str = ""
    title_shutterstock: str = ""
    description: str = ""

    # Keywords — already ordered, max 50 (Adobe + Shutterstock limit).
    keywords: List[str] = field(default_factory=list)

    # Categories — Adobe takes a primary + optional secondary,
    # Shutterstock picks one (max two).
    category_adobe_primary: str = ""
    category_adobe_secondary: str = ""
    categories_shutterstock: List[str] = field(default_factory=list)

    # Risk & improvement insights
    rejection_risks: List[RejectionRisk] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)

    # Marketing — suggested usages + buyer profiles + visual trends.
    marketing_uses: List[str] = field(default_factory=list)
    buyer_profiles: List[str] = field(default_factory=list)
    trends: List[str] = field(default_factory=list)

    # Visual defects — all False in heuristic mode (no AI vision).
    technical_flags: TechnicalFlags = field(default_factory=TechnicalFlags)

    # Editorial / illustration markers (carried over for CSV export).
    editorial: bool = False
    illustration: bool = False
    mature_content: bool = False

    # Platform compliance summary — populated by the
    # platform_compliance helper. Empty list = no warning.
    adobe_warnings: List[str] = field(default_factory=list)
    shutterstock_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": str(self.file_path),
            "source": self.source,
            "scores": self.scores.to_dict(),
            "title_adobe": self.title_adobe,
            "title_shutterstock": self.title_shutterstock,
            "description": self.description,
            "keywords": self.keywords,
            "category_adobe_primary": self.category_adobe_primary,
            "category_adobe_secondary": self.category_adobe_secondary,
            "categories_shutterstock": self.categories_shutterstock,
            "rejection_risks": [r.to_dict() for r in self.rejection_risks],
            "improvements": self.improvements,
            "marketing_uses": self.marketing_uses,
            "buyer_profiles": self.buyer_profiles,
            "trends": self.trends,
            "technical_flags": self.technical_flags.to_dict(),
            "editorial": self.editorial,
            "illustration": self.illustration,
            "mature_content": self.mature_content,
            "adobe_warnings": self.adobe_warnings,
            "shutterstock_warnings": self.shutterstock_warnings,
        }

    def to_adobe_csv_row(self) -> Dict[str, str]:
        """One row for the Adobe Stock contributor CSV (5 columns).

        Hard portal caps applied here (title 200 chars, 49 keywords)
        with word-boundary truncation — never a chopped word, never an
        ellipsis in the portal review queue.
        """
        from ..analysis.limits import ADOBE_KEYWORDS_MAX, ADOBE_TITLE_MAX, clamp_keywords, smart_truncate

        filename = Path(self.file_path).name
        return {
            "Filename": filename,
            "Title": smart_truncate(self.title_adobe or self.title_shutterstock, ADOBE_TITLE_MAX),
            "Keywords": ", ".join(clamp_keywords(self.keywords, ADOBE_KEYWORDS_MAX)),
            "Category": self.category_adobe_primary,
            "Releases": "",  # filled by user if model/property release exists
        }

    def to_shutterstock_csv_row(self) -> Dict[str, str]:
        """One row for the Shutterstock contributor CSV (7 columns).

        Hard portal caps applied here (description 200 chars, 50
        keywords) with word-boundary truncation.
        """
        from ..analysis.limits import (
            SHUTTERSTOCK_DESCRIPTION_MAX,
            SHUTTERSTOCK_KEYWORDS_MAX,
            clamp_keywords,
            smart_truncate,
        )

        filename = Path(self.file_path).name
        # Shutterstock uses Description as the primary text field —
        # we fall back to title if no description was generated.
        description = self.description or self.title_shutterstock
        return {
            "Filename": filename,
            "Description": smart_truncate(description, SHUTTERSTOCK_DESCRIPTION_MAX),
            "Keywords": ", ".join(clamp_keywords(self.keywords, SHUTTERSTOCK_KEYWORDS_MAX)),
            "Categories": ", ".join(self.categories_shutterstock),
            "Editorial": "Yes" if self.editorial else "No",
            "Mature": "Yes" if self.mature_content else "No",
            "Illustration": "Yes" if self.illustration else "No",
        }
