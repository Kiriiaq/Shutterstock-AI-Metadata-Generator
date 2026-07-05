"""
IPTCEngine - Complete IPTC metadata management
Handles editorial workflows, templates, and validation
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..models.metadata_models import SHUTTERSTOCK_CATEGORIES, IPTCFields, ShutterstockMetadata

logger = logging.getLogger(__name__)


# ISO 3166-1 alpha-3 country codes (subset of commonly used)
COUNTRY_CODES = {
    "USA": "United States",
    "GBR": "United Kingdom",
    "FRA": "France",
    "DEU": "Germany",
    "ESP": "Spain",
    "ITA": "Italy",
    "CAN": "Canada",
    "AUS": "Australia",
    "JPN": "Japan",
    "CHN": "China",
    "BRA": "Brazil",
    "MEX": "Mexico",
    "IND": "India",
    "RUS": "Russia",
    "NLD": "Netherlands",
    "BEL": "Belgium",
    "CHE": "Switzerland",
    "AUT": "Austria",
    "PRT": "Portugal",
    "POL": "Poland",
    "SWE": "Sweden",
    "NOR": "Norway",
    "DNK": "Denmark",
    "FIN": "Finland",
    "GRC": "Greece",
    "TUR": "Turkey",
    "ZAF": "South Africa",
    "EGY": "Egypt",
    "MAR": "Morocco",
    "ARE": "United Arab Emirates",
    "SGP": "Singapore",
    "THA": "Thailand",
    "VNM": "Vietnam",
    "KOR": "South Korea",
    "NZL": "New Zealand",
    "ARG": "Argentina",
    "CHL": "Chile",
    "COL": "Colombia",
    "PER": "Peru",
}


@dataclass
class IPTCTemplate:
    """
    Reusable IPTC metadata template
    """

    name: str
    description: str = ""

    # Default values
    byline: Optional[str] = None
    byline_title: Optional[str] = None
    credit: Optional[str] = None
    source: Optional[str] = None
    copyright_notice: Optional[str] = None

    # Contact info
    contact_city: Optional[str] = None
    contact_country: Optional[str] = None
    contact_email: Optional[str] = None
    contact_website: Optional[str] = None

    # Default categories
    default_categories: List[str] = field(default_factory=list)

    # Keywords to always include
    base_keywords: List[str] = field(default_factory=list)

    # Editorial settings
    is_editorial: bool = False
    editorial_instructions: Optional[str] = None

    def apply_to(self, iptc: IPTCFields) -> IPTCFields:
        """Apply template defaults to an IPTCFields object"""
        if self.byline and not iptc.byline:
            iptc.byline = self.byline
        if self.byline_title and not iptc.byline_title:
            iptc.byline_title = self.byline_title
        if self.credit and not iptc.credit:
            iptc.credit = self.credit
        if self.source and not iptc.source:
            iptc.source = self.source
        if self.copyright_notice and not iptc.copyright_notice:
            iptc.copyright_notice = self.copyright_notice

        # Add base keywords
        if self.base_keywords:
            existing = set(iptc.keywords)
            for kw in self.base_keywords:
                if kw not in existing:
                    iptc.keywords.append(kw)

        # Add default categories
        if self.default_categories and not iptc.supplemental_categories:
            iptc.supplemental_categories = self.default_categories.copy()

        # Editorial instructions
        if self.is_editorial and not iptc.special_instructions:
            iptc.special_instructions = self.editorial_instructions or "EDITORIAL USE ONLY"

        return iptc

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "byline": self.byline,
            "byline_title": self.byline_title,
            "credit": self.credit,
            "source": self.source,
            "copyright_notice": self.copyright_notice,
            "contact_city": self.contact_city,
            "contact_country": self.contact_country,
            "contact_email": self.contact_email,
            "contact_website": self.contact_website,
            "default_categories": self.default_categories,
            "base_keywords": self.base_keywords,
            "is_editorial": self.is_editorial,
            "editorial_instructions": self.editorial_instructions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IPTCTemplate":
        """Deserialize from dictionary"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class IPTCEngine:
    """
    Complete IPTC metadata engine for editorial workflows
    """

    def __init__(self):
        """Initialize IPTC Engine"""
        self.templates: Dict[str, IPTCTemplate] = {}
        self._load_default_templates()

    def _load_default_templates(self):
        """Load built-in templates"""
        # Stock photography template
        self.templates["stock_photo"] = IPTCTemplate(
            name="Stock Photography",
            description="Standard template for stock photo submissions",
            is_editorial=False,
        )

        # Editorial template
        self.templates["editorial"] = IPTCTemplate(
            name="Editorial Content",
            description="Template for editorial/news content",
            is_editorial=True,
            editorial_instructions="EDITORIAL USE ONLY - Not for commercial use",
        )

        # Nature/Wildlife template
        self.templates["nature"] = IPTCTemplate(
            name="Nature & Wildlife",
            description="Template for nature and wildlife photography",
            default_categories=["Nature", "Animals/Wildlife"],
            base_keywords=["nature", "wildlife", "outdoor", "natural"],
        )

        # Travel template
        self.templates["travel"] = IPTCTemplate(
            name="Travel Photography",
            description="Template for travel and landmark photos",
            default_categories=["Buildings/Landmarks", "Parks/Outdoor"],
            base_keywords=["travel", "tourism", "destination", "landmark"],
        )

    def create_iptc_from_shutterstock(self, ss_metadata: ShutterstockMetadata) -> IPTCFields:
        """
        Convert Shutterstock metadata to full IPTC fields

        Args:
            ss_metadata: ShutterstockMetadata object

        Returns:
            IPTCFields object
        """
        iptc = IPTCFields(
            object_name=ss_metadata.title[:64] if ss_metadata.title else None,
            headline=ss_metadata.title[:256] if ss_metadata.title else None,
            caption=ss_metadata.description,
            keywords=ss_metadata.keywords.copy(),
            supplemental_categories=ss_metadata.categories.copy(),
        )

        # Category (first 3 chars of first category)
        if ss_metadata.categories:
            iptc.category = ss_metadata.categories[0][:3].upper()

        # Editorial flag
        if ss_metadata.editorial:
            iptc.special_instructions = "EDITORIAL USE ONLY"

        return iptc

    def create_shutterstock_from_iptc(self, iptc: IPTCFields, filename: str) -> ShutterstockMetadata:
        """
        Convert IPTC fields to Shutterstock metadata

        Args:
            iptc: IPTCFields object
            filename: Image filename

        Returns:
            ShutterstockMetadata object
        """
        from ..analysis.limits import SHUTTERSTOCK_DESCRIPTION_MAX, smart_truncate

        # Title: prefer headline, fall back to object_name. Truncate at
        # a word boundary — a clean shorter sentence, never "…".
        title = smart_truncate(iptc.headline or iptc.object_name or "", SHUTTERSTOCK_DESCRIPTION_MAX)

        # Description: use caption
        description = smart_truncate(iptc.caption or title, SHUTTERSTOCK_DESCRIPTION_MAX)

        # Keywords
        keywords = iptc.keywords.copy() if iptc.keywords else []

        # Categories: map supplemental_categories to Shutterstock categories
        categories = []
        for cat in iptc.supplemental_categories or []:
            matched = self._match_shutterstock_category(cat)
            if matched and matched not in categories:
                categories.append(matched)

        # Ensure at least one category
        if not categories:
            categories = ["Miscellaneous"]

        # Editorial flag
        editorial = False
        if iptc.special_instructions:
            editorial = "editorial" in iptc.special_instructions.lower()

        return ShutterstockMetadata(
            filename=filename,
            title=title,
            description=description,
            keywords=keywords,
            categories=categories[:2],  # Max 2
            editorial=editorial,
        )

    def _match_shutterstock_category(self, category: str) -> Optional[str]:
        """Match a category string to Shutterstock official categories"""
        category_lower = category.lower()

        # Direct match
        for ss_cat in SHUTTERSTOCK_CATEGORIES:
            if ss_cat.lower() == category_lower:
                return ss_cat

        # Partial match
        for ss_cat in SHUTTERSTOCK_CATEGORIES:
            if category_lower in ss_cat.lower() or ss_cat.lower() in category_lower:
                return ss_cat

        # Keyword-based matching
        category_mappings = {
            "animal": "Animals/Wildlife",
            "wildlife": "Animals/Wildlife",
            "pet": "Animals/Wildlife",
            "nature": "Nature",
            "landscape": "Nature",
            "forest": "Nature",
            "mountain": "Nature",
            "beach": "Nature",
            "ocean": "Nature",
            "people": "People",
            "person": "People",
            "portrait": "People",
            "family": "People",
            "business": "Business/Finance",
            "office": "Business/Finance",
            "corporate": "Business/Finance",
            "food": "Food and drink",
            "drink": "Food and drink",
            "restaurant": "Food and drink",
            "travel": "Transportation",
            "car": "Transportation",
            "plane": "Transportation",
            "building": "Buildings/Landmarks",
            "architecture": "Buildings/Landmarks",
            "city": "Buildings/Landmarks",
            "sport": "Sports/Recreation",
            "fitness": "Sports/Recreation",
            "technology": "Technology",
            "computer": "Technology",
            "digital": "Technology",
            "medical": "Healthcare/Medical",
            "health": "Healthcare/Medical",
            "doctor": "Healthcare/Medical",
            "education": "Education",
            "school": "Education",
            "abstract": "Abstract",
            "pattern": "Backgrounds/Textures",
            "texture": "Backgrounds/Textures",
            "background": "Backgrounds/Textures",
        }

        for keyword, ss_cat in category_mappings.items():
            if keyword in category_lower:
                return ss_cat

        return None

    def validate_iptc(self, iptc: IPTCFields) -> Tuple[bool, List[str], List[str]]:
        """
        Validate IPTC fields

        Args:
            iptc: IPTCFields object

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # Required fields check
        if not iptc.caption and not iptc.headline:
            errors.append("Caption or headline is required")

        # Keywords check
        if not iptc.keywords or len(iptc.keywords) < 5:
            warnings.append(f"Low keyword count ({len(iptc.keywords or [])}). Recommend at least 7.")

        # Country code validation
        if iptc.country_code:
            if len(iptc.country_code) != 3:
                errors.append("Country code must be 3 characters (ISO 3166-1 alpha-3)")
            elif iptc.country_code.upper() not in COUNTRY_CODES:
                warnings.append(f"Unrecognized country code: {iptc.country_code}")

        # Copyright notice
        if not iptc.copyright_notice:
            warnings.append("Missing copyright notice")

        # Caption length
        if iptc.caption and len(iptc.caption) > 2000:
            errors.append(f"Caption too long ({len(iptc.caption)} chars). Max 2000.")

        # Headline length
        if iptc.headline and len(iptc.headline) > 256:
            errors.append(f"Headline too long ({len(iptc.headline)} chars). Max 256.")

        # Object name length
        if iptc.object_name and len(iptc.object_name) > 64:
            errors.append(f"Object name too long ({len(iptc.object_name)} chars). Max 64.")

        # Urgency range
        if iptc.urgency is not None and (iptc.urgency < 1 or iptc.urgency > 8):
            errors.append(f"Urgency must be 1-8, got {iptc.urgency}")

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def clean_keywords(self, keywords: List[str]) -> List[str]:
        """
        Clean and normalize keywords

        Args:
            keywords: List of keywords

        Returns:
            Cleaned keyword list
        """
        cleaned = []
        seen = set()

        for kw in keywords:
            # Lowercase and strip
            kw = kw.lower().strip()

            # Remove special characters except hyphen
            kw = re.sub(r"[^\w\s-]", "", kw)

            # Normalize whitespace
            kw = re.sub(r"\s+", " ", kw)

            # Skip empty, too short, or duplicates
            if len(kw) < 2:
                continue
            if kw in seen:
                continue

            cleaned.append(kw)
            seen.add(kw)

        return cleaned

    def add_template(self, template: IPTCTemplate):
        """Add a custom template"""
        self.templates[template.name.lower().replace(" ", "_")] = template

    def get_template(self, name: str) -> Optional[IPTCTemplate]:
        """Get a template by name"""
        return self.templates.get(name.lower().replace(" ", "_"))

    def list_templates(self) -> List[str]:
        """List all available template names"""
        return list(self.templates.keys())

