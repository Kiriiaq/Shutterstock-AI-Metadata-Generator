"""
Prompt Templates - Optimized prompts for metadata generation
Supports multiple platforms (Shutterstock, Adobe Stock, generic)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class PromptType(Enum):
    """Type of metadata generation"""

    FULL = "full"  # All fields
    TITLE_ONLY = "title"
    DESCRIPTION_ONLY = "description"
    KEYWORDS_ONLY = "keywords"
    CATEGORIES_ONLY = "categories"


class Platform(Enum):
    """Target platform for metadata"""

    GENERIC = "generic"
    SHUTTERSTOCK = "shutterstock"
    ADOBE_STOCK = "adobe_stock"
    GETTY = "getty"
    ISTOCK = "istock"


@dataclass
class PlatformLimits:
    """Character/count limits for each platform"""

    title_max: int = 200
    description_max: int = 200
    keywords_min: int = 7
    keywords_max: int = 50
    keyword_max_chars: int = 50


# Platform-specific limits (hard caps mirrored from
# src.modules.analysis.limits — Adobe caps keywords at 49, not 50)
PLATFORM_LIMITS = {
    Platform.GENERIC: PlatformLimits(
        title_max=200, description_max=2000, keywords_min=5, keywords_max=50, keyword_max_chars=50
    ),
    Platform.SHUTTERSTOCK: PlatformLimits(
        title_max=200, description_max=200, keywords_min=7, keywords_max=50, keyword_max_chars=50
    ),
    Platform.ADOBE_STOCK: PlatformLimits(
        title_max=200, description_max=200, keywords_min=5, keywords_max=49, keyword_max_chars=50
    ),
    Platform.GETTY: PlatformLimits(
        title_max=250, description_max=2000, keywords_min=5, keywords_max=75, keyword_max_chars=50
    ),
    Platform.ISTOCK: PlatformLimits(
        title_max=200, description_max=200, keywords_min=7, keywords_max=50, keyword_max_chars=50
    ),
}


class PromptTemplates:
    """
    Template manager for AI prompts
    Generates optimized prompts for metadata extraction
    """

    # Base system prompt for all requests — tightly bounded so every
    # image gets exactly ONE quality title and ONE quality description
    # that fit the portal fields without any downstream truncation.
    SYSTEM_PROMPT = """You are a senior stock photography metadata editor for Adobe Stock and Shutterstock.
You write exactly ONE title, ONE description and ONE keyword list per image — final, portal-ready quality.

HARD RULES:
1. Describe only what is VISIBLE. Never guess location, brand, event or person identity.
2. No brand names, trademarks, logos or celebrity names — ever.
3. Never start with "Photo of", "Image of", "A picture of", "Stock photo".
4. No filler adjectives (beautiful, stunning, amazing, nice) — use precise, concrete words.
5. No meta-words as keywords: photo, image, picture, stock, wallpaper, jpg, closeup (unless the framing IS a close-up).
6. English only. Sentence case for title and description.
7. Respect every character budget strictly — text longer than the budget is discarded.
"""

    # Full analysis prompt template
    FULL_ANALYSIS_TEMPLATE = """Analyze this image and write its final stock metadata.

TITLE — one line, {title_target_min}-{title_target_max} characters (hard max {title_max}):
- A specific, searchable summary: main subject + action/state + setting.
- Good: "Two young architects reviewing blueprints in a sunlit open-plan office"
- Bad: "Beautiful photo of people working" (vague, filler, no setting)

DESCRIPTION — exactly one sentence, {desc_target_min}-{desc_target_max} characters (hard max {desc_max}):
- Expands the title with 2-3 concrete visual details (colors, mood, composition).
- Must stand alone as the Shutterstock search text. No second sentence.

KEYWORDS — {kw_target_min} to {kw_target_max} terms, comma-separated:
- Order matters: the FIRST 10 must be the strongest commercial search terms.
- Single words or short 2-word phrases, all lowercase.
- Cover: subject, action, setting, concept, mood, color, composition, demographics (only if visible).
- No duplicates, no plurals of a word already listed, no brands, no meta-words.

CATEGORIES — 1 or 2 from exactly this list: [Animals, Arts, Backgrounds/Textures, Beauty/Fashion, Buildings/Landmarks, Business/Finance, Celebrities, Editorial, Education, Food/Drink, Healthcare/Medical, Holidays, Industrial, Interiors, Miscellaneous, Nature, Objects, Parks/Outdoor, People, Religion, Science, Signs/Symbols, Sports/Recreation, Technology, Transportation, Vintage]

FORMAT YOUR RESPONSE EXACTLY AS:
TITLE: [title]
DESCRIPTION: [description]
KEYWORDS: [keyword1, keyword2, keyword3, ...]
CATEGORIES: [category1, category2]

Analyze the image now:"""

    # Title-only template
    TITLE_TEMPLATE = """Analyze this image and provide a concise, SEO-optimized title.

REQUIREMENTS:
- Maximum {title_max} characters
- Descriptive and searchable
- No filler words
- Commercial stock photography style

FORMAT:
TITLE: [your title here]

Analyze now:"""

    # Description-only template
    DESCRIPTION_TEMPLATE = """Analyze this image and provide a detailed description.

REQUIREMENTS:
- Maximum {desc_max} characters
- Describe the scene, subjects, colors, mood
- Commercial stock photography style
- Focus on visual elements

FORMAT:
DESCRIPTION: [your description here]

Analyze now:"""

    # Keywords-only template
    KEYWORDS_TEMPLATE = """Analyze this image and generate relevant keywords for stock photography.

REQUIREMENTS:
- Between {kw_min} and {kw_max} keywords
- Include specific and general terms
- Consider: subject, action, emotion, color, style, concept
- Comma-separated list
- No duplicates

FORMAT:
KEYWORDS: [keyword1, keyword2, keyword3, ...]

Analyze now:"""

    # Categories-only template
    CATEGORIES_TEMPLATE = """Analyze this image and assign 1-2 categories.

AVAILABLE CATEGORIES:
Animals, Arts, Backgrounds/Textures, Beauty/Fashion, Buildings/Landmarks,
Business/Finance, Celebrities, Editorial, Education, Food/Drink,
Healthcare/Medical, Holidays, Industrial, Interiors, Miscellaneous,
Nature, Objects, Parks/Outdoor, People, Religion, Science,
Signs/Symbols, Sports/Recreation, Technology, Transportation, Vintage

FORMAT:
CATEGORIES: [category1, category2]

Analyze now:"""

    # Editorial content detection
    EDITORIAL_CHECK_TEMPLATE = """Analyze this image for editorial content.

Check for:
1. Recognizable people (celebrities, public figures)
2. Brand logos or trademarks
3. Copyrighted artwork
4. News events
5. Private property without release

RESPOND WITH:
EDITORIAL: [YES/NO]
REASON: [brief explanation if YES]"""

    def __init__(self, platform: Platform = Platform.SHUTTERSTOCK):
        """
        Initialize template manager

        Args:
            platform: Target platform for limits
        """
        self.platform = platform
        self.limits = PLATFORM_LIMITS.get(platform, PLATFORM_LIMITS[Platform.GENERIC])

    def get_prompt(
        self, prompt_type: PromptType = PromptType.FULL, custom_instructions: str = None, language: str = "en"
    ) -> str:
        """
        Get formatted prompt for analysis

        Args:
            prompt_type: Type of metadata to generate
            custom_instructions: Additional instructions
            language: Target language

        Returns:
            Formatted prompt string
        """
        from ..analysis.limits import (
            DESCRIPTION_TARGET_MAX,
            DESCRIPTION_TARGET_MIN,
            KEYWORDS_TARGET_MAX,
            KEYWORDS_TARGET_MIN,
            TITLE_TARGET_MAX,
            TITLE_TARGET_MIN,
        )

        full_kwargs = {
            "title_max": self.limits.title_max,
            "desc_max": self.limits.description_max,
            "title_target_min": TITLE_TARGET_MIN,
            "title_target_max": TITLE_TARGET_MAX,
            "desc_target_min": DESCRIPTION_TARGET_MIN,
            "desc_target_max": DESCRIPTION_TARGET_MAX,
            "kw_target_min": max(KEYWORDS_TARGET_MIN, self.limits.keywords_min),
            "kw_target_max": min(KEYWORDS_TARGET_MAX, self.limits.keywords_max),
        }

        if prompt_type == PromptType.FULL:
            template = self.FULL_ANALYSIS_TEMPLATE.format(**full_kwargs)
        elif prompt_type == PromptType.TITLE_ONLY:
            template = self.TITLE_TEMPLATE.format(title_max=self.limits.title_max)
        elif prompt_type == PromptType.DESCRIPTION_ONLY:
            template = self.DESCRIPTION_TEMPLATE.format(desc_max=self.limits.description_max)
        elif prompt_type == PromptType.KEYWORDS_ONLY:
            template = self.KEYWORDS_TEMPLATE.format(kw_min=self.limits.keywords_min, kw_max=self.limits.keywords_max)
        elif prompt_type == PromptType.CATEGORIES_ONLY:
            template = self.CATEGORIES_TEMPLATE
        else:
            template = self.FULL_ANALYSIS_TEMPLATE.format(**full_kwargs)

        # Add language instruction if not English
        if language and language.lower() != "en":
            template = f"Respond in {language}.\n\n" + template

        # Add custom instructions
        if custom_instructions:
            template = f"{custom_instructions}\n\n{template}"

        return template

    def get_editorial_check_prompt(self) -> str:
        """Get prompt for editorial content check"""
        return self.EDITORIAL_CHECK_TEMPLATE

    def parse_response(self, response: str) -> Dict[str, any]:
        """
        Parse AI response into structured metadata

        Args:
            response: Raw AI response text

        Returns:
            Dictionary with parsed fields
        """
        result = {
            "title": None,
            "description": None,
            "keywords": [],
            "categories": [],
            "editorial": False,
            "raw_response": response,
        }

        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()

            if line.upper().startswith("TITLE:"):
                result["title"] = self._clean_value(line[6:])

            elif line.upper().startswith("DESCRIPTION:"):
                result["description"] = self._clean_value(line[12:])

            elif line.upper().startswith("KEYWORDS:"):
                keywords_str = self._clean_value(line[9:])
                result["keywords"] = self._parse_keywords(keywords_str)

            elif line.upper().startswith("CATEGORIES:"):
                cats_str = self._clean_value(line[11:])
                result["categories"] = self._parse_categories(cats_str)

            elif line.upper().startswith("EDITORIAL:"):
                value = self._clean_value(line[10:]).upper()
                result["editorial"] = value.startswith("YES")

        # Validate and trim to limits
        result = self._apply_limits(result)

        return result

    def _clean_value(self, value: str) -> str:
        """Clean extracted value"""
        value = value.strip()
        # Remove surrounding quotes
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        return value.strip()

    def _parse_keywords(self, keywords_str: str) -> List[str]:
        """Parse keywords string into list"""
        # Handle various separators
        keywords_str = keywords_str.replace(";", ",")
        keywords = [k.strip() for k in keywords_str.split(",")]
        # Filter empty and clean
        keywords = [k for k in keywords if k and len(k) <= self.limits.keyword_max_chars]
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for k in keywords:
            kl = k.lower()
            if kl not in seen:
                seen.add(kl)
                unique.append(k)
        return unique

    def _parse_categories(self, cats_str: str) -> List[str]:
        """Parse categories string into list"""
        cats = [c.strip() for c in cats_str.split(",")]
        # Map to valid categories
        valid_categories = [
            "Animals",
            "Arts",
            "Backgrounds/Textures",
            "Beauty/Fashion",
            "Buildings/Landmarks",
            "Business/Finance",
            "Celebrities",
            "Editorial",
            "Education",
            "Food/Drink",
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

        result = []
        for cat in cats:
            # Find best match
            for valid in valid_categories:
                if cat.lower() in valid.lower() or valid.lower() in cat.lower():
                    if valid not in result:
                        result.append(valid)
                    break

        return result[:2]  # Max 2 categories

    def _apply_limits(self, result: Dict) -> Dict:
        """Enforce platform budgets on the parsed AI result.

        - Title/description are word-boundary trimmed (no "…" ellipsis)
          via ``smart_truncate`` so the portal field never overflows.
        - Keywords are filtered against brand + stuffing lists (defense
          in depth — the prompt forbids them too) and capped.
        """
        from ..analysis.expert_report import BRAND_KEYWORDS, STUFFING_KEYWORDS
        from ..analysis.limits import clamp_keywords, smart_truncate

        if result["title"]:
            result["title"] = smart_truncate(result["title"], self.limits.title_max)

        if result["description"]:
            result["description"] = smart_truncate(result["description"], self.limits.description_max)

        # Drop brands and meta/stuffing words the model may have slipped in.
        title_words = set((result.get("title") or "").lower().split())
        filtered = []
        for kw in result["keywords"]:
            kl = kw.lower().strip()
            if kl in BRAND_KEYWORDS:
                continue
            # Stuffing words are allowed only if they appear in the title
            # (there they describe the image rather than pad the list).
            if kl in STUFFING_KEYWORDS and kl not in title_words:
                continue
            filtered.append(kw)

        result["keywords"] = clamp_keywords(filtered, self.limits.keywords_max)
        return result

    @staticmethod
    def get_supported_platforms() -> List[str]:
        """Get list of supported platforms"""
        return [p.value for p in Platform]

    def change_platform(self, platform: Platform):
        """Change target platform"""
        self.platform = platform
        self.limits = PLATFORM_LIMITS.get(platform, PLATFORM_LIMITS[Platform.GENERIC])
