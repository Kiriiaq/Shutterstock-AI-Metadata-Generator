"""Official metadata limits for Adobe Stock and Shutterstock.

Single source of truth used by the CSV exporters, the IPTC engine, the
expert report and the AI post-processing. The goal: text that always
fits the portal fields, so the contributor never sees a truncated
"…" — neither in our UI nor in the portal review queue.

Documented limits (contributor portals, 2026):

Adobe Stock
    - Title: 200 characters max (~70 recommended for search display)
    - Keywords: 49 max, first 10 carry the most search weight
    - No separate description field (title doubles as description)

Shutterstock
    - Description: 200 characters max (acts as the title in search)
    - Keywords: 7 min, 50 max
    - Title/description under ~100 chars display best in search grids

``smart_truncate`` cuts at a word boundary and NEVER appends an
ellipsis: a clean shorter sentence beats "Vibrant business team br…".
"""

from __future__ import annotations

from dataclasses import dataclass

# Hard portal caps
ADOBE_TITLE_MAX = 200
ADOBE_KEYWORDS_MAX = 49
SHUTTERSTOCK_DESCRIPTION_MAX = 200
SHUTTERSTOCK_KEYWORDS_MAX = 50
SHUTTERSTOCK_KEYWORDS_MIN = 7
KEYWORD_MAX_CHARS = 50

# Editorial sweet spots (what we ask the AI to aim for — well under the
# caps so nothing ever needs trimming downstream)
TITLE_TARGET_MIN = 45
TITLE_TARGET_MAX = 95
DESCRIPTION_TARGET_MIN = 90
DESCRIPTION_TARGET_MAX = 190
KEYWORDS_TARGET_MIN = 25
KEYWORDS_TARGET_MAX = 45


@dataclass(frozen=True)
class PlatformSpec:
    """Metadata caps for one platform."""

    name: str
    title_max: int
    description_max: int
    keywords_min: int
    keywords_max: int


ADOBE = PlatformSpec(
    name="Adobe Stock",
    title_max=ADOBE_TITLE_MAX,
    description_max=ADOBE_TITLE_MAX,  # title doubles as description
    keywords_min=5,
    keywords_max=ADOBE_KEYWORDS_MAX,
)

SHUTTERSTOCK = PlatformSpec(
    name="Shutterstock",
    title_max=SHUTTERSTOCK_DESCRIPTION_MAX,
    description_max=SHUTTERSTOCK_DESCRIPTION_MAX,
    keywords_min=SHUTTERSTOCK_KEYWORDS_MIN,
    keywords_max=SHUTTERSTOCK_KEYWORDS_MAX,
)


def smart_truncate(text: str, max_len: int) -> str:
    """Shorten ``text`` to ``max_len`` at a word boundary, no ellipsis.

    - Strips surrounding whitespace first; returns as-is when it fits.
    - Cuts at the last space before the limit so no word is chopped.
    - Drops a trailing comma/semicolon/dash left dangling by the cut.
    - Falls back to a hard cut only when the first word alone exceeds
      the limit (pathological input).
    """
    text = (text or "").strip()
    if len(text) <= max_len:
        return text

    cut = text[:max_len]
    space = cut.rfind(" ")
    if space > 0:
        cut = cut[:space]
    cut = cut.rstrip(" ,;:-–—.")
    return cut


def clamp_keywords(keywords: list[str], max_count: int) -> list[str]:
    """Cap the keyword list, dropping empties and >50-char outliers."""
    out: list[str] = []
    for kw in keywords:
        kw = (kw or "").strip()
        if not kw or len(kw) > KEYWORD_MAX_CHARS:
            continue
        out.append(kw)
        if len(out) >= max_count:
            break
    return out
