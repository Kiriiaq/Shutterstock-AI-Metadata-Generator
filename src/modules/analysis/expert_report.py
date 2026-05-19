"""Expert microstock report builder — AI-optional.

Two entry points:

- :func:`build_expert_report` — pure heuristic, no AI. Reads existing
  IPTC metadata + does a cheap PIL probe on the file. Suitable for
  low-power machines or batch runs where Ollama isn't installed.

- :func:`build_expert_report_from_ai` — runs the heuristic first,
  then overlays an AI analysis result on top (visual defect
  detection, refined titles, improved keywords). Falls back to the
  heuristic-only output if the AI result is missing or malformed.

Design posture: **lax**. The audit calling this code already covered
the "Shutterstock/Adobe pre-filter on their side" angle — we don't
want to gate uploads here. Every warning is informational; the
scores are coarse 0-10 guides, not strict thresholds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..models.metadata_models import (
    ADOBE_STOCK_CATEGORIES,
    SHUTTERSTOCK_CATEGORIES,
    ExpertMetadataReport,
    ExpertScores,
    ImageMetadata,
    IPTCFields,
    RejectionRisk,
    TechnicalFlags,
    map_shutterstock_to_adobe,
)
from .platform_compliance import (
    ADOBE_MAX_FILE_MB,
    ADOBE_MAX_MEGAPIXELS,
    MIN_MEGAPIXELS,
    SHUTTERSTOCK_MAX_FILE_MB,
    PlatformCompliance,
    check_platform_compliance,
)

logger = logging.getLogger(__name__)


# Shutterstock categories that historically perform well for paid
# campaigns — used as a tiny commercial-score booster. Source: top
# 10 categories by RPI in Shutterstock contributor reports 2023-2024.
HIGH_VALUE_CATEGORIES = {
    "Business/Finance",
    "Healthcare/Medical",
    "Technology",
    "Food and drink",
    "Beauty/Fashion",
    "People",
}


# Marketing/usage suggestions by Shutterstock category. Used when no
# AI is available to produce them. Kept short — these are starting
# points the contributor refines, not a finished list.
MARKETING_USES: Dict[str, List[str]] = {
    "Business/Finance": [
        "publicité fintech",
        "présentation corporate",
        "landing page B2B",
        "article économique",
        "rapport annuel",
    ],
    "Healthcare/Medical": [
        "publicité santé",
        "campagne pharma",
        "site clinique",
        "blog médical",
        "brochure assurance",
    ],
    "Technology": [
        "landing page SaaS",
        "campagne IA",
        "présentation startup",
        "blog tech",
        "publicité digitale",
    ],
    "People": [
        "campagne lifestyle",
        "réseaux sociaux",
        "campagne RH",
        "bannière web",
        "article magazine",
    ],
    "Nature": [
        "fond éditorial",
        "campagne écologique",
        "tourisme",
        "publicité outdoor",
        "calendrier",
    ],
    "Parks/Outdoor": [
        "tourisme",
        "campagne outdoor",
        "site voyage",
        "blog nature",
    ],
    "Food and drink": [
        "publicité restauration",
        "menu",
        "blog culinaire",
        "campagne marque alimentaire",
    ],
    "Sports/Recreation": [
        "publicité sport",
        "campagne fitness",
        "événement sportif",
        "presse sportive",
    ],
    "Education": [
        "site éducatif",
        "manuel scolaire",
        "brochure formation",
        "présentation académique",
    ],
    "Beauty/Fashion": [
        "campagne mode",
        "publicité cosmétique",
        "magazine lifestyle",
        "site e-commerce",
    ],
    "Buildings/Landmarks": [
        "tourisme",
        "agence immobilière",
        "presse architecture",
        "guide ville",
    ],
    "Interiors": [
        "décoration",
        "magazine déco",
        "site immobilier",
        "publicité mobilier",
    ],
    "Industrial": [
        "rapport industriel",
        "site B2B industrie",
        "publicité énergie",
        "brochure technique",
    ],
    "Transportation": [
        "publicité automobile",
        "campagne logistique",
        "site voyage",
        "presse transport",
    ],
}

DEFAULT_MARKETING_USES = [
    "réseaux sociaux",
    "bannière web",
    "article éditorial",
    "blog",
]


# Buyer profiles by Shutterstock category — same logic as marketing
# uses: a coarse default list per category, finished by the user.
BUYER_PROFILES: Dict[str, List[str]] = {
    "Business/Finance": ["agence marketing B2B", "cabinet de conseil", "fintech", "média économique"],
    "Healthcare/Medical": ["pharma", "clinique", "média santé", "agence santé"],
    "Technology": ["SaaS", "startup tech", "agence digitale", "presse IT"],
    "People": ["agence lifestyle", "service RH", "magazine grand public"],
    "Nature": ["agence tourisme", "média écologique", "campagne ONG"],
    "Food and drink": ["restaurateur", "marque alimentaire", "blog food"],
    "Sports/Recreation": ["marque sport", "club fitness", "presse sportive"],
    "Education": ["éditeur scolaire", "organisme de formation", "ed-tech"],
    "Beauty/Fashion": ["marque mode", "e-commerce", "magazine lifestyle"],
    "Buildings/Landmarks": ["office du tourisme", "agence immobilière", "presse architecture"],
    "Technology;": ["SaaS", "agence digitale"],
}

DEFAULT_BUYER_PROFILES = ["agence marketing", "blog éditorial", "presse en ligne"]


TRENDS_BY_CATEGORY: Dict[str, List[str]] = {
    "Technology": ["IA générative", "automatisation", "cybersécurité"],
    "Business/Finance": ["télétravail", "diversité", "ESG"],
    "People": ["authenticité", "diversité", "lifestyle naturel"],
    "Healthcare/Medical": ["télémédecine", "bien-être", "santé mentale"],
    "Nature": ["sustainability", "low impact", "minimalisme"],
    "Food and drink": ["végétal", "circuits courts", "comfort food"],
}

DEFAULT_TRENDS = ["minimaliste", "lumière naturelle"]


# Keywords that frequently trigger keyword-stuffing flags on review.
# We strip them silently from the output unless they're also in the
# title (which means they describe the image, not pad the list).
STUFFING_KEYWORDS = {
    "stock",
    "image",
    "photo",
    "picture",
    "wallpaper",
    "background",
    "shutterstock",
    "adobe",
}


# Brand-name keywords that auto-disqualify on both platforms. Kept
# short and obvious — the AI pass catches the long tail.
BRAND_KEYWORDS = {
    "apple",
    "iphone",
    "nike",
    "adidas",
    "coca-cola",
    "coca cola",
    "pepsi",
    "google",
    "microsoft",
    "facebook",
    "instagram",
    "tiktok",
    "bmw",
    "mercedes",
    "ferrari",
    "rolex",
    "starbucks",
    "mcdonald",
    "disney",
    "marvel",
}


# ============================================================================
# Public API
# ============================================================================


def build_expert_report(
    file_path: Path,
    *,
    iptc: Optional[IPTCFields] = None,
    image_metadata: Optional[ImageMetadata] = None,
    compliance: Optional[PlatformCompliance] = None,
) -> ExpertMetadataReport:
    """Build a complete expert report without calling any AI model.

    All inputs are optional — the builder degrades gracefully:

    - No ``iptc``: empty title/description/keywords, scores reflect
      the gap (SEO ~ 0, commercial low).
    - No ``image_metadata``: dimensions come from the PIL probe inside
      :func:`check_platform_compliance`.
    - No ``compliance``: it's computed on-the-fly.

    The result is fully serialisable and can be handed straight to
    the CSV exporter or to the UI.
    """
    path = Path(file_path)

    if compliance is None:
        compliance = check_platform_compliance(path)

    iptc = iptc or (image_metadata.iptc if image_metadata else IPTCFields())

    title = _pick_title(iptc, path)
    description = iptc.caption or title
    keywords = _clean_keywords(iptc.keywords or [], title=title)

    categories_shutterstock = _normalise_shutterstock_categories(
        iptc.supplemental_categories or []
    )
    category_adobe_primary = _pick_adobe_primary(categories_shutterstock)

    flags = TechnicalFlags()  # all False — heuristic mode can't detect visual defects

    scores = _compute_scores(
        compliance=compliance,
        title=title,
        description=description,
        keywords=keywords,
        categories=categories_shutterstock,
        flags=flags,
    )

    risks = _detect_rejection_risks(
        compliance=compliance,
        title=title,
        keywords=keywords,
        flags=flags,
    )

    improvements = _suggest_improvements(
        title=title,
        description=description,
        keywords=keywords,
        categories=categories_shutterstock,
        compliance=compliance,
    )

    primary_cat = categories_shutterstock[0] if categories_shutterstock else ""
    marketing_uses = MARKETING_USES.get(primary_cat, DEFAULT_MARKETING_USES).copy()
    buyer_profiles = BUYER_PROFILES.get(primary_cat, DEFAULT_BUYER_PROFILES).copy()
    trends = TRENDS_BY_CATEGORY.get(primary_cat, DEFAULT_TRENDS).copy()

    return ExpertMetadataReport(
        file_path=path,
        source="heuristic",
        scores=scores,
        title_adobe=title,
        title_shutterstock=title,
        description=description,
        keywords=keywords,
        category_adobe_primary=category_adobe_primary,
        category_adobe_secondary="",
        categories_shutterstock=categories_shutterstock,
        rejection_risks=risks,
        improvements=improvements,
        marketing_uses=marketing_uses,
        buyer_profiles=buyer_profiles,
        trends=trends,
        technical_flags=flags,
        adobe_warnings=compliance.adobe_warnings,
        shutterstock_warnings=compliance.shutterstock_warnings,
    )


def build_expert_report_from_ai(
    file_path: Path,
    ai_result: Dict[str, Any],
    *,
    iptc: Optional[IPTCFields] = None,
    image_metadata: Optional[ImageMetadata] = None,
    compliance: Optional[PlatformCompliance] = None,
) -> ExpertMetadataReport:
    """Build a report from a heuristic baseline + overlay AI fields.

    ``ai_result`` is the dict returned by ``VisionAnalyzer.analyze_image``
    (or any compatible source). The function never throws on missing
    keys — it merges what it can and keeps the heuristic value
    otherwise.
    """
    report = build_expert_report(
        file_path,
        iptc=iptc,
        image_metadata=image_metadata,
        compliance=compliance,
    )
    return enrich_with_ai_result(report, ai_result)


def enrich_with_ai_result(
    report: ExpertMetadataReport,
    ai_result: Dict[str, Any],
) -> ExpertMetadataReport:
    """Overlay an AI dict onto an existing heuristic report.

    Mutates and returns the input report — callers that want immutable
    behaviour should ``copy.deepcopy`` before. Keeps the contract loose
    so any of these shapes work::

        {"title": "...", "keywords": [...], "categories": [...]}
        {"title_adobe": "...", "title_shutterstock": "..."}

    Missing keys are simply not applied.
    """
    if not isinstance(ai_result, dict):
        logger.debug("enrich_with_ai_result: non-dict input, skipping")
        return report

    # Title — single or split
    title_adobe = ai_result.get("title_adobe") or ai_result.get("title")
    title_shutterstock = ai_result.get("title_shutterstock") or ai_result.get("title")
    if title_adobe:
        report.title_adobe = str(title_adobe).strip()[:200]
    if title_shutterstock:
        report.title_shutterstock = str(title_shutterstock).strip()[:200]

    if ai_result.get("description"):
        report.description = str(ai_result["description"]).strip()[:200]

    if ai_result.get("keywords"):
        merged = _merge_keywords(report.keywords, ai_result["keywords"], title=report.title_shutterstock)
        report.keywords = merged

    if ai_result.get("categories"):
        cats = _normalise_shutterstock_categories(ai_result["categories"])
        if cats:
            report.categories_shutterstock = cats
            report.category_adobe_primary = _pick_adobe_primary(cats)

    # Adobe-explicit categories override the mapping.
    if ai_result.get("category_adobe_primary"):
        cap = str(ai_result["category_adobe_primary"]).strip()
        if cap in ADOBE_STOCK_CATEGORIES:
            report.category_adobe_primary = cap
    if ai_result.get("category_adobe_secondary"):
        cas = str(ai_result["category_adobe_secondary"]).strip()
        if cas in ADOBE_STOCK_CATEGORIES:
            report.category_adobe_secondary = cas

    # Visual flags. AI returns them as a dict of {flag_name: bool}.
    flags_in = ai_result.get("technical_flags") or {}
    if isinstance(flags_in, dict):
        for name, value in flags_in.items():
            if hasattr(report.technical_flags, name):
                setattr(report.technical_flags, name, bool(value))

    # Editorial / illustration / mature passthrough.
    for k in ("editorial", "illustration", "mature_content"):
        if k in ai_result:
            setattr(report, k, bool(ai_result[k]))

    # Soft-merge marketing/risks/improvements lists.
    for k_in, k_out in (
        ("rejection_risks", "rejection_risks"),
        ("improvements", "improvements"),
        ("marketing_uses", "marketing_uses"),
        ("buyer_profiles", "buyer_profiles"),
        ("trends", "trends"),
    ):
        ai_value = ai_result.get(k_in)
        if not ai_value:
            continue
        if k_out == "rejection_risks":
            extra = _coerce_rejection_risks(ai_value)
            report.rejection_risks = _dedupe_risks(report.rejection_risks + extra)
        else:
            current = getattr(report, k_out)
            merged_list = list(dict.fromkeys([*current, *(str(x) for x in ai_value if x)]))
            setattr(report, k_out, merged_list[:20])

    # Scores — if the AI provided them, trust them but clamp.
    ai_scores = ai_result.get("scores")
    if isinstance(ai_scores, dict):
        report.scores = ExpertScores(
            commercial=_clamp_score(ai_scores.get("commercial", report.scores.commercial)),
            technical=_clamp_score(ai_scores.get("technical", report.scores.technical)),
            seo=_clamp_score(ai_scores.get("seo", report.scores.seo)),
            rejection_risk=_clamp_score(ai_scores.get("rejection_risk", report.scores.rejection_risk)),
        )

    report.source = "hybrid" if report.source == "heuristic" else "ai"
    return report


# ============================================================================
# Heuristics
# ============================================================================


def _pick_title(iptc: IPTCFields, path: Path) -> str:
    """Best-effort title from IPTC fields, with filename fallback."""
    for candidate in (iptc.headline, iptc.object_name):
        if candidate and candidate.strip():
            return candidate.strip()[:200]
    # Filename without extension, cleaned: "my_photo-01.jpg" -> "My Photo 01"
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    return stem.title()[:200] if stem else ""


def _clean_keywords(keywords: Iterable[str], *, title: str) -> List[str]:
    """Trim, dedupe, drop brands + stuffing terms. Cap at 50.

    Keywords appearing in the title are preserved even if they match
    the anti-stuffing list — they describe the image, they're not
    padding.
    """
    title_words = {w.lower().strip(".,") for w in (title or "").split()}
    seen, out = set(), []

    for raw in keywords or []:
        if raw is None:
            continue
        k = str(raw).strip().lower()
        if len(k) < 2 or len(k) > 50:
            continue
        if k in seen:
            continue

        if k in BRAND_KEYWORDS:
            continue
        if k in STUFFING_KEYWORDS and k not in title_words:
            continue

        seen.add(k)
        out.append(k)
        if len(out) >= 50:
            break

    return out


def _merge_keywords(base: List[str], extra: Iterable[Any], *, title: str) -> List[str]:
    """Merge two keyword lists, preserving the order of ``base``."""
    cleaned_extra = _clean_keywords([str(x) for x in extra if x], title=title)
    seen, out = set(), []
    for k in [*base, *cleaned_extra]:
        kl = k.lower()
        if kl not in seen:
            seen.add(kl)
            out.append(k)
        if len(out) >= 50:
            break
    return out


def _normalise_shutterstock_categories(raw: Iterable[Any]) -> List[str]:
    """Pick valid Shutterstock categories from a free-form list."""
    out: List[str] = []
    for item in raw or []:
        if not item:
            continue
        s = str(item).strip()
        # Exact match first
        if s in SHUTTERSTOCK_CATEGORIES and s not in out:
            out.append(s)
            continue
        # Case-insensitive containment fallback
        sl = s.lower()
        for cat in SHUTTERSTOCK_CATEGORIES:
            if sl == cat.lower() or sl in cat.lower() or cat.lower() in sl:
                if cat not in out:
                    out.append(cat)
                break
        if len(out) >= 2:  # Shutterstock takes at most 2
            break
    return out


def _pick_adobe_primary(shutterstock_cats: List[str]) -> str:
    """Map the first Shutterstock category to Adobe."""
    if not shutterstock_cats:
        return ""
    return map_shutterstock_to_adobe(shutterstock_cats[0])


def _compute_scores(
    *,
    compliance: PlatformCompliance,
    title: str,
    description: str,
    keywords: List[str],
    categories: List[str],
    flags: TechnicalFlags,
) -> ExpertScores:
    """Compute the four 0-10 scores. Lax & deterministic."""

    # ----- Technical (0-10) -----
    tech = 10
    if compliance.megapixels and compliance.megapixels < MIN_MEGAPIXELS:
        tech -= 3
    if compliance.megapixels and compliance.megapixels > ADOBE_MAX_MEGAPIXELS:
        tech -= 1
    if compliance.file_size_mb and compliance.file_size_mb > SHUTTERSTOCK_MAX_FILE_MB:
        tech -= 2
    elif compliance.file_size_mb and compliance.file_size_mb > ADOBE_MAX_FILE_MB:
        tech -= 1
    fmt_norm = (compliance.format or "").lower().lstrip(".")
    if fmt_norm and fmt_norm not in {"jpeg", "jpg"}:
        tech -= 2
    if compliance.color_space and compliance.color_space not in {"sRGB", ""}:
        tech -= 1
    if flags.any_active():
        tech -= 2

    # ----- SEO (0-10) -----
    seo = 0
    if title and 30 <= len(title) <= 150:
        seo += 3
    elif title:
        seo += 1
    if description and len(description) >= 50:
        seo += 1
    kw_count = len(keywords)
    if kw_count >= 35:
        seo += 4
    elif kw_count >= 20:
        seo += 3
    elif kw_count >= 10:
        seo += 2
    elif kw_count >= 7:
        seo += 1
    if title and keywords:
        title_lower = title.lower()
        matches = sum(1 for kw in keywords if kw.lower() in title_lower)
        if matches >= 2:
            seo += 1
    if categories:
        seo += 1

    # ----- Commercial (0-10) -----
    commercial = 5
    if categories and any(c in HIGH_VALUE_CATEGORIES for c in categories):
        commercial += 2
    if title and len(title) >= 30:
        commercial += 1
    if compliance.megapixels and compliance.megapixels >= 12:
        commercial += 1
    if not flags.any_active():
        commercial += 1
    else:
        commercial -= 1
    if not keywords or not title:
        commercial -= 2

    # ----- Rejection risk (0-10, higher = worse) -----
    risk = 0
    if compliance.megapixels and compliance.megapixels < MIN_MEGAPIXELS:
        risk += 3
    if compliance.file_size_mb and compliance.file_size_mb > SHUTTERSTOCK_MAX_FILE_MB:
        risk += 1
    if kw_count < 7:
        risk += 2
    if kw_count > 50:
        risk += 1
    if not title:
        risk += 1
    if flags.watermark or flags.logo_or_brand:
        risk += 3
    if flags.needs_model_release or flags.needs_property_release:
        risk += 2
    active_others = sum(
        1
        for name in (
            "noise",
            "soft_focus",
            "jpeg_artifacts",
            "oversharpen",
            "hdr_overprocessed",
            "halos",
            "oversaturated",
            "ai_artifacts",
            "bad_hands",
            "unreadable_text",
        )
        if getattr(flags, name)
    )
    risk += min(active_others, 3)

    return ExpertScores(
        commercial=_clamp_score(commercial),
        technical=_clamp_score(tech),
        seo=_clamp_score(seo),
        rejection_risk=_clamp_score(risk),
    )


def _detect_rejection_risks(
    *,
    compliance: PlatformCompliance,
    title: str,
    keywords: List[str],
    flags: TechnicalFlags,
) -> List[RejectionRisk]:
    """List the concrete reasons a reviewer might refuse the image."""
    risks: List[RejectionRisk] = []

    if compliance.megapixels and compliance.megapixels < MIN_MEGAPIXELS:
        risks.append(
            RejectionRisk(
                issue=f"Résolution {compliance.megapixels:.1f} MP < 4 MP",
                cause="Adobe et Shutterstock exigent 4 MP minimum.",
                fix="Upscaler proprement ou ne pas soumettre cette image.",
                severity="blocker",
            )
        )

    if compliance.file_size_mb and compliance.file_size_mb > SHUTTERSTOCK_MAX_FILE_MB:
        risks.append(
            RejectionRisk(
                issue=f"Poids fichier {compliance.file_size_mb:.1f} Mo > 50 Mo",
                cause="Shutterstock plafonne à 50 Mo, Adobe à 45 Mo.",
                fix="Réduire la qualité JPEG (q=10-11 sur Photoshop) ou ré-encoder.",
                severity="warning",
            )
        )
    elif compliance.file_size_mb and compliance.file_size_mb > ADOBE_MAX_FILE_MB:
        risks.append(
            RejectionRisk(
                issue=f"Poids fichier {compliance.file_size_mb:.1f} Mo > 45 Mo (Adobe)",
                cause="Adobe limite à 45 Mo par fichier.",
                fix="Réduire pour Adobe, accepter tel quel pour Shutterstock.",
                severity="info",
            )
        )

    if len(keywords) < 7:
        risks.append(
            RejectionRisk(
                issue=f"{len(keywords)} mots-clés (minimum 7)",
                cause="Shutterstock rejette en dessous de 7 keywords.",
                fix="Ajouter des mots-clés (sujet, action, contexte, émotion).",
                severity="blocker",
            )
        )
    if len(keywords) > 50:
        risks.append(
            RejectionRisk(
                issue=f"{len(keywords)} mots-clés (maximum 50)",
                cause="Plafond Adobe/Shutterstock à 50.",
                fix="Garder les 50 plus pertinents, les autres sont du bruit.",
                severity="warning",
            )
        )

    if not title:
        risks.append(
            RejectionRisk(
                issue="Titre absent",
                cause="Adobe rejette les soumissions sans titre.",
                fix="Renseigner un titre descriptif de 30-150 caractères.",
                severity="blocker",
            )
        )
    elif len(title) < 10:
        risks.append(
            RejectionRisk(
                issue=f"Titre très court ({len(title)} caractères)",
                cause="Un titre trop court réduit la visibilité SEO.",
                fix="Cible 30-150 caractères, descriptif et naturel.",
                severity="info",
            )
        )

    # Flags only fire when an AI pass ran; in heuristic mode this loop
    # is a no-op (all flags False by default).
    if flags.watermark:
        risks.append(
            RejectionRisk(
                issue="Filigrane détecté",
                cause="Refus immédiat des deux plateformes.",
                fix="Repartir du fichier source sans filigrane.",
                severity="blocker",
            )
        )
    if flags.logo_or_brand:
        risks.append(
            RejectionRisk(
                issue="Logo / marque visible",
                cause="Refus pour droit des marques.",
                fix="Cloner / flouter le logo, ou recadrer.",
                severity="blocker",
            )
        )
    if flags.needs_model_release:
        risks.append(
            RejectionRisk(
                issue="Personne identifiable sans release",
                cause="Adobe et Shutterstock exigent un model release.",
                fix="Obtenir le document signé ou flouter le visage.",
                severity="blocker",
            )
        )
    if flags.needs_property_release:
        risks.append(
            RejectionRisk(
                issue="Bien privé identifiable sans release",
                cause="Property release requis pour les bâtiments / œuvres protégés.",
                fix="Obtenir le document ou recadrer.",
                severity="warning",
            )
        )
    if flags.protected_building:
        risks.append(
            RejectionRisk(
                issue="Bâtiment protégé",
                cause="Tour Eiffel de nuit, Burj Khalifa, etc. — droits architecte.",
                fix="Soumettre en éditorial ou recadrer.",
                severity="warning",
            )
        )
    if flags.bad_hands or flags.ai_artifacts:
        risks.append(
            RejectionRisk(
                issue="Défauts IA visibles",
                cause="Doigts incorrects ou artefacts génératifs.",
                fix="Retoucher les zones concernées.",
                severity="warning",
            )
        )

    return risks


def _suggest_improvements(
    *,
    title: str,
    description: str,
    keywords: List[str],
    categories: List[str],
    compliance: PlatformCompliance,
) -> List[str]:
    """Concrete, actionable tweaks."""
    suggestions: List[str] = []

    if len(keywords) < 20:
        suggestions.append("Étoffer les mots-clés (cible 20-35 pour le SEO).")
    if not categories:
        suggestions.append("Renseigner au moins une catégorie Shutterstock.")
    if title and len(title) < 30:
        suggestions.append("Allonger le titre (30-150 caractères, descriptif et naturel).")
    if not description or len(description) < 50:
        suggestions.append("Ajouter une description orientée bénéfice acheteur.")
    if compliance.color_space and compliance.color_space == "CMYK":
        suggestions.append("Convertir l'image en sRGB avant export JPEG.")

    return suggestions


def _coerce_rejection_risks(items: Iterable[Any]) -> List[RejectionRisk]:
    """Accept dicts or RejectionRisk instances; drop the rest."""
    out: List[RejectionRisk] = []
    for it in items:
        if isinstance(it, RejectionRisk):
            out.append(it)
        elif isinstance(it, dict) and it.get("issue"):
            out.append(
                RejectionRisk(
                    issue=str(it.get("issue", "")),
                    cause=str(it.get("cause", "")),
                    fix=str(it.get("fix", "")),
                    severity=str(it.get("severity", "warning")),
                )
            )
    return out


def _dedupe_risks(risks: List[RejectionRisk]) -> List[RejectionRisk]:
    seen, out = set(), []
    for r in risks:
        key = (r.issue.lower(), r.severity)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _clamp_score(value: Any) -> int:
    """Coerce to int and clamp to [0, 10]."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    if n < 0:
        return 0
    if n > 10:
        return 10
    return n
