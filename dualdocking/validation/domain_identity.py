"""Domain-specific sequence identity analysis.

Functions for selecting, matching, and comparing protein domain regions
to compute domain-specific NW identity and compare with catalog claims.
Domain boundaries are auto-discovered from UniProt features and matched
against catalog literature descriptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dualdocking.validation.nw_align import AlignmentResult

if TYPE_CHECKING:
    from dualdocking.validation.pair_catalog import PairRecord

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DomainIdentityReport:
    """Domain-specific NW identity verification result."""

    alignment: AlignmentResult | None = None
    domain_a: str = ""  # e.g. "Protein kinase (712-979)"
    domain_b: str = ""  # e.g. "Protein kinase (720-987)"
    range_a: str = ""  # e.g. "712-979"
    range_b: str = ""  # e.g. "720-987"
    domain_type: str = ""  # parsed type from catalog (kinase, LBD, etc.)
    catalog_claim_pct: str = ""  # e.g. "~80%"
    match_quality: str = ""  # "match", "close", "mismatch", "n/a"
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

# Domain range overrides for proteins without standard UniProt Domain features
_DOMAIN_RANGE_OVERRIDES: dict[str, tuple[str, int, int]] = {
    "P0DTD1": ("Mpro (nsp5)", 3264, 3569),  # SARS-CoV-2 pp1ab
    "P0DTC1": ("PLpro (nsp3 catalytic)", 1634, 1898),  # SARS-CoV-2 pp1a
}

# Map catalog domain keywords -> UniProt Domain description patterns
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "kinase": ["Protein kinase", "PI3K/PI4K"],
    "LBD": ["NR LBD"],
    "SET": ["SET"],
    "BD1": ["Bromo"],
    "BH3": ["BH3", "Bcl-2 homology"],
    "JmjC": ["JmjC"],
    "sirtuin": ["Deacetylase sirtuin", "Sirtuin"],
    "catalytic": ["catalytic", "Peptidase", "Amine oxidase", "Methyltransferase"],
    "HDAC": ["Histone deacetylase"],
    "PDE": ["HDc"],
    "overall": [],
    "active-site": [],
    "none": [],
}


# ---------------------------------------------------------------------------
# Domain parsing & selection
# ---------------------------------------------------------------------------


def _parse_catalog_domain_type(similarity_pct: str) -> str:  # noqa: PLR0911
    """Extract domain type keyword from catalog similarity text.

    Examples::

        "~80% kinase domain ID" -> "kinase"
        "~56% LBD ID" -> "LBD"
        "~70% overall seq ID" -> "overall"
        "Different protein classes" -> "none"
        "84% active-site residue identity" -> "active-site"
    """
    text = similarity_pct.lower()

    # "None" indicators -- different folds, no shared domain
    none_patterns = [
        "different protein classes",
        "different folds",
        "no structural homology",
        "different cys protease",
        "14-tm transporter",
        "sdr dehydrogenase",
        "non-heme fe vs heme",
        "nuclear receptor vs gpcr",
        "nlr sensor nacht domain vs",
    ]
    if any(p in text for p in none_patterns):
        return "none"

    # Specific domain keywords (order matters -- more specific first)
    if "set domain" in text:
        return "SET"
    if "bd1" in text:
        return "BD1"
    if "bh3" in text:
        return "BH3"
    if "jmjc" in text:
        return "JmjC"
    if "lbd" in text:
        return "LBD"
    if "active-site" in text or "active site" in text:
        return "active-site"
    if "catalytic core" in text and "sirtuin" in text:
        return "sirtuin"
    if "cat domain" in text or "catalytic domain" in text or "catalytic mechanism" in text:
        return "catalytic"
    if "catalytic core" in text:
        return "sirtuin"
    if "adomet fold" in text:
        return "catalytic"
    if "pde" in text.split():
        return "PDE"
    if "kinase domain" in text or "kinase domains" in text:
        return "kinase"
    if any(
        p in text
        for p in [
            "pikk superfamily",
            "split-kinase",
            "rtk family",
            "rtk families",
            "cmgc",
            "tkl kinase",
            "atp-site",
            "atp site",
            "atp pocket",
            "atp-cleft",
            "protein tyr kinase",
            "general kinase fold",
        ]
    ):
        return "kinase"
    if "kinase" in text and ("%" in text or "homologous" in text):
        return "kinase"
    if "overall" in text:
        return "overall"
    # Pairs with just a percentage and family context but no explicit domain
    if "binding site" in text or "binding pocket" in text:
        return "kinase"  # CDK2/ERK2, WEE1/CHK1
    if "orthosteric" in text or "allosteric tmd" in text:
        return "GPCR"
    if re.search(r"gorge|catalytic triad|cholinesterase", text):
        return "overall"
    if re.search(r"substrate.cavity", text):
        return "overall"
    if re.search(r"<\s*\d+%", text) and "different" in text:
        return "none"
    return "unknown"


def _extract_catalog_pct(similarity_pct: str) -> str:
    """Extract the percentage claim from similarity text.

    Returns the first percentage found, e.g. "~80%" or "49%" or "85-92%".
    """
    # Match patterns like ~80%, 49%, 85-92%, <30%, <15%
    m = re.search(r"[~<]?\d+(?:\s*[–-]\s*\d+)?%", similarity_pct)
    return m.group(0) if m else ""


def _match_domains_by_keywords(
    keyword: str,
    domains_a: list[dict[str, Any]],
    domains_b: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Find matching domains in both proteins by keyword patterns."""
    patterns = _DOMAIN_KEYWORDS.get(keyword, [])
    if not patterns:
        return None, None

    def _find(domains: list[dict[str, Any]]) -> dict[str, Any] | None:
        for pat in patterns:
            pat_lower = pat.lower()
            for d in domains:
                if pat_lower in d["name"].lower():
                    return d
        return None

    return _find(domains_a), _find(domains_b)


def _find_shared_domain(
    domains_a: list[dict[str, Any]],
    domains_b: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Find a shared domain name between two protein domain lists."""
    for da in domains_a:
        name_a = da["name"].lower()
        for db in domains_b:
            if name_a == db["name"].lower():
                return da, db
    return None, None


def _largest_domain(domains: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the largest domain by residue span."""
    if not domains:
        return None
    return max(domains, key=lambda d: d["end"] - d["start"])


def select_domains_for_pair(
    pair: PairRecord,
    uniprot_a: str,
    uniprot_b: str,
) -> tuple[str, str, str, str, str]:
    """Select domain ranges for domain-specific NW alignment.

    Fetches domain annotations from UniProt and selects the most appropriate
    domain regions to align based on the catalog similarity description.

    Returns:
        Tuple of (range_a, range_b, domain_name_a, domain_name_b, domain_type).
        Empty range strings mean "use full-length".
    """
    domain_type = _parse_catalog_domain_type(pair.similarity_pct)

    # No domain alignment for these categories
    if domain_type in ("none", "overall", "active-site", "GPCR"):
        return "", "", "", "", domain_type

    # Check manual overrides first (e.g. SARS-CoV-2 polyproteins)
    override_a = _DOMAIN_RANGE_OVERRIDES.get(uniprot_a)
    override_b = _DOMAIN_RANGE_OVERRIDES.get(uniprot_b)
    if override_a and override_b:
        return (
            f"{override_a[1]}-{override_a[2]}",
            f"{override_b[1]}-{override_b[2]}",
            override_a[0],
            override_b[0],
            domain_type,
        )

    # Fetch UniProt domains (lazy import to avoid circular dependency)
    from dualdocking.validation.pair_verify import fetch_uniprot_domains

    try:
        domains_a = fetch_uniprot_domains(uniprot_a)
        domains_b = fetch_uniprot_domains(uniprot_b)
    except Exception as exc:
        from loguru import logger

        logger.warning("Failed to fetch domains: {}", exc)
        return "", "", "", "", domain_type

    if not domains_a and not domains_b:
        return "", "", "", "", domain_type

    # Try keyword matching
    da, db = _match_domains_by_keywords(domain_type, domains_a, domains_b)
    if da and db:
        return (
            f"{da['start']}-{da['end']}",
            f"{db['start']}-{db['end']}",
            da["name"],
            db["name"],
            domain_type,
        )

    # If only one side matched by keyword, try finding any match on the other
    if da and not db:
        db = _largest_domain(domains_b)
        if db:
            return (
                f"{da['start']}-{da['end']}",
                f"{db['start']}-{db['end']}",
                da["name"],
                db["name"],
                domain_type,
            )
    if db and not da:
        da = _largest_domain(domains_a)
        if da:
            return (
                f"{da['start']}-{da['end']}",
                f"{db['start']}-{db['end']}",
                da["name"],
                db["name"],
                domain_type,
            )

    # Fallback: shared domain name
    da, db = _find_shared_domain(domains_a, domains_b)
    if da and db:
        return (
            f"{da['start']}-{da['end']}",
            f"{db['start']}-{db['end']}",
            da["name"],
            db["name"],
            domain_type,
        )

    # Final fallback: largest domain of each
    da = _largest_domain(domains_a)
    db = _largest_domain(domains_b)
    if da and db:
        return (
            f"{da['start']}-{da['end']}",
            f"{db['start']}-{db['end']}",
            da["name"],
            db["name"],
            domain_type,
        )

    return "", "", "", "", domain_type


def _assess_match_quality(computed_pct: float, claim_str: str) -> str:
    """Compare computed domain identity with catalog claim percentage.

    Returns "match", "close", "mismatch", or "n/a".
    """
    if not claim_str:
        return "n/a"
    # Extract numeric values from claim like "~80%", "85-92%", "<30%"
    claim_str = claim_str.replace("\u2013", "-").replace("\u2014", "-")
    range_match = re.match(r"[~<>]?(\d+)\s*-\s*(\d+)%?", claim_str)
    single_match = re.match(r"[~<>]?(\d+)%?", claim_str)

    if range_match:
        lo, hi = float(range_match.group(1)), float(range_match.group(2))
        if lo <= computed_pct <= hi:
            return "match"
        if abs(computed_pct - (lo + hi) / 2) < 15:
            return "close"
        return "mismatch"
    if single_match:
        target = float(single_match.group(1))
        diff = abs(computed_pct - target)
        if diff < 8:
            return "match"
        if diff < 15:
            return "close"
        return "mismatch"
    return "n/a"
