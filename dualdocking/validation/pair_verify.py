"""Verify target pairs from the HTML catalog against external APIs.

Orchestrates PDB metadata, sequence identity, domain-specific identity,
ChEMBL bioactivity, and DOI resolution checks.  Core algorithms live in
submodules:

- :mod:`~dualdocking.validation.nw_align` -- Needleman-Wunsch alignment
- :mod:`~dualdocking.validation.pair_catalog` -- HTML catalog parser
- :mod:`~dualdocking.validation.domain_identity` -- domain selection & matching
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests
from loguru import logger

from dualdocking.data.http_utils import get_with_retry
from dualdocking.validation.domain_identity import (  # noqa: F401
    DomainIdentityReport,
    _assess_match_quality,
    _extract_catalog_pct,
    _parse_catalog_domain_type,
    select_domains_for_pair,
)
from dualdocking.validation.nw_align import (  # noqa: F401
    BLOSUM62,
    AlignmentResult,
    nw_align,
)
from dualdocking.validation.pair_catalog import (  # noqa: F401
    PairRecord,
    parse_catalog,
)

# ---------------------------------------------------------------------------
# Disk cache for API responses
# ---------------------------------------------------------------------------

_CACHE_DIR: Path | None = None


def set_cache_dir(path: Path | None) -> None:
    """Set the disk-cache directory for API responses.

    Pass *None* to disable caching.
    """
    global _CACHE_DIR
    _CACHE_DIR = path
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        logger.info("Verify cache dir: {}", path)


def _cache_key(prefix: str, *parts: str) -> str:
    """Build a safe filename from *prefix* and arbitrary string parts."""
    raw = "|".join(parts)
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", "_".join(parts))[:80]
    return f"{prefix}_{safe}_{h}.json"


def _cache_get(key: str) -> dict[str, Any] | None:
    if _CACHE_DIR is None:
        return None
    p = _CACHE_DIR / key
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            logger.debug("Cache hit: {}", key)
            return data
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _cache_put(key: str, data: dict[str, Any]) -> None:
    if _CACHE_DIR is None:
        return
    p = _CACHE_DIR / key
    tmp = p.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
    except OSError as exc:
        logger.warning("Cache write failed for {}: {}", key, exc)


# ---------------------------------------------------------------------------
# Dataclasses (verification reports)
# ---------------------------------------------------------------------------


@dataclass
class PdbReport:
    """Verification result for a single PDB entry."""

    pdb_id: str = ""
    resolution: float | None = None
    organism: str = ""
    gene_names: list[str] = field(default_factory=list)
    ligands: list[str] = field(default_factory=list)
    title: str = ""
    method: str = ""
    match: bool = True
    notes: list[str] = field(default_factory=list)


@dataclass
class SeqReport:
    """Sequence identity verification result."""

    alignment: AlignmentResult | None = None
    catalog_claim: str = ""
    match: bool = True
    notes: list[str] = field(default_factory=list)


@dataclass
class ChemblReport:
    """ChEMBL bioactivity summary for one target."""

    chembl_id: str = ""
    total_activities: int = 0
    ic50_count: int = 0
    ki_count: int = 0
    kd_count: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class DoiReport:
    """DOI resolution check."""

    doi: str = ""
    resolves: bool = False
    status_code: int = 0
    redirect_url: str = ""
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# UniProt Domain Fetching
# ---------------------------------------------------------------------------

_UNIPROT_JSON_URL = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"

# Manual UniProt corrections for pairs with wrong PDB auto-resolution
_UNIPROT_OVERRIDES: dict[int, dict[str, str]] = {
    2: {"uniprot_a": "P11802"},  # CDK4 (not CyclinD1 P24385); PDB 2W96 = CDK4+CyclinD3
    9: {"uniprot_b": "P37231"},  # PPARgamma (not RXRalpha P19793); PDB 1FM9 = PPARg+RXRa heterodimer
}


def fetch_uniprot_domains(uniprot_id: str) -> list[dict[str, Any]]:
    """Fetch Domain features from UniProt JSON API.

    Returns list of ``{"name": "Protein kinase", "start": 712, "end": 979}`` dicts.
    Results are disk-cached.
    """
    ckey = _cache_key("uniprot_domains", uniprot_id)
    cached = _cache_get(ckey)
    if cached is not None:
        logger.debug("[cache] UniProt domains for {} from cache", uniprot_id)
        return cached.get("domains", [])

    url = _UNIPROT_JSON_URL.format(uniprot_id=uniprot_id)
    resp = get_with_retry(url, timeout=20)
    data = resp.json()

    domains: list[dict[str, Any]] = []
    for feat in data.get("features", []):
        if feat.get("type") == "Domain":
            loc = feat.get("location", {})
            start_val = loc.get("start", {}).get("value")
            end_val = loc.get("end", {}).get("value")
            desc = feat.get("description", "")
            if start_val is not None and end_val is not None:
                domains.append({"name": desc, "start": start_val, "end": end_val})

    logger.info("UniProt {} domains: {}", uniprot_id, [d["name"] for d in domains])
    _cache_put(ckey, {"uniprot_id": uniprot_id, "domains": domains})
    return domains


# ---------------------------------------------------------------------------
# PDB Metadata Verification
# ---------------------------------------------------------------------------

_RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
_RCSB_POLYMER_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
_RCSB_NONPOLY_URL = "https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/{entity_id}"


def verify_pdb(pdb_id: str) -> PdbReport:
    """Verify PDB metadata against RCSB REST API.

    Queries entry endpoint for metadata, then fetches each polymer and
    nonpolymer entity individually.  Results are disk-cached by PDB ID.
    """
    pdb_id = pdb_id.upper()

    # Check cache
    ckey = _cache_key("pdb", pdb_id)
    cached = _cache_get(ckey)
    if cached is not None:
        logger.info("[cache] PDB {} from cache", pdb_id)
        return PdbReport(**cached)

    report = PdbReport(pdb_id=pdb_id)

    polymer_ids: list[str] = []
    nonpoly_ids: list[str] = []

    # Entry data
    try:
        resp = get_with_retry(_RCSB_ENTRY_URL.format(pdb_id=pdb_id), timeout=15)
        data = resp.json()
        info = data.get("rcsb_entry_info", {})
        report.resolution = info.get("resolution_combined", [None])[0]
        report.method = info.get("experimental_method", "")
        report.title = data.get("struct", {}).get("title", "")

        ids = data.get("rcsb_entry_container_identifiers", {})
        polymer_ids = ids.get("polymer_entity_ids", [])
        nonpoly_ids = ids.get("non_polymer_entity_ids", [])
    except Exception as exc:
        report.notes.append(f"Entry fetch failed: {exc}")
        report.match = False

    # Polymer entities (organism, gene)
    for eid in polymer_ids:
        try:
            url = _RCSB_POLYMER_URL.format(pdb_id=pdb_id, entity_id=eid)
            resp = get_with_retry(url, timeout=15)
            poly = resp.json()
            for s in poly.get("rcsb_entity_source_organism", []):
                org = s.get("ncbi_scientific_name", "")
                if org:
                    report.organism = org
                for g in s.get("rcsb_gene_name", []):
                    gname = g.get("value", "")
                    if gname and gname not in report.gene_names:
                        report.gene_names.append(gname)
        except Exception as exc:
            report.notes.append(f"Polymer entity {eid} fetch failed: {exc}")

    # Nonpolymer entities (ligands)
    for eid in nonpoly_ids:
        try:
            url = _RCSB_NONPOLY_URL.format(pdb_id=pdb_id, entity_id=eid)
            resp = get_with_retry(url, timeout=15)
            np_ent = resp.json()
            comp_id = np_ent.get("pdbx_entity_nonpoly", {}).get("comp_id", "")
            if comp_id:
                report.ligands.append(comp_id)
        except Exception as exc:
            report.notes.append(f"Nonpolymer entity {eid} fetch failed: {exc}")

    # Only cache successful results (no error notes)
    if not report.notes:
        _cache_put(ckey, asdict(report))

    return report


# ---------------------------------------------------------------------------
# Auto-resolution: PDB -> UniProt -> ChEMBL
# ---------------------------------------------------------------------------

_CHEMBL_TARGET_URL = "https://www.ebi.ac.uk/chembl/api/data/target.json"


def resolve_uniprot_from_pdb(pdb_id: str) -> list[str]:
    """Resolve UniProt accessions from a PDB entry via RCSB polymer entities.

    Queries the RCSB entry endpoint for polymer entity IDs, then each polymer
    entity for ``rcsb_polymer_entity_container_identifiers.uniprot_ids``.

    Returns:
        De-duplicated list of UniProt accessions (may be empty).
    """
    pdb_id = pdb_id.upper()
    ckey = _cache_key("uniprot_from_pdb", pdb_id)
    cached = _cache_get(ckey)
    if cached is not None:
        logger.info("[cache] UniProt for PDB {} from cache", pdb_id)
        return cached["uniprot_ids"]

    uniprot_ids: list[str] = []

    # Fetch entry to get polymer entity IDs
    resp = get_with_retry(_RCSB_ENTRY_URL.format(pdb_id=pdb_id), timeout=15)
    data = resp.json()
    polymer_ids = data.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])

    for eid in polymer_ids:
        url = _RCSB_POLYMER_URL.format(pdb_id=pdb_id, entity_id=eid)
        resp = get_with_retry(url, timeout=15)
        poly = resp.json()
        uids = poly.get("rcsb_polymer_entity_container_identifiers", {}).get("uniprot_ids", [])
        for uid in uids:
            if uid not in uniprot_ids:
                uniprot_ids.append(uid)

    _cache_put(ckey, {"pdb_id": pdb_id, "uniprot_ids": uniprot_ids})
    return uniprot_ids


def resolve_chembl_from_uniprot(uniprot_id: str) -> str | None:
    """Resolve a ChEMBL target ID from a UniProt accession.

    Queries the ChEMBL target endpoint filtered by ``target_accession``.

    Returns:
        ChEMBL target ID (e.g. ``"CHEMBL203"``) or *None* if not found.
    """
    ckey = _cache_key("chembl_from_uniprot", uniprot_id)
    cached = _cache_get(ckey)
    if cached is not None:
        logger.info("[cache] ChEMBL for UniProt {} from cache", uniprot_id)
        return cached.get("chembl_id")

    url = f"{_CHEMBL_TARGET_URL}?target_components__accession={uniprot_id}&target_type=SINGLE+PROTEIN&limit=1&format=json"
    resp = get_with_retry(url, timeout=30)
    data = resp.json()
    targets = data.get("targets", [])
    chembl_id: str | None = None
    if targets:
        chembl_id = targets[0].get("target_chembl_id")

    _cache_put(ckey, {"uniprot_id": uniprot_id, "chembl_id": chembl_id})
    return chembl_id


# ---------------------------------------------------------------------------
# Sequence Identity Verification
# ---------------------------------------------------------------------------

_UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"


def _fetch_uniprot_fasta(uniprot_id: str) -> str:
    """Fetch a protein sequence from UniProt and return the raw AA string.

    Results are disk-cached by UniProt ID (sequences are stable).
    """
    ckey = _cache_key("fasta", uniprot_id)
    cached = _cache_get(ckey)
    if cached is not None:
        logger.debug("[cache] FASTA {} from cache", uniprot_id)
        return cached["sequence"]

    resp = get_with_retry(_UNIPROT_FASTA_URL.format(uniprot_id=uniprot_id), timeout=20)
    lines = resp.text.strip().splitlines()
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    _cache_put(ckey, {"uniprot_id": uniprot_id, "sequence": seq})
    return seq


def _parse_domain(domain_str: str) -> tuple[int, int] | None:
    """Parse domain range like '671-998' into 1-based (start, end)."""
    m = re.match(r"(\d+)\s*-\s*(\d+)", domain_str.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def verify_sequence_identity(
    uniprot_a: str,
    uniprot_b: str,
    *,
    domain_a: str = "",
    domain_b: str = "",
    catalog_claim: str = "",
) -> SeqReport:
    """Verify sequence identity between two UniProt sequences.

    Fetches FASTA from UniProt, optionally slices to domain ranges,
    then runs Needleman-Wunsch alignment.
    """
    report = SeqReport(catalog_claim=catalog_claim)

    try:
        seq_a = _fetch_uniprot_fasta(uniprot_a)
        seq_b = _fetch_uniprot_fasta(uniprot_b)
    except Exception as exc:
        report.notes.append(f"FASTA fetch failed: {exc}")
        report.match = False
        return report

    # Slice to domains if specified (1-based inclusive)
    if domain_a:
        rng = _parse_domain(domain_a)
        if rng:
            seq_a = seq_a[rng[0] - 1 : rng[1]]
            report.notes.append(f"Sliced {uniprot_a} to {domain_a} ({len(seq_a)} res)")

    if domain_b:
        rng = _parse_domain(domain_b)
        if rng:
            seq_b = seq_b[rng[0] - 1 : rng[1]]
            report.notes.append(f"Sliced {uniprot_b} to {domain_b} ({len(seq_b)} res)")

    aln = nw_align(seq_a, seq_b)
    report.alignment = aln
    logger.info(
        "NW alignment {}/{}: {:.1f}% identity ({} matches / {} aligned, {} gaps)",
        uniprot_a,
        uniprot_b,
        aln.identity_pct,
        aln.matches,
        aln.aligned_length,
        aln.gaps,
    )

    return report


# ---------------------------------------------------------------------------
# ChEMBL Bioactivity Verification
# ---------------------------------------------------------------------------

_CHEMBL_ACTIVITY_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"


def verify_chembl(chembl_id: str) -> ChemblReport:
    """Query ChEMBL for bioactivity summary of a target.

    Results are disk-cached by ChEMBL target ID.

    Args:
        chembl_id: ChEMBL target ID (e.g. ``"CHEMBL203"``).
    """
    ckey = _cache_key("chembl", chembl_id)
    cached = _cache_get(ckey)
    if cached is not None:
        logger.info("[cache] ChEMBL {} from cache", chembl_id)
        return ChemblReport(**cached)

    report = ChemblReport(chembl_id=chembl_id)

    for assay_type in ("IC50", "Ki", "Kd"):
        try:
            url = f"{_CHEMBL_ACTIVITY_URL}?target_chembl_id={chembl_id}&standard_type={assay_type}&limit=0&format=json"
            resp = get_with_retry(url, timeout=30)
            data = resp.json()
            total = data.get("page_meta", {}).get("total_count", 0)
            if assay_type == "IC50":
                report.ic50_count = total
            elif assay_type == "Ki":
                report.ki_count = total
            else:
                report.kd_count = total
            report.total_activities += total
        except Exception as exc:
            report.notes.append(f"ChEMBL {assay_type} query failed: {exc}")

    if not report.notes:
        _cache_put(ckey, asdict(report))

    return report


# ---------------------------------------------------------------------------
# DOI Resolution Verification
# ---------------------------------------------------------------------------


def verify_doi(doi: str) -> DoiReport:
    """Check if a DOI resolves via doi.org.

    Uses HTTP HEAD with no redirects to check status.
    Results are disk-cached by DOI.
    """
    ckey = _cache_key("doi", doi)
    cached = _cache_get(ckey)
    if cached is not None:
        logger.info("[cache] DOI {} from cache", doi)
        return DoiReport(**cached)

    report = DoiReport(doi=doi)
    url = f"https://doi.org/{doi}" if not doi.startswith("http") else doi

    try:
        resp = requests.head(url, allow_redirects=False, timeout=15)
        report.status_code = resp.status_code
        report.resolves = resp.status_code in (200, 301, 302, 303, 307, 308)
        report.redirect_url = resp.headers.get("Location", "")
    except Exception as exc:
        report.notes.append(f"DOI resolution failed: {exc}")

    if not report.notes:
        _cache_put(ckey, asdict(report))

    return report


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def verify_pair(  # noqa: PLR0913
    pair: PairRecord,
    *,
    do_pdb: bool = False,
    do_doi: bool = False,
    do_sequence: bool = False,
    do_chembl: bool = False,
    uniprot_a: str = "",
    uniprot_b: str = "",
    domain_a: str = "",
    domain_b: str = "",
    chembl_a: str = "",
    chembl_b: str = "",
) -> dict[str, Any]:
    """Run all requested verifications for a single pair.

    When *do_sequence* or *do_chembl* is True and the corresponding UniProt /
    ChEMBL IDs are not supplied, they are auto-resolved from the PDB entries:

    ``PDB -> RCSB polymer entity -> UniProt -> ChEMBL target``

    Returns:
        JSON-serializable dict with pair data and verification results.
    """
    result: dict[str, Any] = asdict(pair)
    verification: dict[str, Any] = {}
    resolved: dict[str, str] = {}

    # --- Auto-resolve UniProt IDs from PDB if needed ---
    if (do_sequence or do_chembl) and not uniprot_a and pair.pdb_a:
        try:
            uids = resolve_uniprot_from_pdb(pair.pdb_a)
            if uids:
                uniprot_a = uids[0]
                logger.info("Auto-resolved UniProt A: {} -> {}", pair.pdb_a, uniprot_a)
                resolved["uniprot_a"] = uniprot_a
        except Exception as exc:
            logger.warning("Failed to resolve UniProt for PDB {}: {}", pair.pdb_a, exc)

    if (do_sequence or do_chembl) and not uniprot_b and pair.pdb_b:
        try:
            uids = resolve_uniprot_from_pdb(pair.pdb_b)
            if uids:
                uniprot_b = uids[0]
                logger.info("Auto-resolved UniProt B: {} -> {}", pair.pdb_b, uniprot_b)
                resolved["uniprot_b"] = uniprot_b
        except Exception as exc:
            logger.warning("Failed to resolve UniProt for PDB {}: {}", pair.pdb_b, exc)

    # --- Apply manual UniProt overrides for known bad auto-resolutions ---
    overrides = _UNIPROT_OVERRIDES.get(pair.pair_number, {})
    if overrides:
        if "uniprot_a" in overrides:
            old_a = uniprot_a
            uniprot_a = overrides["uniprot_a"]
            resolved["uniprot_a"] = uniprot_a
            logger.info("Override UniProt A: {} -> {} (pair #{})", old_a, uniprot_a, pair.pair_number)
        if "uniprot_b" in overrides:
            old_b = uniprot_b
            uniprot_b = overrides["uniprot_b"]
            resolved["uniprot_b"] = uniprot_b
            logger.info("Override UniProt B: {} -> {} (pair #{})", old_b, uniprot_b, pair.pair_number)

    # --- Auto-resolve ChEMBL IDs from UniProt if needed ---
    if do_chembl and not chembl_a and uniprot_a:
        try:
            cid = resolve_chembl_from_uniprot(uniprot_a)
            if cid:
                chembl_a = cid
                logger.info("Auto-resolved ChEMBL A: {} -> {}", uniprot_a, chembl_a)
                resolved["chembl_a"] = chembl_a
        except Exception as exc:
            logger.warning("Failed to resolve ChEMBL for UniProt {}: {}", uniprot_a, exc)

    if do_chembl and not chembl_b and uniprot_b:
        try:
            cid = resolve_chembl_from_uniprot(uniprot_b)
            if cid:
                chembl_b = cid
                logger.info("Auto-resolved ChEMBL B: {} -> {}", uniprot_b, chembl_b)
                resolved["chembl_b"] = chembl_b
        except Exception as exc:
            logger.warning("Failed to resolve ChEMBL for UniProt {}: {}", uniprot_b, exc)

    if resolved:
        result["resolved_ids"] = resolved

    # --- Verification checks ---
    if do_pdb:
        verification["pdb_a"] = asdict(verify_pdb(pair.pdb_a))
        verification["pdb_b"] = asdict(verify_pdb(pair.pdb_b))

    if (do_sequence or (uniprot_a and uniprot_b)) and uniprot_a and uniprot_b:
        # Full-length NW identity (always without domain slicing)
        seq_report = verify_sequence_identity(
            uniprot_a,
            uniprot_b,
            domain_a=domain_a,
            domain_b=domain_b,
            catalog_claim=pair.similarity_pct,
        )
        verification["sequence_identity"] = asdict(seq_report)

        # Domain-specific NW identity
        range_a, range_b, dname_a, dname_b, dtype = select_domains_for_pair(
            pair,
            uniprot_a,
            uniprot_b,
        )
        if range_a and range_b:
            dom_seq = verify_sequence_identity(
                uniprot_a,
                uniprot_b,
                domain_a=range_a,
                domain_b=range_b,
                catalog_claim=pair.similarity_pct,
            )
            claim_pct = _extract_catalog_pct(pair.similarity_pct)
            match_quality = _assess_match_quality(
                dom_seq.alignment.identity_pct if dom_seq.alignment else 0.0,
                claim_pct,
            )
            dom_report = DomainIdentityReport(
                alignment=dom_seq.alignment,
                domain_a=f"{dname_a} ({range_a})" if dname_a else range_a,
                domain_b=f"{dname_b} ({range_b})" if dname_b else range_b,
                range_a=range_a,
                range_b=range_b,
                domain_type=dtype,
                catalog_claim_pct=claim_pct,
                match_quality=match_quality,
                notes=dom_seq.notes,
            )
            verification["domain_identity"] = asdict(dom_report)
        else:
            # No domain alignment applicable -- record why
            verification["domain_identity"] = asdict(
                DomainIdentityReport(
                    domain_type=dtype,
                    match_quality="n/a",
                    notes=[f"Domain alignment not applicable (type={dtype})"],
                )
            )

    if (do_chembl or chembl_a) and chembl_a:
        verification["chembl_a"] = asdict(verify_chembl(chembl_a))
    if (do_chembl or chembl_b) and chembl_b:
        verification["chembl_b"] = asdict(verify_chembl(chembl_b))

    if do_doi and pair.doi:
        verification["doi"] = asdict(verify_doi(pair.doi))

    if verification:
        result["verification"] = verification

    return result
