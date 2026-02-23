"""Verify target pair(s) from the HTML catalog against external APIs.

Thin CLI wrapper around :mod:`dualdocking.validation.pair_verify`.

Examples::

    uv run python cli/s7_audit/verify_pair.py --pair 1 --catalog target_pairs_catalog.html
    uv run python cli/s7_audit/verify_pair.py --pair "EGFR/HER2" --verify-doi
    uv run python cli/s7_audit/verify_pair.py --all --verify-doi --output-json report/data/verify_all.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@click.command()
@click.option("--pair", default="", help="Pair number (int) or name (e.g. 'EGFR/HER2').")
@click.option("--all", "all_pairs", is_flag=True, help="Verify all 50 pairs.")
@click.option("--catalog", default="target_pairs_catalog.html", help="Path to HTML catalog.")
@click.option("--uniprot-a", default="", help="UniProt ID for target A (sequence identity).")
@click.option("--uniprot-b", default="", help="UniProt ID for target B (sequence identity).")
@click.option("--domain-a", default="", help="Domain range for target A (e.g. '671-998').")
@click.option("--domain-b", default="", help="Domain range for target B (e.g. '706-1009').")
@click.option("--chembl-a", default="", help="ChEMBL target ID for target A.")
@click.option("--chembl-b", default="", help="ChEMBL target ID for target B.")
@click.option("--verify-pdb/--no-verify-pdb", default=False, help="Verify PDB metadata via RCSB.")
@click.option("--verify-doi/--no-verify-doi", default=False, help="Verify DOI resolution.")
@click.option("--verify-sequence/--no-verify-sequence", default=False, help="Verify sequence identity (auto-resolves UniProt).")
@click.option("--verify-chembl/--no-verify-chembl", default=False, help="Verify ChEMBL bioactivity (auto-resolves IDs).")
@click.option("--full", "full_check", is_flag=True, help="Enable all checks (pdb+doi+sequence+chembl).")
@click.option("--output-json", default="", help="Write JSON report to this path.")
@click.option("--cache-dir", default="output/cache/verify", help="Disk cache directory for API responses.")
@click.option("--no-cache", is_flag=True, help="Disable disk cache (re-fetch everything).")
def main(  # noqa: PLR0913
    pair: str,
    all_pairs: bool,
    catalog: str,
    uniprot_a: str,
    uniprot_b: str,
    domain_a: str,
    domain_b: str,
    chembl_a: str,
    chembl_b: str,
    verify_pdb: bool,
    verify_doi: bool,
    verify_sequence: bool,
    verify_chembl: bool,
    full_check: bool,
    output_json: str,
    cache_dir: str,
    no_cache: bool,
) -> None:
    """Parse & verify target pair(s) from the HTML catalog."""
    from dualdocking.validation.pair_verify import parse_catalog, set_cache_dir, verify_pair

    # --full enables all checks
    if full_check:
        verify_pdb = True
        verify_doi = True
        verify_sequence = True
        verify_chembl = True

    # Set up disk cache
    if no_cache:
        set_cache_dir(None)
    else:
        cache_path = PROJECT_ROOT / cache_dir if not Path(cache_dir).is_absolute() else Path(cache_dir)
        set_cache_dir(cache_path)

    catalog_path = PROJECT_ROOT / catalog if not Path(catalog).is_absolute() else Path(catalog)
    if not catalog_path.exists():
        logger.error("Catalog not found: {}", catalog_path)
        sys.exit(1)

    pairs = parse_catalog(catalog_path)

    if not pair and not all_pairs:
        logger.error("Specify --pair <number|name> or --all")
        sys.exit(1)

    # Select pair(s)
    selected: list = []
    if all_pairs:
        selected = pairs
    elif pair.isdigit():
        num = int(pair)
        selected = [p for p in pairs if p.pair_number == num]
        if not selected:
            logger.error("Pair #{} not found (have {} pairs)", num, len(pairs))
            sys.exit(1)
    else:
        # Match by name (case-insensitive, flexible separator)
        query = pair.lower().replace("/", "").replace(" ", "").replace("_", "")
        for p in pairs:
            key = f"{p.target_a}{p.target_b}".lower().replace("/", "").replace(" ", "").replace("_", "")
            key_rev = f"{p.target_b}{p.target_a}".lower().replace("/", "").replace(" ", "").replace("_", "")
            if query in (key, key_rev):
                selected.append(p)
                break
        if not selected:
            logger.error("Pair '{}' not found", pair)
            sys.exit(1)

    # Run verification
    results = []
    for p in selected:
        logger.info("--- Pair #{}: {} / {} ---", p.pair_number, p.target_a, p.target_b)
        result = verify_pair(
            p,
            do_pdb=verify_pdb,
            do_doi=verify_doi,
            do_sequence=verify_sequence,
            do_chembl=verify_chembl,
            uniprot_a=uniprot_a,
            uniprot_b=uniprot_b,
            domain_a=domain_a,
            domain_b=domain_b,
            chembl_a=chembl_a,
            chembl_b=chembl_b,
        )
        results.append(result)

        # Print summary
        click.echo(f"Pair #{p.pair_number}: {p.target_a} / {p.target_b}")
        click.echo(f"  Category: {p.category}")
        click.echo(f"  Family:   {p.target_family}")
        click.echo(f"  PDB A:    {p.pdb_a} ({p.pdb_a_desc})")
        click.echo(f"  PDB B:    {p.pdb_b} ({p.pdb_b_desc})")
        click.echo(f"  DOI:      {p.doi}")
        click.echo(f"  Similarity: {p.similarity_level} — {p.similarity_pct}")
        if p.drug_status:
            click.echo(f"  Drug:     {p.drug_status}")

        resolved = result.get("resolved_ids", {})
        if resolved:
            parts = [f"{k}={v}" for k, v in resolved.items()]
            click.echo(f"  Resolved: {', '.join(parts)}")

        v = result.get("verification", {})
        if "pdb_a" in v:
            pa = v["pdb_a"]
            click.echo(f"  [PDB-A] {pa['pdb_id']}: res={pa['resolution']}, org={pa['organism']}, ligands={pa['ligands']}")
        if "pdb_b" in v:
            pb = v["pdb_b"]
            click.echo(f"  [PDB-B] {pb['pdb_id']}: res={pb['resolution']}, org={pb['organism']}, ligands={pb['ligands']}")
        if "sequence_identity" in v:
            si = v["sequence_identity"]
            aln = si.get("alignment") or {}
            click.echo(f"  [SeqID] {aln.get('identity_pct', '?')}% ({aln.get('matches', '?')}/{aln.get('aligned_length', '?')})")
        if "domain_identity" in v:
            di = v["domain_identity"]
            di_aln = di.get("alignment") or {}
            di_pct = di_aln.get("identity_pct")
            if di_pct is not None:
                da_str = di.get("domain_a", "")
                db_str = di.get("domain_b", "")
                claim = di.get("catalog_claim_pct", "")
                quality = di.get("match_quality", "")
                quality_sym = {"match": "✓", "close": "≈", "mismatch": "✗"}.get(quality, "?")
                click.echo(f"  [DomainID] {di_pct}% {da_str} / {db_str} vs catalog {claim} — {quality_sym} {quality}")
            else:
                dtype = di.get("domain_type", "")
                notes = di.get("notes", [])
                click.echo(f"  [DomainID] n/a (type={dtype}) {'; '.join(notes)}")
        if "doi" in v:
            d = v["doi"]
            click.echo(f"  [DOI] resolves={d['resolves']} status={d['status_code']}")
        if "chembl_a" in v:
            ca = v["chembl_a"]
            click.echo(f"  [ChEMBL-A] {ca['chembl_id']}: IC50={ca['ic50_count']}, Ki={ca['ki_count']}, Kd={ca['kd_count']}")
        if "chembl_b" in v:
            cb = v["chembl_b"]
            click.echo(f"  [ChEMBL-B] {cb['chembl_id']}: IC50={cb['ic50_count']}, Ki={cb['ki_count']}, Kd={cb['kd_count']}")
        click.echo()

    # Output JSON
    output = results[0] if len(results) == 1 else results
    if output_json:
        out_path = PROJECT_ROOT / output_json if not Path(output_json).is_absolute() else Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        logger.info("JSON report -> {}", out_path)


if __name__ == "__main__":
    main()
