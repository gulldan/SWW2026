"""Fetch ChEMBL actives for all targets in pairs.yaml.

Downloads per-target actives and computes per-pair dual-active sets.
Results are cached in data/chembl/ for use in Stage 2 validation.

Usage::

    uv run python scripts/fetch_chembl_actives.py
    uv run python scripts/fetch_chembl_actives.py --target-id BACE1 --target-id GSK3B
    uv run python scripts/fetch_chembl_actives.py --min-pchembl 6.0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CHEMBL_DATA_DIR = PROJECT_ROOT / "data" / "chembl"


@click.command()
@click.option("--config", default="configs/pairs.yaml", help="Path to pairs config")
@click.option("--min-pchembl", default=5.0, type=float, help="Minimum pChEMBL value")
@click.option("--target-id", multiple=True, help="Specific target IDs (repeatable)")
@click.option("--output-dir", default=str(CHEMBL_DATA_DIR), help="Output directory")
@click.option("--force/--no-force", default=False, help="Re-download even if cached")
def main(
    config: str,
    min_pchembl: float,
    target_id: tuple[str, ...],
    output_dir: str,
    force: bool,
) -> None:
    """Fetch ChEMBL actives for all targets and compute dual-active overlaps."""
    import yaml

    from dualdocking.chembl_validation import (
        TARGET_CHEMBL_MAP,
        fetch_actives,
        find_dual_actives,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if force:
        # Remove cached files to force re-download
        for f in out.glob("*_actives.json"):
            f.unlink()

    config_path = PROJECT_ROOT / config
    pairs = yaml.safe_load(config_path.read_text())["pairs"]

    # Collect unique target IDs
    all_targets: set[str] = set()
    for pair_cfg in pairs.values():
        all_targets.add(pair_cfg["target_a"]["id"])
        all_targets.add(pair_cfg["target_b"]["id"])

    if target_id:
        all_targets = all_targets & set(target_id)

    # Filter to targets with ChEMBL mapping
    fetchable = sorted(t for t in all_targets if t in TARGET_CHEMBL_MAP)
    skipped = sorted(all_targets - set(fetchable))
    if skipped:
        logger.warning("No ChEMBL mapping for targets: {}", skipped)

    # Fetch per-target actives
    target_actives: dict[str, list[str]] = {}
    for tid in fetchable:
        logger.info("Fetching actives for {} (pChEMBL >= {})...", tid, min_pchembl)
        actives = fetch_actives(tid, min_pchembl=min_pchembl, cache_dir=out)
        target_actives[tid] = actives
        logger.info("  {} -> {} unique actives", tid, len(actives))

    # Compute dual-active overlaps for each pair
    summary: dict[str, dict] = {}
    for pair_id, pair_cfg in pairs.items():
        ta = pair_cfg["target_a"]["id"]
        tb = pair_cfg["target_b"]["id"]

        if ta not in target_actives or tb not in target_actives:
            continue

        dual = find_dual_actives(target_actives[ta], target_actives[tb])
        dual_path = out / f"{pair_id}_dual_actives.json"
        dual_path.write_text(json.dumps(dual))

        summary[pair_id] = {
            "target_a": ta,
            "target_b": tb,
            "actives_a": len(target_actives[ta]),
            "actives_b": len(target_actives[tb]),
            "dual_actives": len(dual),
        }
        logger.info(
            "  {} ({}/{}) -> {} dual actives",
            pair_id,
            ta,
            tb,
            len(dual),
        )

    # Write summary
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary written to {}", summary_path)

    # Print summary table
    click.echo("\n--- ChEMBL Actives Summary ---")
    click.echo(f"{'Pair':<20} {'A':>8} {'B':>8} {'Dual':>8}")
    click.echo("-" * 48)
    for pair_id, info in sorted(summary.items()):
        click.echo(
            f"{pair_id:<20} {info['actives_a']:>8} {info['actives_b']:>8} {info['dual_actives']:>8}"
        )


if __name__ == "__main__":
    main()
