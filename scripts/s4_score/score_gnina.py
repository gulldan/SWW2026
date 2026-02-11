"""Score generated molecules with GNINA as cross-validation scorer.

Reads ``scored_joined.parquet`` (from the Boltz-2 pipeline), runs GNINA
CNN docking for both targets, and writes ``gnina_scores.parquet``.

Usage:
    uv run python scripts/s4_score/score_gnina.py --pair CDK2_ERK2 --mode double --seed 42
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_binding_box(
    data_dir: Path,
    target_id: str,
) -> tuple[Path | None, tuple[float, float, float] | None, tuple[float, float, float]]:
    """Load binding box for a target.

    Tries autobox from co-crystallized ligand first, then falls back to
    binding_box.extract_binding_box for center/size from the receptor PDB.

    Returns:
        (autobox_ligand_path_or_None, center_or_None, box_size_tuple)
    """
    from dualdocking.scoring.binding_box import extract_binding_box

    autobox = data_dir / target_id / f"{target_id}_ligand.pdb"
    if autobox.exists():
        return autobox, None, (25.0, 25.0, 25.0)

    # Fall back to extracting box from receptor PDB
    receptor = data_dir / target_id / f"{target_id}_receptor.pdb"
    center_str, size_str = extract_binding_box(receptor)
    center = tuple(float(v) for v in center_str.split(","))
    size = tuple(float(v) for v in size_str.split(","))
    return None, center, size  # type: ignore[return-value]


@click.command()
@click.option("--pair", required=True, help="Pair ID from configs/pairs.yaml")
@click.option("--mode", type=click.Choice(["single_a", "single_b", "double"]), required=True)
@click.option("--seed", default=42, type=int)
@click.option("--config", default="configs/pairs.yaml")
@click.option("--force", is_flag=True, default=False, help="Re-score even if gnina_scores already exist")
@click.option("--docker-image", default="gnina/gnina:v1.1", help="GNINA Docker image")
@click.option("--timeout", default=300, type=int, help="Timeout per batch in seconds")
@click.option("--exhaustiveness", default=8, type=int, help="GNINA search exhaustiveness")
def main(
    pair: str,
    mode: str,
    seed: int,
    config: str,
    force: bool,
    docker_image: str,
    timeout: int,
    exhaustiveness: int,
) -> None:
    """Dock and score molecules against both targets using GNINA CNN scorer."""
    from dualdocking.config import load_pairs
    from dualdocking.data.io_utils import read_df, stem_exists, write_df
    from dualdocking.scoring.gnina import score_batch

    pairs = load_pairs(PROJECT_ROOT / config)["pairs"]
    if pair not in pairs:
        logger.error("Pair '{}' not found in {}", pair, config)
        sys.exit(1)

    pair_cfg = pairs[pair]
    result_dir = PROJECT_ROOT / "output" / "results" / pair / mode / f"seed_{seed}"

    # Check for existing output
    gnina_stem = result_dir / "gnina_scores"
    if not force and stem_exists(gnina_stem):
        logger.info("GNINA scores already exist at {}, skipping (use --force to re-score)", gnina_stem)
        return

    # Load scored molecules
    joined_stem = result_dir / "scored_joined"
    if not stem_exists(joined_stem):
        logger.error("scored_joined not found at {}", joined_stem)
        sys.exit(1)
    joined_df = read_df(joined_stem)
    smiles_list: list[str] = joined_df["smiles"].tolist()
    logger.info("Loaded {} molecules from {}", len(smiles_list), joined_stem)

    target_a_id = pair_cfg["target_a"]["id"]
    target_b_id = pair_cfg["target_b"]["id"]
    data_dir = PROJECT_ROOT / "output" / "data" / "targets"

    # Score target A
    receptor_a = data_dir / target_a_id / f"{target_a_id}_receptor.pdb"
    if not receptor_a.exists():
        logger.error("Receptor PDB not found: {}", receptor_a)
        sys.exit(1)

    autobox_a, center_a, size_a = _load_binding_box(data_dir, target_a_id)
    logger.info("Docking {} molecules against {} with GNINA", len(smiles_list), target_a_id)
    results_a = score_batch(
        smiles_list,
        receptor_a,
        autobox_ligand=autobox_a,
        center=center_a,
        box_size=size_a,
        exhaustiveness=exhaustiveness,
        docker_image=docker_image,
        timeout=timeout,
        seed=seed,
    )

    # Score target B
    receptor_b = data_dir / target_b_id / f"{target_b_id}_receptor.pdb"
    if not receptor_b.exists():
        logger.error("Receptor PDB not found: {}", receptor_b)
        sys.exit(1)

    autobox_b, center_b, size_b = _load_binding_box(data_dir, target_b_id)
    logger.info("Docking {} molecules against {} with GNINA", len(smiles_list), target_b_id)
    results_b = score_batch(
        smiles_list,
        receptor_b,
        autobox_ligand=autobox_b,
        center=center_b,
        box_size=size_b,
        exhaustiveness=exhaustiveness,
        docker_image=docker_image,
        timeout=timeout,
        seed=seed,
    )

    # Build output dataframe
    gnina_df = pd.DataFrame(
        {
            "smiles": smiles_list,
            "gnina_cnn_score_a": [r.cnn_score for r in results_a],
            "gnina_cnn_affinity_a": [r.cnn_affinity for r in results_a],
            "gnina_vina_score_a": [r.vina_score for r in results_a],
            "gnina_cnn_score_b": [r.cnn_score for r in results_b],
            "gnina_cnn_affinity_b": [r.cnn_affinity for r in results_b],
            "gnina_vina_score_b": [r.vina_score for r in results_b],
            "gnina_valid_a": [r.valid for r in results_a],
            "gnina_valid_b": [r.valid for r in results_b],
        }
    )

    write_df(gnina_df, gnina_stem, csv_too=True)
    logger.info("GNINA scores saved to {}", gnina_stem)

    valid_a = np.array([r.valid for r in results_a])
    valid_b = np.array([r.valid for r in results_b])
    aff_a = np.array([r.cnn_affinity for r in results_a])
    aff_b = np.array([r.cnn_affinity for r in results_b])
    logger.info(
        "GNINA stats - A: {}/{} valid, mean_cnn_affinity={:.2f} | B: {}/{} valid, mean_cnn_affinity={:.2f}",
        valid_a.sum(),
        len(valid_a),
        np.nanmean(aff_a),
        valid_b.sum(),
        len(valid_b),
        np.nanmean(aff_b),
    )


if __name__ == "__main__":
    main()
