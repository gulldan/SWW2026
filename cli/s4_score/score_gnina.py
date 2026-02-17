"""Score generated molecules with GNINA as cross-validation scorer.

Reads ``scored_joined.parquet`` (from the Boltz-2 pipeline), runs GNINA
CNN docking for both targets, and writes ``gnina_scores.parquet``.

Usage:
    uv run python cli/s4_score/score_gnina.py --pair CDK2_ERK2 --mode double --seed 42
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@click.command()
@click.option("--pair", required=True, help="Pair ID from configs/pairs.yaml")
@click.option("--mode", type=str, required=True, help="Generation mode folder name (e.g. single_a, double)")
@click.option("--seed", default=42, type=int)
@click.option("--config", default="configs/pairs.yaml")
@click.option("--force", is_flag=True, default=False, help="Re-score even if gnina_scores already exist")
@click.option("--docker-image", default="gnina/gnina:v1.3.2", help="GNINA Docker image (ignored if GNINA_BINARY is set)")
@click.option("--timeout", default=300, type=int, help="Min timeout per sub-batch in seconds")
@click.option("--exhaustiveness", default=8, type=int, help="GNINA search exhaustiveness")
@click.option("--sub-batch-size", default=50, type=int, help="Molecules per GNINA invocation")
def main(
    pair: str,
    mode: str,
    seed: int,
    config: str,
    force: bool,
    docker_image: str,
    timeout: int,
    exhaustiveness: int,
    sub_batch_size: int,
) -> None:
    """Dock and score molecules against both targets using GNINA CNN scorer."""
    from dualdocking.config import load_pairs
    from dualdocking.data.io_utils import read_df, stem_exists, write_df
    from dualdocking.scoring.binding_box import extract_binding_box
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
    pdb_id_a = pair_cfg["target_a"]["pdb_id"].lower()
    pdb_id_b = pair_cfg["target_b"]["pdb_id"].lower()
    data_dir = PROJECT_ROOT / "output" / "data"
    receptors_dir = data_dir / "calibration" / "receptors"
    targets_dir = data_dir / "targets"

    def _resolve_receptor_and_box(
        target_id: str, pdb_id: str, side: str,
    ) -> tuple[Path, tuple[float, float, float] | None, tuple[float, float, float]]:
        """Resolve receptor PDB and binding box from actual data layout."""
        receptor = receptors_dir / f"{target_id}.fixed.pdb"
        if not receptor.exists():
            logger.error("Receptor PDB not found: {}", receptor)
            sys.exit(1)

        # Extract binding box from the raw PDB (has co-crystallized ligand)
        raw_pdb = targets_dir / pair / side / f"{pdb_id}.pdb"
        if raw_pdb.exists():
            center_str, size_str = extract_binding_box(raw_pdb)
        else:
            logger.warning("Raw PDB {} not found, extracting box from fixed receptor", raw_pdb)
            center_str, size_str = extract_binding_box(receptor)

        center = tuple(float(v) for v in center_str.split(","))
        size = tuple(float(v) for v in size_str.split(","))
        return receptor, center, size  # type: ignore[return-value]

    # Score target A
    receptor_a, center_a, size_a = _resolve_receptor_and_box(target_a_id, pdb_id_a, "A")
    logger.info("Docking {} molecules against {} with GNINA", len(smiles_list), target_a_id)
    results_a = score_batch(
        smiles_list,
        receptor_a,
        center=center_a,
        box_size=size_a,
        exhaustiveness=exhaustiveness,
        docker_image=docker_image,
        timeout=timeout,
        seed=seed,
        sub_batch_size=sub_batch_size,
    )

    # Score target B
    receptor_b, center_b, size_b = _resolve_receptor_and_box(target_b_id, pdb_id_b, "B")
    logger.info("Docking {} molecules against {} with GNINA", len(smiles_list), target_b_id)
    results_b = score_batch(
        smiles_list,
        receptor_b,
        center=center_b,
        box_size=size_b,
        exhaustiveness=exhaustiveness,
        docker_image=docker_image,
        timeout=timeout,
        seed=seed,
        sub_batch_size=sub_batch_size,
    )

    # Build output dataframe
    valid_a = [r.valid for r in results_a]
    valid_b = [r.valid for r in results_b]
    aff_a = [r.cnn_affinity for r in results_a]
    aff_b = [r.cnn_affinity for r in results_b]

    gnina_df = pd.DataFrame(
        {
            "smiles": smiles_list,
            "gnina_cnn_score_a": [r.cnn_score for r in results_a],
            "gnina_cnn_affinity_a": aff_a,
            "gnina_vina_score_a": [r.vina_score for r in results_a],
            "gnina_cnn_score_b": [r.cnn_score for r in results_b],
            "gnina_cnn_affinity_b": aff_b,
            "gnina_vina_score_b": [r.vina_score for r in results_b],
            "gnina_valid_a": valid_a,
            "gnina_valid_b": valid_b,
        }
    )

    write_df(gnina_df, gnina_stem, csv_too=True)
    logger.info("GNINA scores saved to {}", gnina_stem)

    n_valid_a = sum(valid_a)
    n_valid_b = sum(valid_b)
    logger.info(
        "GNINA stats - A: {}/{} valid, mean_cnn_affinity={:.2f} | B: {}/{} valid, mean_cnn_affinity={:.2f}",
        n_valid_a,
        len(valid_a),
        np.nanmean(aff_a),
        n_valid_b,
        len(valid_b),
        np.nanmean(aff_b),
    )


if __name__ == "__main__":
    main()
