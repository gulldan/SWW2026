"""2D molecule visualisation with score annotations.

Reads ``report/data/all_results.parquet``, renders a grid of RDKit 2D
molecule images for each pair/mode with readable score labels and
colour-coded hit markers.

This is a thin CLI wrapper around :mod:`dualdocking.reporting.plots.molecules`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
from loguru import logger

from dualdocking.reporting.plots.molecules import (
    MoleculeRenderConfig,
    is_hit,
    load_target_names,
    load_thresholds,
    render_all_pairs,
    render_molecule_grid,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Backward-compatible aliases for tests that import private names
_load_target_names = load_target_names
_load_thresholds = load_thresholds
_is_hit = is_hit
_render_grid = render_molecule_grid


@click.command()
@click.option("--input", "input_path", default="report/data/all_results.parquet", help="Parquet with results")
@click.option("--output-dir", default="report/figures", help="Output directory for PNGs")
@click.option("--config", default="configs/pairs.yaml", help="Target pairs config")
@click.option("--thresholds", default="configs/metrics.yaml", help="Hit thresholds")
@click.option("--top-n", default=5, type=int, help="Number of top molecules for top_hits")
def main(input_path: str, output_dir: str, config: str, thresholds: str, top_n: int) -> None:
    """Render 2D molecule grids with score annotations for each pair.

    Produces per-pair per-mode molecule grid PNGs and a combined top-hits
    grid.

    Args:
        input_path: Path to the aggregated results Parquet file.
        output_dir: Output directory for figure PNGs.
        config: Path to pairs.yaml.
        thresholds: Path to metrics.yaml (threshold definitions).
        top_n: Number of top molecules to include in the top-hits grid.
    """
    in_path = PROJECT_ROOT / input_path
    if not in_path.exists():
        logger.error("Input file not found: {}. Run aggregation first.", in_path)
        sys.exit(1)

    df = pd.read_parquet(in_path)
    fig_dir = PROJECT_ROOT / output_dir
    target_names = load_target_names(PROJECT_ROOT / config)
    thresh = load_thresholds(PROJECT_ROOT / thresholds)

    from dualdocking.config import load_experiment
    from dualdocking.scoring.objectives import parse_objectives_config

    try:
        exp_cfg = load_experiment(PROJECT_ROOT / "configs")
        modes = exp_cfg.get("modes", ["single_a", "single_b", "double"])
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load experiment.yaml: {}. Using default modes.", exc)
        exp_cfg = {}
        modes = ["single_a", "single_b", "double"]

    objectives = parse_objectives_config(exp_cfg)
    primary_col = next((obj.column_name for obj in objectives if obj.is_primary), "combined_min")
    if primary_col not in df.columns:
        logger.warning("Primary objective {} missing in data, falling back to combined_min", primary_col)
        primary_col = "combined_min"

    render_config = MoleculeRenderConfig(
        modes=modes,
        primary_col=primary_col,
        top_n=top_n,
    )
    render_all_pairs(df, fig_dir, target_names, thresh, config=render_config)


if __name__ == "__main__":
    main()
