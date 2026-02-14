"""Generate comparison plots for dual-target experiment modes.

Reads ``report/data/all_results.parquet`` and produces per-pair histograms,
ECDF curves, and boxplots comparing score distributions across generation
modes.

This is a thin CLI wrapper around :mod:`dualdocking.reporting.plots.distributions`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from loguru import logger

from dualdocking.reporting.plots.distributions import PlotConfig, plot_all_pairs, plot_score_comparison

# Re-export for backward compatibility (tests import from this module)
__all__ = ["PlotConfig", "main", "plot_all_pairs", "plot_score_comparison"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@click.command()
@click.option("--input", "input_path", default="report/data/all_results.parquet")
@click.option("--output-dir", default="report/figures")
def main(input_path: str, output_dir: str) -> None:
    """Generate comparison plots for all pairs.

    Reads the aggregated Parquet file and produces per-pair figures
    comparing score distributions across generation modes.

    Args:
        input_path: Path to the aggregated results Parquet file.
        output_dir: Output directory for figure PNGs.
    """
    in_path = PROJECT_ROOT / input_path
    if not in_path.exists():
        logger.error("Input file not found: {}. Run aggregation first.", in_path)
        sys.exit(1)

    fig_dir = PROJECT_ROOT / output_dir

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
    objective_columns = [(obj.column_name, obj.column_name) for obj in objectives]

    config = PlotConfig(
        data_path=in_path,
        output_dir=fig_dir,
        modes=modes,
        objective_columns=objective_columns or None,
    )
    plot_all_pairs(config)


if __name__ == "__main__":
    main()
