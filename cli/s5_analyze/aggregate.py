"""Aggregate scored results across all pairs into a single analysis dataset.

Thin CLI wrapper around :func:`dualdocking.analysis.metrics.aggregation.aggregate`.
"""

from __future__ import annotations

from pathlib import Path

import click
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@click.command()
@click.option("--config", default="configs/pairs.yaml")
@click.option("--experiment-config", default="configs/experiment.yaml")
@click.option("--output", default="report/data/all_results.parquet")
def main(config: str, experiment_config: str, output: str) -> None:
    """Aggregate results from all pairs and modes into a single Parquet file."""
    from dualdocking.analysis.metrics.aggregation import AggregateConfig, aggregate
    from dualdocking.config import load_experiment, load_pairs

    all_pairs = load_pairs(PROJECT_ROOT / config)["pairs"]
    pairs = {pid: cfg for pid, cfg in all_pairs.items() if cfg.get("set") != "blocked"}
    exp_cfg = load_experiment(PROJECT_ROOT / experiment_config)

    cfg = AggregateConfig(
        project_root=PROJECT_ROOT,
        pairs=pairs,
        modes=exp_cfg.get("modes", ["single_a", "single_b", "double"]),
        seeds=exp_cfg.get("seeds", [42, 137, 256, 512, 1024]),
        quality_gate=exp_cfg.get("quality_gate", {}),
        output_path=Path(output),
    )
    result = aggregate(cfg)
    logger.info("Done: {}", result)


if __name__ == "__main__":
    main()
