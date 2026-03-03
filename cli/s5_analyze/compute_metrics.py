"""Compute summary metrics and hierarchical bootstrap statistics.

Thin CLI wrapper around :func:`dualdocking.analysis.metrics.computation.compute_metrics`.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@click.command()
@click.option("--results-root", default="output/results", help="Root directory with results")
@click.option("--pairs-config", default="configs/pairs.yaml")
@click.option("--experiment-config", default="configs/experiment.yaml")
@click.option("--metrics-config", default="configs/metrics.yaml")
@click.option("--output-metrics", default="report/metrics.json")
@click.option("--output-stats", default="report/stats.json")
@click.option("--allow-incomplete/--no-allow-incomplete", default=True, help="Allow missing pair-mode-seed combos")
@click.option("--pair", multiple=True, default=None, help="Filter to specific pair(s). Default: all non-blocked.")
@click.option("--seeds", "seeds_override", default=None, help="Comma-separated seed list override. Default: all from config.")
def main(
    results_root: str,
    pairs_config: str,
    experiment_config: str,
    metrics_config: str,
    output_metrics: str,
    output_stats: str,
    allow_incomplete: bool,
    pair: tuple[str, ...],
    seeds_override: str | None,
) -> None:
    """Compute per-pair metrics and run hierarchical bootstrap comparisons."""
    from dualdocking.analysis.metrics.computation import ComputeMetricsConfig, compute_metrics
    from dualdocking.config import load_experiment, load_metrics, load_pairs

    all_pairs = load_pairs(PROJECT_ROOT / pairs_config)["pairs"]
    pairs = {pid: cfg for pid, cfg in all_pairs.items() if cfg.get("set") != "blocked"}
    if pair:
        pairs = {pid: cfg for pid, cfg in pairs.items() if pid in pair}
    exp_cfg = load_experiment(PROJECT_ROOT / experiment_config)
    metrics_cfg = load_metrics(PROJECT_ROOT / metrics_config)
    seeds = [int(s) for s in seeds_override.split(",")] if seeds_override else exp_cfg.get("seeds", [42, 137, 256, 512, 1024])

    cfg = ComputeMetricsConfig(
        project_root=PROJECT_ROOT,
        pairs=pairs,
        modes=exp_cfg.get("modes", ["single_a", "single_b", "double"]),
        seeds=seeds,
        metrics_cfg=metrics_cfg,
        exp_cfg=exp_cfg,
        quality_gate=exp_cfg.get("quality_gate", {}),
        results_root=Path(results_root),
        allow_incomplete=allow_incomplete,
    )
    result = compute_metrics(cfg)

    out_metrics_path = PROJECT_ROOT / output_metrics
    out_stats_path = PROJECT_ROOT / output_stats
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    out_stats_path.parent.mkdir(parents=True, exist_ok=True)

    out_metrics_path.write_text(json.dumps(result.metrics, indent=2))
    out_stats_path.write_text(json.dumps(result.stats, indent=2))

    logger.info("Metrics -> {}", out_metrics_path)
    logger.info("Stats -> {}", out_stats_path)


if __name__ == "__main__":
    main()
