"""Threshold sensitivity analysis (SCI-06 / KVA-270).

Thin CLI wrapper around :func:`dualdocking.analysis.metrics.sensitivity.run_sensitivity`.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@click.command()
@click.option("--results-root", default="output/results")
@click.option("--pairs-config", default="configs/pairs.yaml")
@click.option("--experiment-config", default="configs/experiment.yaml")
@click.option("--metrics-config", default="configs/metrics.yaml")
@click.option("--output-csv", default="report/data/threshold_sensitivity.csv")
@click.option("--output-json", default="report/data/threshold_sensitivity.json")
@click.option("--allow-incomplete/--no-allow-incomplete", default=True)
def main(
    results_root: str,
    pairs_config: str,
    experiment_config: str,
    metrics_config: str,
    output_csv: str,
    output_json: str,
    allow_incomplete: bool,
) -> None:
    """Run threshold sensitivity analysis across pre-registered thresholds."""
    from dualdocking.analysis.metrics.sensitivity import SensitivityConfig, run_sensitivity
    from dualdocking.config import load_experiment, load_metrics, load_pairs

    pairs = load_pairs(PROJECT_ROOT / pairs_config)["pairs"]
    exp_cfg = load_experiment(PROJECT_ROOT / experiment_config)
    metrics_cfg = load_metrics(PROJECT_ROOT / metrics_config)

    sensitivity_thresholds: list[float] = metrics_cfg.get("sensitivity_thresholds", [-1.0, 0.0, 1.0])

    cfg = SensitivityConfig(
        project_root=PROJECT_ROOT,
        pairs=pairs,
        modes=exp_cfg.get("modes", ["single_a", "single_b", "double"]),
        seeds=exp_cfg.get("seeds", [42, 137, 256, 512, 1024]),
        metrics_cfg=metrics_cfg,
        exp_cfg=exp_cfg,
        quality_gate=exp_cfg.get("quality_gate", {}),
        results_root=Path(results_root),
        thresholds=sensitivity_thresholds,
        allow_incomplete=allow_incomplete,
    )
    result = run_sensitivity(cfg)

    csv_path = PROJECT_ROOT / output_csv
    json_path = PROJECT_ROOT / output_json
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if result.rows:
        from dualdocking.data.io_utils import write_df

        write_df(pd.DataFrame(result.rows), csv_path.with_suffix(""))
        logger.info("Data -> {}", csv_path.with_suffix(".parquet"))
    else:
        logger.warning("No data to write to CSV (no results found)")

    json_path.write_text(json.dumps(result.json_out, indent=2))
    logger.info("JSON -> {}", json_path)

    # Print robustness summary
    for thr_key, s in result.json_out.get("robustness_summary", {}).items():
        status = "PASS" if s["passes_success_criterion"] else "FAIL"
        logger.info(
            "T={}: {}/{} core pairs significant [{}]",
            thr_key,
            s["n_core_significant"],
            s["n_core_total"],
            status,
        )


if __name__ == "__main__":
    main()
