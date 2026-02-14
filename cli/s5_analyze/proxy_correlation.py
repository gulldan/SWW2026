"""Proxy vs final correlation analysis (SCI-07 / KVA-271).

Thin CLI wrapper around :func:`dualdocking.analysis.metrics.correlation.run_correlation`.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@click.command()
@click.option("--results-root", default="output/results")
@click.option("--pairs-config", default="configs/pairs.yaml")
@click.option("--experiment-config", default="configs/experiment.yaml")
@click.option("--output-csv", default="report/data/proxy_final_corr.csv")
@click.option("--output-json", default="report/data/proxy_final_corr.json")
def main(
    results_root: str,
    pairs_config: str,
    experiment_config: str,
    output_csv: str,
    output_json: str,
) -> None:
    """Compute proxy (docking) vs final (Boltz-2) correlation per pair/mode."""
    from dualdocking.analysis.metrics.correlation import CorrelationConfig, run_correlation
    from dualdocking.config import load_experiment, load_pairs

    pairs = load_pairs(PROJECT_ROOT / pairs_config)["pairs"]
    exp_cfg = load_experiment(PROJECT_ROOT / experiment_config)

    cfg = CorrelationConfig(
        project_root=PROJECT_ROOT,
        pairs=pairs,
        modes=exp_cfg.get("modes", ["single_a", "single_b", "double"]),
        seeds=exp_cfg.get("seeds", [42, 137, 256, 512, 1024]),
        results_root=Path(results_root),
    )
    result = run_correlation(cfg)

    csv_path = PROJECT_ROOT / output_csv
    json_path = PROJECT_ROOT / output_json
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if result.rows:
        from dualdocking.data.io_utils import write_df

        write_df(pd.DataFrame(result.rows), csv_path.with_suffix(""))
        logger.info("Data -> {}", csv_path.with_suffix(".parquet"))
    else:
        logger.warning("No proxy-final pairs found (no docking data available)")

    json_path.write_text(json.dumps(result.json_out, indent=2))
    logger.info("JSON -> {}", json_path)

    all_rhos = [r["spearman_rho"] for r in result.rows if np.isfinite(r.get("spearman_rho", float("nan")))]
    if all_rhos:
        logger.info(
            "Median Spearman rho = {:.3f} [{:.3f}, {:.3f}], proxy suitable: {}",
            float(np.median(all_rhos)),
            float(np.min(all_rhos)),
            float(np.max(all_rhos)),
            result.json_out["summary"]["proxy_suitable"],
        )


if __name__ == "__main__":
    main()
