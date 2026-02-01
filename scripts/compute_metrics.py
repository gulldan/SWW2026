"""Вычисление метрик двойных хитов и статистических тестов.

Читает report/data/all_results.parquet + configs/metrics.yaml,
записывает report/metrics.json и report/stats.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@click.command()
@click.option("--input", "input_path", default="report/data/all_results.parquet")
@click.option("--metrics-config", default="configs/metrics.yaml")
def main(input_path: str, metrics_config: str) -> None:
    """Вычислить метрики и статистические сравнения."""
    from dualdocking.config import load_yaml
    from dualdocking.metrics import compare_distributions, compute_dual_hits

    in_path = PROJECT_ROOT / input_path
    if not in_path.exists():
        logger.error("Входной файл не найден: {}", in_path)
        sys.exit(1)

    df = pd.read_parquet(in_path)
    metrics_cfg = load_yaml(PROJECT_ROOT / metrics_config)

    all_metrics: dict = {}
    all_stats: dict = {}

    for pair_id in df["pair"].unique():
        threshold_a = metrics_cfg["thresholds"].get(pair_id, {}).get("target_a", 0.0)
        threshold_b = metrics_cfg["thresholds"].get(pair_id, {}).get("target_b", 0.0)

        pair_metrics: dict = {}
        pair_stats: dict = {}

        for mode in ("single", "double"):
            subset = df[(df["pair"] == pair_id) & (df["mode"] == mode)]
            if subset.empty:
                continue

            scores_a = subset["score_a"].to_numpy()
            scores_b = subset["score_b"].to_numpy()

            hits = compute_dual_hits(scores_a, scores_b, threshold_a, threshold_b)
            pair_metrics[mode] = hits.to_dict()

        # Статистическое сравнение single vs double
        single = df[(df["pair"] == pair_id) & (df["mode"] == "single")]
        double = df[(df["pair"] == pair_id) & (df["mode"] == "double")]

        if not single.empty and not double.empty:
            for col in ("score_a", "score_b", "combined_min"):
                pair_stats[col] = compare_distributions(
                    single[col].to_numpy(),
                    double[col].to_numpy(),
                )

        all_metrics[pair_id] = pair_metrics
        all_stats[pair_id] = pair_stats

    report_dir = PROJECT_ROOT / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    (report_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    (report_dir / "stats.json").write_text(json.dumps(all_stats, indent=2))

    logger.info("Метрики -> report/metrics.json")
    logger.info("Статистика -> report/stats.json")


if __name__ == "__main__":
    main()
