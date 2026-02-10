"""Aggregate scored results across all pairs into a single analysis dataset.

Reads ``results/<pair>/<mode>/seed_<seed>/scored_joined.csv`` for every
pair, mode, and seed, validates quality gates, concatenates the frames,
and writes ``report/data/all_results.parquet``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@click.command()
@click.option("--config", default="configs/pairs.yaml")
@click.option("--experiment-config", default="configs/experiment.yaml")
@click.option("--output", default="report/data/all_results.parquet")
def main(config: str, experiment_config: str, output: str) -> None:
    """Aggregate results from all pairs and modes into a single Parquet file.

    Iterates over every (pair, mode, seed) combination, applies the quality
    gate, and concatenates valid runs.

    Args:
        config: Path to pairs.yaml.
        experiment_config: Path to experiment.yaml.
        output: Output Parquet file path relative to project root.
    """
    from dualdocking.config import load_experiment, load_pairs

    all_pairs = load_pairs(PROJECT_ROOT / config)["pairs"]
    # Exclude blocked pairs from aggregation (they have no results).
    pairs = {pid: cfg for pid, cfg in all_pairs.items() if cfg.get("set") != "blocked"}
    exp_cfg = load_experiment(PROJECT_ROOT / experiment_config)
    seeds = exp_cfg.get("seeds", [42, 137, 256, 512, 1024])
    modes = exp_cfg.get("modes", ["single_a", "single_b", "double"])
    frames: list[pd.DataFrame] = []

    from dualdocking.analysis.fairness import validate_reward_quality

    qg_cfg = exp_cfg.get("quality_gate", {})

    for pair_id in pairs:
        for mode in modes:
            for seed in seeds:
                run_dir = PROJECT_ROOT / "output" / "results" / pair_id / mode / f"seed_{seed}"
                path = run_dir / "scored_joined.csv"
                if not path.exists():
                    logger.warning("Отсутствует: {}", path)
                    continue
                rq = validate_reward_quality(
                    run_dir,
                    max_invalid_fraction=qg_cfg.get("max_invalid_fraction", 0.10),
                    max_timeout_fraction=qg_cfg.get("max_timeout_fraction", 0.05),
                )
                if not rq["run_valid"]:
                    logger.warning("Quality gate failed, skipping {}: {}", run_dir, rq["issues"])
                    continue
                df = pd.read_csv(path)
                df["pair"] = pair_id
                df["mode"] = mode
                df["seed"] = seed
                frames.append(df)
                logger.info("Загружено {} строк из {}", len(df), path)

    if not frames:
        logger.error("Результаты не найдены. Сначала запустите генерацию и скоринг.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    out_path = PROJECT_ROOT / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    logger.info("Агрегировано {} строк -> {}", len(combined), out_path)


if __name__ == "__main__":
    main()
