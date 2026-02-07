"""Compute summary metrics and hierarchical bootstrap stats.

Reads results/<pair>/<mode>/seed_<seed>/scored_joined.csv and writes:
- report/metrics.json (summary mean/SEM)
- report/stats.json (16 tests with FDR correction)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _sem(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def _load_scored_csv(path: Path, thr_a: float, thr_b: float) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Backward compatibility: rename legacy columns
    if "affinity_a" not in df.columns and "score_a" in df.columns:
        df = df.rename(columns={"score_a": "affinity_a"})
    if "affinity_b" not in df.columns and "score_b" in df.columns:
        df = df.rename(columns={"score_b": "affinity_b"})

    # Canonical SMILES
    if "smiles_canonical" not in df.columns:
        if "smiles" in df.columns:
            from dualdocking.chem_sanitize import canonicalize

            df["smiles_canonical"] = [canonicalize(s) or "" for s in df["smiles"].astype(str).tolist()]
        else:
            raise ValueError(f"smiles_canonical missing in {path}")

    # is_hit_both
    if "is_hit_both" not in df.columns:
        aff_a = df["affinity_a"].to_numpy()
        aff_b = df["affinity_b"].to_numpy()
        df["is_hit_both"] = (aff_a <= thr_a) & (aff_b <= thr_b)

    # Deduplicate by canonical smiles
    df = df[df["smiles_canonical"].astype(str).str.len() > 0]
    df = df.drop_duplicates(subset=["smiles_canonical"], keep="first")
    return df


@click.command()
@click.option("--results-root", default="results", help="Root directory with results")
@click.option("--pairs-config", default="configs/pairs.yaml")
@click.option("--experiment-config", default="configs/experiment.yaml")
@click.option("--metrics-config", default="configs/metrics.yaml")
@click.option("--output-metrics", default="report/metrics.json")
@click.option("--output-stats", default="report/stats.json")
def main(
    results_root: str,
    pairs_config: str,
    experiment_config: str,
    metrics_config: str,
    output_metrics: str,
    output_stats: str,
) -> None:
    """Compute metrics and statistical comparisons."""
    from dualdocking.config import load_experiment, load_metrics, load_pairs
    from dualdocking.fairness import validate_fairness, validate_reward_quality
    from dualdocking.statistical import (
        BootstrapResult,
        cliffs_delta,
        cohens_d,
        fdr_correction,
        hierarchical_bootstrap_test,
    )

    pairs = load_pairs(PROJECT_ROOT / pairs_config)["pairs"]
    exp_cfg = load_experiment(PROJECT_ROOT / experiment_config)
    metrics_cfg = load_metrics(PROJECT_ROOT / metrics_config)

    modes = exp_cfg.get("modes", ["single_a", "single_b", "double"])
    seeds = exp_cfg.get("seeds", [42, 137, 256, 512, 1024])

    results_root_path = PROJECT_ROOT / results_root
    qg_cfg = exp_cfg.get("quality_gate", {})
    data: dict[str, dict[str, dict[int, pd.DataFrame]]] = {}

    # Fairness gate + quality gate + load data
    for pair_id in pairs:
        thresholds = metrics_cfg.get("thresholds", {}).get(pair_id, {})
        thr_a = thresholds.get("target_a", 0.0)
        thr_b = thresholds.get("target_b", 0.0)

        data[pair_id] = {}
        for mode in modes:
            data[pair_id][mode] = {}
            for seed in seeds:
                run_dir = results_root_path / pair_id / mode / f"seed_{seed}"
                scored_path = run_dir / "scored_joined.csv"
                if not scored_path.exists():
                    logger.warning("Missing scored_joined.csv: {}", scored_path)
                    continue

                fairness = validate_fairness(run_dir)
                if not fairness["budget_consistent"]:
                    logger.error("Fairness check failed for {}: {}", run_dir, fairness["issues"])
                    sys.exit(1)

                reward_quality = validate_reward_quality(
                    run_dir,
                    max_invalid_fraction=qg_cfg.get("max_invalid_fraction", 0.10),
                    max_timeout_fraction=qg_cfg.get("max_timeout_fraction", 0.05),
                )
                if not reward_quality["run_valid"]:
                    logger.warning("Quality gate failed for {}: {}", run_dir, reward_quality["issues"])
                    continue

                df = _load_scored_csv(scored_path, thr_a, thr_b)
                data[pair_id][mode][seed] = df

    # Summary metrics
    metrics_out: dict[str, dict[str, dict[str, float]]] = {}
    for pair_id in pairs:
        thresholds = metrics_cfg.get("thresholds", {}).get(pair_id, {})
        thr_a = thresholds.get("target_a", 0.0)
        thr_b = thresholds.get("target_b", 0.0)

        metrics_out[pair_id] = {}
        for mode in modes:
            seed_dfs = data.get(pair_id, {}).get(mode, {})
            if not seed_dfs:
                continue

            hr_a_values: list[float] = []
            hr_both_values: list[float] = []
            balance_values: list[float] = []

            for df in seed_dfs.values():
                aff_a = df["affinity_a"].to_numpy()
                aff_b = df["affinity_b"].to_numpy()
                valid = np.isfinite(aff_a) & np.isfinite(aff_b)
                aff_a = aff_a[valid]
                aff_b = aff_b[valid]
                if len(aff_a) == 0:
                    continue

                hr_a_values.append(float(np.mean(aff_a <= thr_a)))
                hr_both_values.append(float(np.mean((aff_a <= thr_a) & (aff_b <= thr_b))))
                balance_values.append(float(np.median(np.abs(aff_a - aff_b))))

            if not hr_a_values:
                continue

            metrics_out[pair_id][mode] = {
                "hr_a_mean": float(np.mean(hr_a_values)),
                "hr_a_sem": _sem(hr_a_values),
                "hr_both_mean": float(np.mean(hr_both_values)),
                "hr_both_sem": _sem(hr_both_values),
                "balance_mean": float(np.mean(balance_values)),
                "balance_sem": _sem(balance_values),
            }

    # Statistical tests
    stats_out: dict[str, dict[str, dict[str, float | bool | str]]] = {}
    p_values: list[float] = []
    index: list[tuple[str, str]] = []

    def _metric_hr_both(df: pd.DataFrame) -> float:
        return float(df["is_hit_both"].mean()) if not df.empty else float("nan")

    def _metric_balance(df: pd.DataFrame) -> float:
        if df.empty:
            return float("nan")
        return float(np.median(np.abs(df["affinity_a"].to_numpy() - df["affinity_b"].to_numpy())))

    for pair_id in pairs:
        stats_out[pair_id] = {}
        tests = [
            ("H1a", "double", "single_a", "hr_both", _metric_hr_both),
            ("H1b", "double", "single_b", "hr_both", _metric_hr_both),
            ("H2a", "double", "single_a", "balance", _metric_balance),
            ("H2b", "double", "single_b", "balance", _metric_balance),
        ]

        for test_id, mode_b, mode_a, metric_name, metric_fn in tests:
            group_a = list(data.get(pair_id, {}).get(mode_a, {}).values())
            group_b = list(data.get(pair_id, {}).get(mode_b, {}).values())
            if not group_a or not group_b:
                continue

            result: BootstrapResult = hierarchical_bootstrap_test(group_a, group_b, metric_fn, n_bootstrap=10000, rng_seed=42)
            # Per-seed metric values for effect sizes
            values_a = [metric_fn(df) for df in group_a]
            values_b = [metric_fn(df) for df in group_b]

            stats_out[pair_id][test_id] = {
                "comparison": f"{mode_b}_vs_{mode_a}",
                "metric": metric_name,
                "observed_diff": result.observed_diff,
                "p_value_raw": result.p_value,
                "cliffs_delta": cliffs_delta(values_b, values_a),
                "cohens_d": cohens_d(values_a, values_b),
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
            }

            p_values.append(result.p_value)
            index.append((pair_id, test_id))

    # FDR correction
    p_adj = fdr_correction(p_values, alpha=0.05)
    for (pair_id, test_id), p_value_adj in zip(index, p_adj, strict=False):
        entry = stats_out[pair_id][test_id]
        entry["p_value_adjusted"] = p_value_adj
        entry["significant"] = (
            p_value_adj < 0.05
            and abs(float(entry["cliffs_delta"])) >= 0.33
            and (float(entry["ci_lower"]) > 0 or float(entry["ci_upper"]) < 0)
        )

    out_metrics_path = PROJECT_ROOT / output_metrics
    out_stats_path = PROJECT_ROOT / output_stats
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    out_stats_path.parent.mkdir(parents=True, exist_ok=True)

    out_metrics_path.write_text(json.dumps(metrics_out, indent=2))
    out_stats_path.write_text(json.dumps(stats_out, indent=2))

    logger.info("Метрики -> {}", out_metrics_path)
    logger.info("Статистика -> {}", out_stats_path)


if __name__ == "__main__":
    main()
