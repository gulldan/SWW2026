"""Threshold sensitivity analysis (SCI-06 / KVA-270).

Repeats the primary hit-rate analysis at multiple threshold settings to check
robustness of the primary conclusion.  Pre-registered thresholds::

    T = -1.0 (100 nM), 0.0 (1 uM), +1.0 (10 uM)

Outputs:
  - ``report/data/threshold_sensitivity.csv``  (flat table)
  - ``report/data/threshold_sensitivity.json`` (structured, per-threshold stats)
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_N_BOOTSTRAP = 10_000

DEFAULT_THRESHOLDS = [-1.0, 0.0, 1.0]


def _sem(values: list[float]) -> float:
    """Compute the standard error of the mean.

    Args:
        values: Numeric observations.

    Returns:
        SEM, or 0.0 if fewer than 2 finite values.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def _load_scored(path: Path) -> pd.DataFrame:
    """Load scored data (Parquet with CSV fallback), keeping raw affinity columns only.

    Args:
        path: Stem path (without extension) to ``scored_joined``.

    Returns:
        Deduplicated DataFrame with ``affinity_a``, ``affinity_b``,
        ``smiles_canonical`` columns.

    Raises:
        ValueError: If ``smiles_canonical`` is missing and cannot be derived.
    """
    from dualdocking.data.io_utils import read_df

    df = read_df(path)
    if "affinity_a" not in df.columns and "score_a" in df.columns:
        df = df.rename(columns={"score_a": "affinity_a"})
    if "affinity_b" not in df.columns and "score_b" in df.columns:
        df = df.rename(columns={"score_b": "affinity_b"})
    if "smiles_canonical" not in df.columns:
        if "smiles" in df.columns:
            from dualdocking.data.chem_sanitize import canonicalize

            df["smiles_canonical"] = [canonicalize(s) or "" for s in df["smiles"].astype(str).tolist()]
        else:
            raise ValueError(f"smiles_canonical missing in {path}")
    df = df[df["smiles_canonical"].astype(str).str.len() > 0]
    df = df.drop_duplicates(subset=["smiles_canonical"], keep="first")
    return df


def _compute_hr_both(df: pd.DataFrame, thr: float) -> float:
    """Compute dual hit rate at symmetric threshold T_A = T_B = *thr*.

    Args:
        df: DataFrame with ``affinity_a`` and ``affinity_b`` columns.
        thr: Symmetric hit-rate threshold (lower is better).

    Returns:
        Fraction of molecules hitting both targets, or NaN if no valid data.
    """
    aff_a = df["affinity_a"].to_numpy()
    aff_b = df["affinity_b"].to_numpy()
    valid = np.isfinite(aff_a) & np.isfinite(aff_b)
    aff_a, aff_b = aff_a[valid], aff_b[valid]
    if len(aff_a) == 0:
        return float("nan")
    return float(np.mean((aff_a <= thr) & (aff_b <= thr)))


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
    """Run threshold sensitivity analysis across pre-registered thresholds.

    Sweeps multiple hit-rate thresholds, re-runs the H1a/H1b bootstrap tests
    at each threshold, applies FDR correction, and evaluates robustness of the
    success criterion.

    Args:
        results_root: Root directory containing per-pair results.
        pairs_config: Path to pairs.yaml.
        experiment_config: Path to experiment.yaml.
        metrics_config: Path to metrics.yaml.
        output_csv: Output CSV path relative to project root.
        output_json: Output JSON path relative to project root.
        allow_incomplete: Allow missing pair-mode-seed combinations.
    """
    from dualdocking.analysis.fairness import validate_fairness, validate_reward_quality
    from dualdocking.analysis.statistical import (
        cliffs_delta,
        cohens_d,
        fdr_correction,
        hierarchical_bootstrap_test,
    )
    from dualdocking.config import load_experiment, load_metrics, load_pairs
    from dualdocking.data.io_utils import stem_exists

    pairs = load_pairs(PROJECT_ROOT / pairs_config)["pairs"]
    exp_cfg = load_experiment(PROJECT_ROOT / experiment_config)
    metrics_cfg = load_metrics(PROJECT_ROOT / metrics_config)

    modes = exp_cfg.get("modes", ["single_a", "single_b", "double"])
    seeds = exp_cfg.get("seeds", [42, 137, 256, 512, 1024])
    qg_cfg = exp_cfg.get("quality_gate", {})

    # Read sensitivity thresholds from config or use pre-registered defaults
    sensitivity_thresholds: list[float] = metrics_cfg.get("sensitivity_thresholds", DEFAULT_THRESHOLDS)
    logger.info("Sensitivity thresholds: {}", sensitivity_thresholds)

    results_root_path = PROJECT_ROOT / results_root

    # ---- Load all DataFrames (threshold-agnostic) ----
    raw_data: dict[str, dict[str, dict[int, pd.DataFrame]]] = {}
    for pair_id in pairs:
        raw_data[pair_id] = {}
        for mode in modes:
            raw_data[pair_id][mode] = {}
            for seed in seeds:
                run_dir = results_root_path / pair_id / mode / f"seed_{seed}"
                scored_stem = run_dir / "scored_joined"
                if not stem_exists(scored_stem):
                    continue

                fairness = validate_fairness(run_dir)
                if not fairness["budget_consistent"]:
                    logger.error("Fairness fail: {}", run_dir)
                    sys.exit(1)

                rq = validate_reward_quality(
                    run_dir,
                    max_invalid_fraction=qg_cfg.get("max_invalid_fraction", 0.10),
                    max_timeout_fraction=qg_cfg.get("max_timeout_fraction", 0.05),
                )
                if not rq["run_valid"]:
                    logger.warning("Quality gate fail: {}", run_dir)
                    continue

                raw_data[pair_id][mode][seed] = _load_scored(scored_stem)

    # Completeness check
    missing_list = [
        f"{pair_id}/{mode}/seed_{seed}"
        for pair_id in pairs
        for mode in modes
        for seed in seeds
        if seed not in raw_data.get(pair_id, {}).get(mode, {})
    ]
    if missing_list:
        if allow_incomplete:
            logger.warning("Missing {} run(s): {}", len(missing_list), missing_list)
        else:
            logger.error("Missing {} run(s) and --no-allow-incomplete: {}", len(missing_list), missing_list)
            sys.exit(1)

    # ---- Sweep thresholds ----
    rows: list[dict] = []
    json_out: dict = {"thresholds_tested": sensitivity_thresholds, "results": {}}

    for thr in sensitivity_thresholds:
        thr_key = str(thr)
        json_out["results"][thr_key] = {}

        # Metric function factory bound to this threshold
        def _make_metric_fn(threshold: float) -> Callable[[pd.DataFrame], float]:
            """Create a dual hit-rate metric function for a given threshold.

            Args:
                threshold: Symmetric hit-rate threshold.

            Returns:
                Callable that computes dual hit rate on a DataFrame.
            """

            def _fn(frame: pd.DataFrame) -> float:
                return _compute_hr_both(frame, threshold)

            return _fn

        metric_fn = _make_metric_fn(thr)

        p_values: list[float] = []
        p_index: list[tuple[str, str]] = []

        for pair_id in pairs:
            pair_stats: dict = {}
            pair_metrics: dict = {}

            for mode in modes:
                seed_dfs = raw_data.get(pair_id, {}).get(mode, {})
                if not seed_dfs:
                    continue

                hr_both_vals = [_compute_hr_both(df, thr) for df in seed_dfs.values()]
                hr_both_vals = [v for v in hr_both_vals if np.isfinite(v)]
                if not hr_both_vals:
                    continue

                pair_metrics[mode] = {
                    "hr_both_mean": float(np.mean(hr_both_vals)),
                    "hr_both_sem": _sem(hr_both_vals),
                    "n_seeds": len(hr_both_vals),
                }

            # Statistical tests: H1a (double vs single_a), H1b (double vs single_b)
            for test_id, mode_treat, mode_ctrl in [
                ("H1a", "double", "single_a"),
                ("H1b", "double", "single_b"),
            ]:
                group_ctrl = list(raw_data.get(pair_id, {}).get(mode_ctrl, {}).values())
                group_treat = list(raw_data.get(pair_id, {}).get(mode_treat, {}).values())
                if not group_ctrl or not group_treat:
                    continue

                result = hierarchical_bootstrap_test(
                    group_ctrl,
                    group_treat,
                    metric_fn,
                    n_bootstrap=_N_BOOTSTRAP,
                    rng_seed=42,
                )
                vals_ctrl = [metric_fn(df) for df in group_ctrl]
                vals_treat = [metric_fn(df) for df in group_treat]

                test_stats = {
                    "comparison": f"{mode_treat}_vs_{mode_ctrl}",
                    "observed_diff": result.observed_diff,
                    "p_value_raw": result.p_value,
                    "cliffs_delta": cliffs_delta(vals_treat, vals_ctrl),
                    "cohens_d": cohens_d(vals_ctrl, vals_treat),
                    "ci_lower": result.ci_lower,
                    "ci_upper": result.ci_upper,
                }
                pair_stats[test_id] = test_stats
                p_values.append(result.p_value)
                p_index.append((pair_id, test_id))

            json_out["results"][thr_key][pair_id] = {
                "metrics": pair_metrics,
                "stats": pair_stats,
            }

        # FDR correction per threshold
        p_adj = fdr_correction(p_values, alpha=0.05)
        for (pair_id, test_id), pv_adj in zip(p_index, p_adj, strict=False):
            entry = json_out["results"][thr_key][pair_id]["stats"][test_id]
            entry["p_value_adjusted"] = pv_adj
            entry["significant"] = (
                pv_adj < 0.05
                and abs(float(entry["cliffs_delta"])) >= 0.33
                and (float(entry["ci_lower"]) > 0 or float(entry["ci_upper"]) < 0)
            )

        # Flatten to CSV rows
        for pair_id, pair_data in json_out["results"][thr_key].items():
            for mode, m in pair_data.get("metrics", {}).items():
                rows.append(
                    {
                        "threshold": thr,
                        "pair": pair_id,
                        "mode": mode,
                        "hr_both_mean": m["hr_both_mean"],
                        "hr_both_sem": m["hr_both_sem"],
                        "n_seeds": m["n_seeds"],
                    }
                )
            for test_id, s in pair_data.get("stats", {}).items():
                # Find the matching row for mode=double to annotate
                for r in rows:
                    if r["threshold"] == thr and r["pair"] == pair_id and r["mode"] == "double":
                        r[f"{test_id}_diff"] = s["observed_diff"]
                        r[f"{test_id}_p_adj"] = s.get("p_value_adjusted", s["p_value_raw"])
                        r[f"{test_id}_cliffs_d"] = s["cliffs_delta"]
                        r[f"{test_id}_sig"] = s.get("significant", False)

    # ---- Robustness summary ----
    core_pairs = [p for p, cfg in pairs.items() if cfg.get("set") == "core"]
    summary: dict[str, dict] = {}
    for thr in sensitivity_thresholds:
        thr_key = str(thr)
        n_pass = 0
        for pair_id in core_pairs:
            pair_data = json_out["results"].get(thr_key, {}).get(pair_id, {})
            stats = pair_data.get("stats", {})
            h1a = stats.get("H1a", {})
            h1b = stats.get("H1b", {})
            if h1a.get("significant") or h1b.get("significant"):
                n_pass += 1
        summary[thr_key] = {
            "n_core_significant": n_pass,
            "n_core_total": len(core_pairs),
            "passes_success_criterion": n_pass >= 3,
        }
    json_out["robustness_summary"] = summary

    # ---- Write outputs ----
    csv_path = PROJECT_ROOT / output_csv
    json_path = PROJECT_ROOT / output_json
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        from dualdocking.data.io_utils import write_df

        write_df(pd.DataFrame(rows), csv_path.with_suffix(""))
        logger.info("Data -> {}", csv_path.with_suffix(".parquet"))
    else:
        logger.warning("No data to write to CSV (no results found)")

    json_path.write_text(json.dumps(json_out, indent=2))
    logger.info("JSON -> {}", json_path)

    # Print robustness summary
    for thr_key, s in summary.items():
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
