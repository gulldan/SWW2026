"""Compute summary metrics and hierarchical bootstrap statistics.

Reads ``results/<pair>/<mode>/seed_<seed>/scored_joined.parquet`` (with CSV
fallback) and writes:

- ``report/metrics.json`` — per-pair per-mode summary (mean/SEM).
- ``report/stats.json`` — statistical tests with FDR correction, degradation
  guard, and completeness information.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_N_BOOTSTRAP = 10_000


def _sem(values: list[float]) -> float:
    """Compute the standard error of the mean for a list of values.

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


def _load_scored(path: Path, thr_a: float, thr_b: float) -> pd.DataFrame:
    """Load scored data (Parquet with CSV fallback), add hit flags, deduplicate by canonical SMILES.

    Args:
        path: Stem path (without extension) to ``scored_joined``.
        thr_a: Hit threshold for target A (lower is better).
        thr_b: Hit threshold for target B (lower is better).

    Returns:
        Deduplicated DataFrame with ``is_hit_a``, ``is_hit_b``,
        ``is_hit_both`` columns.

    Raises:
        ValueError: If ``smiles_canonical`` is missing and cannot be derived.
    """
    from dualdocking.data.io_utils import read_df

    df = read_df(path)

    # Backward compatibility: rename legacy columns
    if "affinity_a" not in df.columns and "score_a" in df.columns:
        df = df.rename(columns={"score_a": "affinity_a"})
    if "affinity_b" not in df.columns and "score_b" in df.columns:
        df = df.rename(columns={"score_b": "affinity_b"})

    # Canonical SMILES
    if "smiles_canonical" not in df.columns:
        if "smiles" in df.columns:
            from dualdocking.data.chem_sanitize import canonicalize

            df["smiles_canonical"] = [canonicalize(s) or "" for s in df["smiles"].astype(str).tolist()]
        else:
            raise ValueError(f"smiles_canonical missing in {path}")

    # Per-target hit flags + dual hit
    aff_a = df["affinity_a"].to_numpy()
    aff_b = df["affinity_b"].to_numpy()
    if "is_hit_a" not in df.columns:
        df["is_hit_a"] = aff_a <= thr_a
    if "is_hit_b" not in df.columns:
        df["is_hit_b"] = aff_b <= thr_b
    if "is_hit_both" not in df.columns:
        df["is_hit_both"] = (aff_a <= thr_a) & (aff_b <= thr_b)

    # Deduplicate by canonical smiles
    df = df[df["smiles_canonical"].astype(str).str.len() > 0]
    df = df.drop_duplicates(subset=["smiles_canonical"], keep="first")
    return df


def _compute_degradation_guard(
    pairs: dict[str, dict],
    data: dict[str, dict[str, dict[int, pd.DataFrame]]],
    cliffs_delta_fn: object,
) -> dict[str, dict[str, dict[str, float | bool]]]:
    """Degradation guard per SAP Section 5.2.

    Checks per-target hit-rate: double must not degrade vs corresponding
    single. Threshold: Cliff's delta > -0.47 (below large-effect boundary).

    Args:
        pairs: Pair configuration dict.
        data: Nested dict of pair_id -> mode -> seed -> DataFrame.
        cliffs_delta_fn: Callable computing Cliff's delta effect size.

    Returns:
        Dict mapping pair_id -> comparison_name -> {cliffs_delta, degraded}.
    """
    no_degradation_thr = -0.47
    degradation_out: dict[str, dict[str, dict[str, float | bool]]] = {}

    def _metric_hr_a(df: pd.DataFrame) -> float:
        return float(df["is_hit_a"].mean()) if not df.empty else float("nan")

    def _metric_hr_b(df: pd.DataFrame) -> float:
        return float(df["is_hit_b"].mean()) if not df.empty else float("nan")

    for pair_id in pairs:
        group_double = list(data.get(pair_id, {}).get("double", {}).values())
        group_single_a = list(data.get(pair_id, {}).get("single_a", {}).values())
        group_single_b = list(data.get(pair_id, {}).get("single_b", {}).values())

        pair_deg: dict[str, dict[str, float | bool]] = {}
        if group_double and group_single_a:
            vals_double = [_metric_hr_a(df) for df in group_double]
            vals_single = [_metric_hr_a(df) for df in group_single_a]
            delta = cliffs_delta_fn(vals_double, vals_single)
            pair_deg["hr_a_double_vs_single_a"] = {
                "cliffs_delta": delta,
                "degraded": delta <= no_degradation_thr,
            }

        if group_double and group_single_b:
            vals_double = [_metric_hr_b(df) for df in group_double]
            vals_single = [_metric_hr_b(df) for df in group_single_b]
            delta = cliffs_delta_fn(vals_double, vals_single)
            pair_deg["hr_b_double_vs_single_b"] = {
                "cliffs_delta": delta,
                "degraded": delta <= no_degradation_thr,
            }

        if pair_deg:
            degradation_out[pair_id] = pair_deg

    return degradation_out


def _compute_sensitivity_analyses(
    pairs: dict[str, dict],
    modes: list[str],
    data: dict[str, dict[str, dict[int, pd.DataFrame]]],
    metric_hr_both_fn: object,
) -> dict[str, object]:
    """Compute sensitivity analyses (SAP Section 9).

    These are SECONDARY analyses -- not part of primary decision rule.
    Each sub-analysis is wrapped in try/except so failures don't block
    the main pipeline.
    """
    sensitivity: dict[str, object] = {}

    try:
        from dualdocking.analysis.power_analysis import detectable_effect as _det_effect
        from dualdocking.analysis.power_analysis import sensitivity_table as _sens_table

        sensitivity["power_analysis"] = {
            "detectable_cliffs_delta_m5": _det_effect(n_seeds=5, n_simulations=500, n_bootstrap=499),
            "sensitivity_table": _sens_table(seed_range=[3, 5, 10, 15], n_simulations=500, n_bootstrap=499),
        }
    except Exception as exc:
        logger.warning("Power analysis failed: {}", exc)

    try:
        from dualdocking.analysis.wild_bootstrap import wild_cluster_bootstrap

        wild_results: dict[str, dict[str, object]] = {}
        for pair_id in pairs:
            pair_wild: dict[str, object] = {}
            for test_id, mode_b, mode_a in [("H1a", "double", "single_a"), ("H1b", "double", "single_b")]:
                group_a_dfs = list(data.get(pair_id, {}).get(mode_a, {}).values())
                group_b_dfs = list(data.get(pair_id, {}).get(mode_b, {}).values())
                if not group_a_dfs or not group_b_dfs:
                    continue
                vals_a = np.array([metric_hr_both_fn(df) for df in group_a_dfs])
                vals_b = np.array([metric_hr_both_fn(df) for df in group_b_dfs])
                wb = wild_cluster_bootstrap(vals_b, vals_a, n_boot=9999, alternative="less")
                pair_wild[test_id] = {
                    "p_value": wb.p_value,
                    "observed_diff": wb.observed_diff,
                    "ci_lower": wb.ci_lower,
                    "ci_upper": wb.ci_upper,
                    "alternative": wb.alternative,
                }
            if pair_wild:
                wild_results[pair_id] = pair_wild
        sensitivity["wild_bootstrap"] = wild_results
    except Exception as exc:
        logger.warning("Wild bootstrap failed: {}", exc)

    try:
        from dualdocking.analysis.rliable_metrics import compute_iqm, compute_iqm_ci, probability_of_improvement

        rliable_out: dict[str, object] = {}
        for mode in modes:
            hr_both_vals = [
                float(df["is_hit_both"].mean())
                for pair_id in pairs
                for df in data.get(pair_id, {}).get(mode, {}).values()
                if not df.empty
            ]
            if hr_both_vals:
                arr = np.array(hr_both_vals)
                iqm = compute_iqm(arr)
                lo, hi = compute_iqm_ci(arr, n_boot=5000)
                rliable_out[mode] = {"iqm": iqm, "iqm_ci_lower": lo, "iqm_ci_upper": hi}

        for baseline in ("single_a", "single_b"):
            trt_vals = [
                float(df["is_hit_both"].mean())
                for pair_id in pairs
                for df in data.get(pair_id, {}).get("double", {}).values()
                if not df.empty
            ]
            ctrl_vals = [
                float(df["is_hit_both"].mean())
                for pair_id in pairs
                for df in data.get(pair_id, {}).get(baseline, {}).values()
                if not df.empty
            ]
            if trt_vals and ctrl_vals:
                poi = probability_of_improvement(np.array(trt_vals), np.array(ctrl_vals))
                rliable_out[f"poi_double_vs_{baseline}"] = poi

        sensitivity["rliable"] = rliable_out
    except Exception as exc:
        logger.warning("Rliable metrics failed: {}", exc)

    try:
        from dualdocking.analysis.mixed_effects import fit_mixed_model

        me_rows = []
        for pair_id in pairs:
            for mode in modes:
                for seed_id, df in data.get(pair_id, {}).get(mode, {}).items():
                    if not df.empty:
                        me_rows.append(
                            {
                                "pair": pair_id,
                                "mode": mode,
                                "seed": seed_id,
                                "hr_both": float(df["is_hit_both"].mean()),
                            }
                        )
        if me_rows:
            me_df = pd.DataFrame(me_rows)
            me_result = fit_mixed_model(me_df, outcome="hr_both", fixed="mode", group="pair")
            sensitivity["mixed_effects"] = {
                "converged": me_result.converged,
                "coefficients": me_result.coefficients,
                "p_values": me_result.p_values,
                "random_effects_var": me_result.random_effects_var,
                "n_groups": me_result.n_groups,
                "n_obs": me_result.n_obs,
            }
    except Exception as exc:
        logger.warning("Mixed-effects model failed: {}", exc)

    return sensitivity


def _compute_summary_metrics(
    pairs: dict[str, dict],
    modes: list[str],
    seeds: list[int],
    data: dict[str, dict[str, dict[int, pd.DataFrame]]],
    metrics_cfg: dict,
    exp_cfg: dict,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute per-pair per-mode summary metrics (hit rates, balance, diversity, filters).

    Args:
        pairs: Pair configuration dict.
        modes: List of generation modes.
        seeds: List of seed values.
        data: Nested dict of pair_id -> mode -> seed -> DataFrame.
        metrics_cfg: Metrics configuration.
        exp_cfg: Experiment configuration.

    Returns:
        Nested dict mapping pair_id -> mode -> metric_name -> value.
    """
    from dualdocking.analysis.diversity import compute_all_diversity_metrics
    from dualdocking.scoring.objectives import parse_objectives_config

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
            hr_b_values: list[float] = []
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
                hr_b_values.append(float(np.mean(aff_b <= thr_b)))
                hr_both_values.append(float(np.mean((aff_a <= thr_a) & (aff_b <= thr_b))))
                balance_values.append(float(np.median(np.abs(aff_a - aff_b))))

            if not hr_a_values:
                continue

            mode_metrics: dict[str, float | int] = {
                "hr_a_mean": float(np.mean(hr_a_values)),
                "hr_a_sem": _sem(hr_a_values),
                "hr_b_mean": float(np.mean(hr_b_values)),
                "hr_b_sem": _sem(hr_b_values),
                "hr_both_mean": float(np.mean(hr_both_values)),
                "hr_both_sem": _sem(hr_both_values),
                "balance_mean": float(np.mean(balance_values)),
                "balance_sem": _sem(balance_values),
                "n_valid_runs": len(seed_dfs),
                "n_expected_runs": len(seeds),
                "missing_rate": round(1.0 - len(seed_dfs) / len(seeds), 4) if seeds else 0.0,
            }

            # Per-objective mean/SEM
            objectives = parse_objectives_config(exp_cfg)
            for obj in objectives:
                col = obj.column_name
                obj_values: list[float] = []
                for df in seed_dfs.values():
                    if col in df.columns:
                        vals = df[col].dropna().to_numpy()
                        if len(vals) > 0:
                            obj_values.append(float(np.mean(vals)))
                if obj_values:
                    mode_metrics[f"{col}_mean"] = float(np.mean(obj_values))
                    mode_metrics[f"{col}_sem"] = _sem(obj_values)

            # Filtered hit rates (backward compat: only if filter columns present)
            for df in seed_dfs.values():
                if "passes_all_filters" in df.columns:
                    n_pass = int(df["passes_all_filters"].sum())
                    filter_pass_rate = n_pass / len(df) if len(df) > 0 else 0.0
                    df_filt = df[df["passes_all_filters"]]
                    mode_metrics["filter_pass_rate"] = filter_pass_rate
                    mode_metrics["hr_a_filtered"] = float(df_filt["is_hit_a"].mean()) if len(df_filt) > 0 else float("nan")
                    mode_metrics["hr_b_filtered"] = float(df_filt["is_hit_b"].mean()) if len(df_filt) > 0 else float("nan")
                    mode_metrics["hr_both_filtered"] = float(df_filt["is_hit_both"].mean()) if len(df_filt) > 0 else float("nan")
                    break  # use first seed with filter data for summary

            # Chemical diversity metrics
            all_smiles: list[str] = []
            for df in seed_dfs.values():
                smi_col = "smiles_canonical" if "smiles_canonical" in df.columns else "smiles"
                if smi_col in df.columns:
                    all_smiles.extend(df[smi_col].dropna().astype(str).tolist())
            if all_smiles:
                div_result = compute_all_diversity_metrics(all_smiles)
                for k, v in div_result.to_dict().items():
                    mode_metrics[f"diversity_{k}"] = v

            metrics_out[pair_id][mode] = mode_metrics

    return metrics_out


def _compute_gnina_cross_validation(
    pairs: dict[str, dict],
    modes: list[str],
    data: dict[str, dict[str, dict[int, pd.DataFrame]]],
) -> dict[str, dict[str, object]]:
    """Compute GNINA cross-validation metrics if gnina columns are present.

    For each pair/mode, computes Spearman correlations between GNINA CNN
    affinity and Boltz-2 affinity, and between GNINA Vina scores and
    UniDock scores.

    Args:
        pairs: Pair configuration dict.
        modes: List of generation modes.
        data: Nested dict of pair_id -> mode -> seed -> DataFrame.

    Returns:
        Dict mapping pair_id -> mode -> metrics dict. Empty if no GNINA data.
    """
    from scipy.stats import spearmanr

    _false = False  # named constant avoids FBT003 lint

    gnina_xval: dict[str, dict[str, object]] = {}
    for pair_id in pairs:
        pair_gnina: dict[str, object] = {}
        for mode in modes:
            seed_dfs = data.get(pair_id, {}).get(mode, {})
            if not seed_dfs:
                continue

            gnina_affs_a: list[float] = []
            gnina_affs_b: list[float] = []
            boltz_affs_a: list[float] = []
            boltz_affs_b: list[float] = []
            for df in seed_dfs.values():
                if "gnina_cnn_affinity_a" not in df.columns:
                    continue
                valid_mask = df["gnina_valid_a"].fillna(value=_false) & df["gnina_valid_b"].fillna(value=_false)
                valid_df = df[valid_mask]
                if len(valid_df) == 0:
                    continue
                gnina_affs_a.extend(valid_df["gnina_cnn_affinity_a"].tolist())
                gnina_affs_b.extend(valid_df["gnina_cnn_affinity_b"].tolist())
                boltz_affs_a.extend(valid_df["affinity_a"].tolist())
                boltz_affs_b.extend(valid_df["affinity_b"].tolist())

            if len(gnina_affs_a) < 5:
                continue

            gnina_a = np.array(gnina_affs_a)
            gnina_b = np.array(gnina_affs_b)
            boltz_a = np.array(boltz_affs_a)
            boltz_b = np.array(boltz_affs_b)

            rho_a_boltz, _ = spearmanr(gnina_a, boltz_a)
            rho_b_boltz, _ = spearmanr(gnina_b, boltz_b)

            mode_gnina: dict[str, float] = {
                "n_valid": len(gnina_affs_a),
                "mean_cnn_affinity_a": float(np.nanmean(gnina_a)),
                "mean_cnn_affinity_b": float(np.nanmean(gnina_b)),
                "spearman_gnina_boltz_a": float(rho_a_boltz),
                "spearman_gnina_boltz_b": float(rho_b_boltz),
            }

            for df in seed_dfs.values():
                if "gnina_vina_score_a" in df.columns and "score_a" in df.columns:
                    valid_mask = df["gnina_valid_a"].fillna(value=_false)
                    vdf = df[valid_mask]
                    if len(vdf) >= 5:
                        rho_vina_a, _ = spearmanr(
                            vdf["gnina_vina_score_a"].to_numpy(),
                            vdf["score_a"].to_numpy(),
                        )
                        mode_gnina["spearman_gnina_vina_unidock_a"] = float(rho_vina_a)
                if "gnina_vina_score_b" in df.columns and "score_b" in df.columns:
                    valid_mask = df["gnina_valid_b"].fillna(value=_false)
                    vdf = df[valid_mask]
                    if len(vdf) >= 5:
                        rho_vina_b, _ = spearmanr(
                            vdf["gnina_vina_score_b"].to_numpy(),
                            vdf["score_b"].to_numpy(),
                        )
                        mode_gnina["spearman_gnina_vina_unidock_b"] = float(rho_vina_b)
                break  # use first seed with GNINA data

            pair_gnina[mode] = mode_gnina

        if pair_gnina:
            gnina_xval[pair_id] = pair_gnina

    return gnina_xval


@click.command()
@click.option("--results-root", default="output/results", help="Root directory with results")
@click.option("--pairs-config", default="configs/pairs.yaml")
@click.option("--experiment-config", default="configs/experiment.yaml")
@click.option("--metrics-config", default="configs/metrics.yaml")
@click.option("--output-metrics", default="report/metrics.json")
@click.option("--output-stats", default="report/stats.json")
@click.option("--allow-incomplete/--no-allow-incomplete", default=True, help="Allow missing pair-mode-seed combos")
def main(
    results_root: str,
    pairs_config: str,
    experiment_config: str,
    metrics_config: str,
    output_metrics: str,
    output_stats: str,
    allow_incomplete: bool,
) -> None:
    """Compute per-pair metrics and run hierarchical bootstrap comparisons.

    For each core pair, runs H1a/H1b (dual hit-rate) and H2a/H2b (balance)
    tests, applies BH FDR correction on the primary family, evaluates
    the degradation guard, and writes structured JSON outputs.

    Args:
        results_root: Root directory containing per-pair results.
        pairs_config: Path to pairs.yaml.
        experiment_config: Path to experiment.yaml.
        metrics_config: Path to metrics.yaml.
        output_metrics: Output path for metrics JSON.
        output_stats: Output path for stats JSON.
        allow_incomplete: Allow missing pair-mode-seed combinations.
    """
    from dualdocking.analysis.fairness import validate_fairness, validate_reward_quality
    from dualdocking.analysis.statistical import (
        BootstrapResult,
        cliffs_delta,
        cohens_d,
        fdr_correction,
        hierarchical_bootstrap_test,
    )
    from dualdocking.config import load_experiment, load_metrics, load_pairs
    from dualdocking.data.io_utils import stem_exists

    all_pairs = load_pairs(PROJECT_ROOT / pairs_config)["pairs"]
    # Exclude blocked pairs from analysis (they have no results).
    pairs = {pid: cfg for pid, cfg in all_pairs.items() if cfg.get("set") != "blocked"}
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
                scored_stem = run_dir / "scored_joined"
                if not stem_exists(scored_stem):
                    logger.warning("Missing scored_joined: {}", scored_stem)
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

                df = _load_scored(scored_stem, thr_a, thr_b)
                data[pair_id][mode][seed] = df

    # Summary metrics
    metrics_out = _compute_summary_metrics(pairs, modes, seeds, data, metrics_cfg, exp_cfg)

    # Completeness check
    missing_list: list[str] = []
    n_expected = 0
    n_valid = 0
    for pair_id in pairs:
        for mode in modes:
            for seed in seeds:
                n_expected += 1
                if seed in data.get(pair_id, {}).get(mode, {}):
                    n_valid += 1
                else:
                    missing_list.append(f"{pair_id}/{mode}/seed_{seed}")

    missing_rate = 1.0 - n_valid / n_expected if n_expected else 0.0
    if missing_list:
        if allow_incomplete:
            logger.warning("Missing {} run(s): {}", len(missing_list), missing_list)
        else:
            logger.error("Missing {} run(s) and --no-allow-incomplete set: {}", len(missing_list), missing_list)
            sys.exit(1)

    # Identify core pairs for FDR scope (SAP 4.2: primary family = H1a+H1b core only)
    core_pair_ids = {pid for pid, cfg in pairs.items() if cfg.get("set", "core") == "core"}

    # Statistical tests
    stats_out: dict[str, dict[str, dict[str, float | bool | str]]] = {}
    primary_p_values: list[float] = []
    primary_index: list[tuple[str, str]] = []

    def _metric_hr_a(df: pd.DataFrame) -> float:
        """Compute target-A hit rate from a seed DataFrame."""
        return float(df["is_hit_a"].mean()) if not df.empty else float("nan")

    def _metric_hr_b(df: pd.DataFrame) -> float:
        """Compute target-B hit rate from a seed DataFrame."""
        return float(df["is_hit_b"].mean()) if not df.empty else float("nan")

    def _metric_hr_both(df: pd.DataFrame) -> float:
        """Compute dual hit rate from a seed DataFrame."""
        return float(df["is_hit_both"].mean()) if not df.empty else float("nan")

    def _metric_balance(df: pd.DataFrame) -> float:
        """Compute median absolute affinity difference (balance) from a seed DataFrame."""
        if df.empty:
            return float("nan")
        return float(np.median(np.abs(df["affinity_a"].to_numpy() - df["affinity_b"].to_numpy())))

    def _balance_fast_fn(arr: np.ndarray, indices: np.ndarray) -> float:
        """Fast balance metric for bootstrap resampling.

        Args:
            arr: Pre-computed absolute affinity differences.
            indices: Bootstrap sample indices.

        Returns:
            Median of resampled values.
        """
        return float(np.median(arr[indices]))

    for pair_id in pairs:
        stats_out[pair_id] = {}
        tests = [
            ("H1a", "double", "single_a", "hr_both", _metric_hr_both, "is_hit_both", None),
            ("H1b", "double", "single_b", "hr_both", _metric_hr_both, "is_hit_both", None),
            ("H2a", "double", "single_a", "balance", _metric_balance, "_abs_aff_diff", _balance_fast_fn),
            ("H2b", "double", "single_b", "balance", _metric_balance, "_abs_aff_diff", _balance_fast_fn),
        ]

        for test_id, mode_b, mode_a, metric_name, metric_fn, fast_col, fast_fn in tests:
            group_a = list(data.get(pair_id, {}).get(mode_a, {}).values())
            group_b = list(data.get(pair_id, {}).get(mode_b, {}).values())
            if not group_a or not group_b:
                continue

            # Pre-compute derived column for balance metric
            if fast_col == "_abs_aff_diff":
                for df in (d for d in group_a + group_b if "_abs_aff_diff" not in d.columns):
                    df["_abs_aff_diff"] = np.abs(df["affinity_a"].to_numpy() - df["affinity_b"].to_numpy())

            result: BootstrapResult = hierarchical_bootstrap_test(
                group_a,
                group_b,
                metric_fn,
                n_bootstrap=_N_BOOTSTRAP,
                rng_seed=42,
                fast_column=fast_col,
                fast_fn=fast_fn,
            )
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

            is_primary = pair_id in core_pair_ids and test_id in ("H1a", "H1b")
            if is_primary:
                primary_p_values.append(result.p_value)
                primary_index.append((pair_id, test_id))

    # FDR correction — primary family only (SAP 4.2: core pairs × H1a/H1b)
    primary_p_adj = fdr_correction(primary_p_values, alpha=0.05)
    for (pair_id, test_id), p_value_adj in zip(primary_index, primary_p_adj, strict=False):
        entry = stats_out[pair_id][test_id]
        entry["p_value_adjusted"] = p_value_adj
        entry["significant"] = (
            p_value_adj < 0.05
            and abs(float(entry["cliffs_delta"])) >= 0.33
            and (float(entry["ci_lower"]) > 0 or float(entry["ci_upper"]) < 0)
        )

    # Secondary / auxiliary tests: not in primary FDR family per SAP 4.2
    for pair_id in pairs:
        for test_id in ("H1a", "H1b", "H2a", "H2b"):
            entry = stats_out.get(pair_id, {}).get(test_id)
            if entry is None:
                continue
            if "p_value_adjusted" not in entry:
                entry["p_value_adjusted"] = None
                entry["significant"] = False

    stats_out["_degradation"] = _compute_degradation_guard(pairs, data, cliffs_delta)

    stats_out["_completeness"] = {
        "n_expected": n_expected,
        "n_valid": n_valid,
        "missing_rate": round(missing_rate, 4),
        "missing_runs": missing_list,
    }

    stats_out["_sensitivity_analyses"] = _compute_sensitivity_analyses(pairs, modes, data, _metric_hr_both)
    stats_out["_gnina_cross_validation"] = _compute_gnina_cross_validation(pairs, modes, data)

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
