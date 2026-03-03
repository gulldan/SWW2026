"""Compute summary metrics and hierarchical bootstrap statistics.

Reads scored results per (pair, mode, seed), computes hit rates, effect sizes,
FDR-corrected p-values, degradation guard, and sensitivity analyses.

Outputs structured dicts suitable for JSON serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from dualdocking.analysis.metrics.common import (
    add_hit_flags,
    load_scored_affinity,
)
from dualdocking.analysis.metrics.common import (
    sem as _sem_impl,
)

_N_BOOTSTRAP = 10_000
_H2_BOOTSTRAP_CAP = 2_000
_MIN_SEEDS_FOR_SENSITIVITY = 3
_POWER_SIMULATIONS = 120
_POWER_BOOTSTRAP = 199
_WILD_BOOTSTRAP = 2_999
_RLIABLE_BOOTSTRAP = 1_500


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComputeMetricsConfig:
    """Configuration for metric computation.

    Attributes:
        project_root: Absolute path to project root.
        pairs: Pair configuration dict (pair_id -> config), excluding blocked.
        modes: Generation modes.
        seeds: Random seeds.
        metrics_cfg: Parsed metrics.yaml contents (thresholds etc.).
        exp_cfg: Parsed experiment.yaml contents.
        quality_gate: Quality gate thresholds dict.
        results_root: Root results directory (absolute or relative to project_root).
        allow_incomplete: Allow missing pair-mode-seed combinations.
        n_bootstrap: Number of bootstrap resamples for statistical tests.
    """

    project_root: Path
    pairs: dict[str, dict]
    modes: list[str]
    seeds: list[int]
    metrics_cfg: dict[str, Any]
    exp_cfg: dict[str, Any]
    quality_gate: dict[str, float] = field(default_factory=dict)
    results_root: Path = Path("output/results")
    allow_incomplete: bool = True
    n_bootstrap: int = _N_BOOTSTRAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sem(values: list[float]) -> float:
    """Compute standard error of the mean.

    Args:
        values: Numeric observations.

    Returns:
        SEM, or 0.0 if fewer than 2 finite values.
    """
    return _sem_impl(values)


def load_scored(path: Path, thr_a: float, thr_b: float) -> pd.DataFrame:
    """Load scored data, add hit flags, deduplicate by canonical SMILES.

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
    return add_hit_flags(load_scored_affinity(path), thr_a, thr_b)


# ---------------------------------------------------------------------------
# Metric functions (used as callbacks for bootstrap)
# ---------------------------------------------------------------------------


def metric_hr_a(df: pd.DataFrame) -> float:
    """Compute target-A hit rate from a seed DataFrame."""
    return float(df["is_hit_a"].mean()) if not df.empty else float("nan")


def metric_hr_b(df: pd.DataFrame) -> float:
    """Compute target-B hit rate from a seed DataFrame."""
    return float(df["is_hit_b"].mean()) if not df.empty else float("nan")


def metric_hr_both(df: pd.DataFrame) -> float:
    """Compute dual hit rate from a seed DataFrame."""
    return float(df["is_hit_both"].mean()) if not df.empty else float("nan")


def metric_balance(df: pd.DataFrame) -> float:
    """Compute median absolute affinity difference (balance)."""
    if df.empty:
        return float("nan")
    return float(np.median(np.abs(df["affinity_a"].to_numpy() - df["affinity_b"].to_numpy())))


def balance_fast_fn(arr: np.ndarray, indices: np.ndarray) -> float:
    """Fast balance metric for bootstrap resampling.

    Args:
        arr: Pre-computed absolute affinity differences.
        indices: Bootstrap sample indices.

    Returns:
        Median of resampled values.
    """
    return float(np.median(arr[indices]))


# ---------------------------------------------------------------------------
# Sub-computations
# ---------------------------------------------------------------------------

DataDict = dict[str, dict[str, dict[int, pd.DataFrame]]]


def compute_degradation_guard(
    pairs: dict[str, dict],
    data: DataDict,
    cliffs_delta_fn: Any,
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

    for pair_id in pairs:
        group_double = list(data.get(pair_id, {}).get("double", {}).values())
        group_single_a = list(data.get(pair_id, {}).get("single_a", {}).values())
        group_single_b = list(data.get(pair_id, {}).get("single_b", {}).values())

        pair_deg: dict[str, dict[str, float | bool]] = {}
        if group_double and group_single_a:
            vals_double = [metric_hr_a(df) for df in group_double]
            vals_single = [metric_hr_a(df) for df in group_single_a]
            delta = cliffs_delta_fn(vals_double, vals_single)
            pair_deg["hr_a_double_vs_single_a"] = {
                "cliffs_delta": delta,
                "degraded": delta <= no_degradation_thr,
            }

        if group_double and group_single_b:
            vals_double = [metric_hr_b(df) for df in group_double]
            vals_single = [metric_hr_b(df) for df in group_single_b]
            delta = cliffs_delta_fn(vals_double, vals_single)
            pair_deg["hr_b_double_vs_single_b"] = {
                "cliffs_delta": delta,
                "degraded": delta <= no_degradation_thr,
            }

        if pair_deg:
            degradation_out[pair_id] = pair_deg

    return degradation_out


def compute_sensitivity_analyses(
    pairs: dict[str, dict],
    modes: list[str],
    data: DataDict,
) -> dict[str, Any]:
    """Compute sensitivity analyses (SAP Section 9).

    These are SECONDARY analyses -- not part of primary decision rule.
    Each sub-analysis is wrapped in try/except so failures don't block
    the main pipeline.

    Args:
        pairs: Pair configuration dict.
        modes: Generation modes.
        data: Nested dict of pair_id -> mode -> seed -> DataFrame.

    Returns:
        Dict of sensitivity analysis results.
    """
    sensitivity: dict[str, Any] = {}

    # Sensitivity procedures are unstable/underpowered with very few seeds.
    max_seeds_available = max(
        (len(data.get(pair_id, {}).get(mode, {})) for pair_id in pairs for mode in modes),
        default=0,
    )
    if max_seeds_available < _MIN_SEEDS_FOR_SENSITIVITY:
        msg = (
            f"Skipping sensitivity analyses: only {max_seeds_available} seed(s) available. "
            f"Need >= {_MIN_SEEDS_FOR_SENSITIVITY} for meaningful resampling."
        )
        logger.warning(msg)
        return {
            "skipped": True,
            "reason": "insufficient_seeds",
            "max_seeds_available": max_seeds_available,
            "min_required": _MIN_SEEDS_FOR_SENSITIVITY,
        }

    try:
        from dualdocking.analysis.power_analysis import detectable_effect as _det_effect
        from dualdocking.analysis.power_analysis import sensitivity_table as _sens_table

        sensitivity["power_analysis"] = {
            "detectable_cliffs_delta_m5": _det_effect(
                n_seeds=5,
                n_simulations=_POWER_SIMULATIONS,
                n_bootstrap=_POWER_BOOTSTRAP,
            ),
            "sensitivity_table": _sens_table(
                seed_range=[3, 5, 10, 15],
                n_simulations=_POWER_SIMULATIONS,
                n_bootstrap=_POWER_BOOTSTRAP,
            ),
        }
    except Exception as exc:
        logger.warning("Power analysis failed: {}", exc)

    try:
        from dualdocking.analysis.wild_bootstrap import wild_cluster_bootstrap

        wild_results: dict[str, dict[str, Any]] = {}
        for pair_id in pairs:
            pair_wild: dict[str, Any] = {}
            for test_id, mode_b, mode_a in [("H1a", "double", "single_a"), ("H1b", "double", "single_b")]:
                group_a_dfs = list(data.get(pair_id, {}).get(mode_a, {}).values())
                group_b_dfs = list(data.get(pair_id, {}).get(mode_b, {}).values())
                if not group_a_dfs or not group_b_dfs:
                    continue
                vals_a = np.array([metric_hr_both(df) for df in group_a_dfs])
                vals_b = np.array([metric_hr_both(df) for df in group_b_dfs])
                wb = wild_cluster_bootstrap(vals_b, vals_a, n_boot=_WILD_BOOTSTRAP, alternative="less")
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

        rliable_out: dict[str, Any] = {}
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
                lo, hi = compute_iqm_ci(arr, n_boot=_RLIABLE_BOOTSTRAP)
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


def compute_summary_metrics(
    pairs: dict[str, dict],
    modes: list[str],
    seeds: list[int],
    data: DataDict,
    metrics_cfg: dict[str, Any],
    exp_cfg: dict[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute per-pair per-mode summary metrics.

    Includes hit rates, balance, diversity, objective means, and filter
    pass rates.

    Args:
        pairs: Pair configuration dict.
        modes: Generation modes.
        seeds: Seed values.
        data: Nested dict of pair_id -> mode -> seed -> DataFrame.
        metrics_cfg: Metrics configuration (thresholds etc.).
        exp_cfg: Experiment configuration.

    Returns:
        Nested dict mapping pair_id -> mode -> metric_name -> value.
    """
    from dualdocking.analysis.diversity import compute_all_diversity_metrics
    from dualdocking.scoring.objectives import parse_objectives_config

    metrics_out: dict[str, dict[str, dict[str, Any]]] = {}
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

            mode_metrics: dict[str, Any] = {
                "hr_a_mean": float(np.mean(hr_a_values)),
                "hr_a_sem": sem(hr_a_values),
                "hr_b_mean": float(np.mean(hr_b_values)),
                "hr_b_sem": sem(hr_b_values),
                "hr_both_mean": float(np.mean(hr_both_values)),
                "hr_both_sem": sem(hr_both_values),
                "balance_mean": float(np.mean(balance_values)),
                "balance_sem": sem(balance_values),
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
                    mode_metrics[f"{col}_sem"] = sem(obj_values)

            # Filtered hit rates (backward compat)
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


def compute_gnina_cross_validation(
    pairs: dict[str, dict],
    modes: list[str],
    data: DataDict,
) -> dict[str, dict[str, Any]]:
    """Compute GNINA cross-validation metrics if gnina columns are present.

    For each pair/mode, computes Spearman correlations between GNINA CNN
    affinity and Boltz-2 affinity, and between GNINA Vina scores and
    UniDock scores.

    Args:
        pairs: Pair configuration dict.
        modes: Generation modes.
        data: Nested dict of pair_id -> mode -> seed -> DataFrame.

    Returns:
        Dict mapping pair_id -> mode -> metrics dict. Empty if no GNINA data.
    """
    from scipy.stats import spearmanr

    _false = False  # named constant avoids FBT003 lint

    gnina_xval: dict[str, dict[str, Any]] = {}
    for pair_id in pairs:
        pair_gnina: dict[str, Any] = {}
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


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------


def load_all_data(
    config: ComputeMetricsConfig,
) -> tuple[DataDict, list[str]]:
    """Load all scored DataFrames, applying fairness and quality gates.

    Args:
        config: Computation configuration.

    Returns:
        Tuple of (data dict, missing_runs list).
    """
    from dualdocking.analysis.fairness import validate_fairness, validate_reward_quality
    from dualdocking.data.io_utils import stem_exists

    results_root = config.results_root if config.results_root.is_absolute() else config.project_root / config.results_root

    data: DataDict = {}

    for pair_id in config.pairs:
        thresholds = config.metrics_cfg.get("thresholds", {}).get(pair_id, {})
        thr_a = thresholds.get("target_a", 0.0)
        thr_b = thresholds.get("target_b", 0.0)

        data[pair_id] = {}
        for mode in config.modes:
            data[pair_id][mode] = {}
            for seed in config.seeds:
                run_dir = results_root / pair_id / mode / f"seed_{seed}"
                scored_stem = run_dir / "scored_joined"
                if not stem_exists(scored_stem):
                    logger.warning("Missing scored_joined: {}", scored_stem)
                    continue

                fairness = validate_fairness(run_dir)
                if not fairness["budget_consistent"]:
                    logger.warning("Fairness check failed for {}: {} — skipping", run_dir, fairness["issues"])
                    continue

                reward_quality = validate_reward_quality(
                    run_dir,
                    max_invalid_fraction=config.quality_gate.get("max_invalid_fraction", 0.10),
                    max_timeout_fraction=config.quality_gate.get("max_timeout_fraction", 0.05),
                )
                if not reward_quality["run_valid"]:
                    logger.warning("Quality gate failed for {}: {}", run_dir, reward_quality["issues"])
                    continue

                df = load_scored(scored_stem, thr_a, thr_b)
                data[pair_id][mode][seed] = df

    # Completeness check
    missing_list = [
        f"{pair_id}/{mode}/seed_{seed}"
        for pair_id in config.pairs
        for mode in config.modes
        for seed in config.seeds
        if seed not in data.get(pair_id, {}).get(mode, {})
    ]

    return data, missing_list


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComputeMetricsResult:
    """Result of compute_metrics.

    Attributes:
        metrics: Per-pair per-mode summary metrics.
        stats: Statistical test results including FDR, degradation, sensitivity.
    """

    metrics: dict[str, dict[str, dict[str, float]]]
    stats: dict[str, Any]


def compute_metrics(config: ComputeMetricsConfig) -> ComputeMetricsResult:
    """Compute per-pair metrics and run hierarchical bootstrap comparisons.

    For each core pair, runs H1a/H1b (dual hit-rate) and H2a/H2b (balance)
    tests, applies BH FDR correction on the primary family, evaluates
    the degradation guard, and computes sensitivity analyses.

    Args:
        config: Computation configuration.

    Returns:
        ComputeMetricsResult with metrics and stats dicts.

    Raises:
        SystemExit: If fairness checks fail or incomplete data is disallowed.
    """
    import sys

    from dualdocking.analysis.statistical import (
        BootstrapResult,
        cliffs_delta,
        cohens_d,
        fdr_correction,
        hierarchical_bootstrap_test,
    )

    data, missing_list = load_all_data(config)

    n_expected = len(config.pairs) * len(config.modes) * len(config.seeds)
    n_valid = n_expected - len(missing_list)
    missing_rate = 1.0 - n_valid / n_expected if n_expected else 0.0

    if missing_list:
        if config.allow_incomplete:
            logger.warning("Missing {} run(s): {}", len(missing_list), missing_list)
        else:
            logger.error("Missing {} run(s) and --no-allow-incomplete set: {}", len(missing_list), missing_list)
            sys.exit(1)

    # Summary metrics
    metrics_out = compute_summary_metrics(config.pairs, config.modes, config.seeds, data, config.metrics_cfg, config.exp_cfg)

    # Identify core pairs for FDR scope (SAP 4.2: primary family = H1a+H1b core only)
    core_pair_ids = {pid for pid, cfg in config.pairs.items() if cfg.get("set", "core") == "core"}

    # Statistical tests
    stats_out: dict[str, Any] = {}
    primary_p_values: list[float] = []
    primary_index: list[tuple[str, str]] = []

    for pair_id in config.pairs:
        stats_out[pair_id] = {}
        tests = [
            ("H1a", "double", "single_a", "hr_both", metric_hr_both, "is_hit_both", None),
            ("H1b", "double", "single_b", "hr_both", metric_hr_both, "is_hit_both", None),
            ("H2a", "double", "single_a", "balance", metric_balance, "_abs_aff_diff", balance_fast_fn),
            ("H2b", "double", "single_b", "balance", metric_balance, "_abs_aff_diff", balance_fast_fn),
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
                n_bootstrap=(config.n_bootstrap if test_id in ("H1a", "H1b") else min(config.n_bootstrap, _H2_BOOTSTRAP_CAP)),
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

    # FDR correction -- primary family only (SAP 4.2: core pairs x H1a/H1b)
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
    for pair_id in config.pairs:
        for test_id in ("H1a", "H1b", "H2a", "H2b"):
            entry = stats_out.get(pair_id, {}).get(test_id)
            if entry is None:
                continue
            if "p_value_adjusted" not in entry:
                entry["p_value_adjusted"] = None
                entry["significant"] = False

    stats_out["_degradation"] = compute_degradation_guard(config.pairs, data, cliffs_delta)

    stats_out["_completeness"] = {
        "n_expected": n_expected,
        "n_valid": n_valid,
        "missing_rate": round(missing_rate, 4),
        "missing_runs": missing_list,
    }

    stats_out["_sensitivity_analyses"] = compute_sensitivity_analyses(config.pairs, config.modes, data)
    stats_out["_gnina_cross_validation"] = compute_gnina_cross_validation(config.pairs, config.modes, data)

    return ComputeMetricsResult(metrics=metrics_out, stats=stats_out)
