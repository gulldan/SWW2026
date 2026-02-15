"""Threshold sensitivity analysis (SCI-06 / KVA-270).

Repeats the primary hit-rate analysis at multiple threshold settings to check
robustness of the primary conclusion.  Pre-registered thresholds::

    T = -1.0 (100 nM), 0.0 (1 uM), +1.0 (10 uM)

Outputs structured dicts and flat rows suitable for CSV/Parquet and JSON.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from dualdocking.analysis.metrics.common import load_scored_affinity
from dualdocking.analysis.metrics.common import sem as _sem_impl

_N_BOOTSTRAP = 10_000

DEFAULT_THRESHOLDS = [-1.0, 0.0, 1.0]


@dataclass(frozen=True)
class SensitivityConfig:
    """Configuration for threshold sensitivity analysis.

    Attributes:
        project_root: Absolute path to project root.
        pairs: Pair configuration dict (pair_id -> config).
        modes: Generation modes.
        seeds: Random seeds.
        metrics_cfg: Parsed metrics.yaml contents.
        exp_cfg: Parsed experiment.yaml contents.
        quality_gate: Quality gate thresholds dict.
        results_root: Root results directory.
        thresholds: Threshold values to sweep.
        allow_incomplete: Allow missing pair-mode-seed combinations.
        n_bootstrap: Number of bootstrap resamples.
    """

    project_root: Path
    pairs: dict[str, dict]
    modes: list[str]
    seeds: list[int]
    metrics_cfg: dict[str, Any]
    exp_cfg: dict[str, Any]
    quality_gate: dict[str, float] = field(default_factory=dict)
    results_root: Path = Path("output/results")
    thresholds: list[float] = field(default_factory=lambda: list(DEFAULT_THRESHOLDS))
    allow_incomplete: bool = True
    n_bootstrap: int = _N_BOOTSTRAP


@dataclass(frozen=True)
class SensitivityResult:
    """Result of threshold sensitivity analysis.

    Attributes:
        json_out: Structured results dict (thresholds_tested, results, robustness_summary).
        rows: Flat list of dicts for CSV/Parquet output.
    """

    json_out: dict[str, Any]
    rows: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sem(values: list[float]) -> float:
    """Compute standard error of the mean."""
    return _sem_impl(values)


def _load_scored(path: Path) -> pd.DataFrame:
    """Load scored data, keeping raw affinity columns only."""
    return load_scored_affinity(path)


def compute_hr_both(df: pd.DataFrame, thr: float) -> float:
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_sensitivity(config: SensitivityConfig) -> SensitivityResult:
    """Run threshold sensitivity analysis across pre-registered thresholds.

    Sweeps multiple hit-rate thresholds, re-runs the H1a/H1b bootstrap tests
    at each threshold, applies FDR correction, and evaluates robustness.

    Args:
        config: Sensitivity analysis configuration.

    Returns:
        SensitivityResult with structured JSON and flat CSV data.

    Raises:
        SystemExit: If fairness checks fail or incomplete data is disallowed.
    """
    import sys

    from dualdocking.analysis.fairness import validate_fairness, validate_reward_quality
    from dualdocking.analysis.statistical import (
        cliffs_delta,
        cohens_d,
        fdr_correction,
        hierarchical_bootstrap_test,
    )
    from dualdocking.data.io_utils import stem_exists

    results_root = (
        config.results_root
        if config.results_root.is_absolute()
        else config.project_root / config.results_root
    )

    # ---- Load all DataFrames (threshold-agnostic) ----
    raw_data: dict[str, dict[str, dict[int, pd.DataFrame]]] = {}
    for pair_id in config.pairs:
        raw_data[pair_id] = {}
        for mode in config.modes:
            raw_data[pair_id][mode] = {}
            for seed in config.seeds:
                run_dir = results_root / pair_id / mode / f"seed_{seed}"
                scored_stem = run_dir / "scored_joined"
                if not stem_exists(scored_stem):
                    continue

                fairness = validate_fairness(run_dir)
                if not fairness["budget_consistent"]:
                    logger.error("Fairness fail: {}", run_dir)
                    sys.exit(1)

                rq = validate_reward_quality(
                    run_dir,
                    max_invalid_fraction=config.quality_gate.get("max_invalid_fraction", 0.10),
                    max_timeout_fraction=config.quality_gate.get("max_timeout_fraction", 0.05),
                )
                if not rq["run_valid"]:
                    logger.warning("Quality gate fail: {}", run_dir)
                    continue

                df = _load_scored(scored_stem)
                # Drop NaN affinity rows so binary hit column is clean for
                # the Binomial fast path in hierarchical_bootstrap_test.
                valid = df["affinity_a"].notna() & df["affinity_b"].notna()
                raw_data[pair_id][mode][seed] = df[valid].copy() if not valid.all() else df

    # Completeness check
    missing_list = [
        f"{pair_id}/{mode}/seed_{seed}"
        for pair_id in config.pairs
        for mode in config.modes
        for seed in config.seeds
        if seed not in raw_data.get(pair_id, {}).get(mode, {})
    ]
    if missing_list:
        if config.allow_incomplete:
            logger.warning("Missing {} run(s): {}", len(missing_list), missing_list)
        else:
            logger.error("Missing {} run(s) and --no-allow-incomplete: {}", len(missing_list), missing_list)
            sys.exit(1)

    # ---- Sweep thresholds ----
    rows: list[dict[str, Any]] = []
    json_out: dict[str, Any] = {"thresholds_tested": config.thresholds, "results": {}}

    _hit_col = "_is_hit_both_thr"

    for thr in config.thresholds:
        thr_key = str(thr)
        json_out["results"][thr_key] = {}

        def _make_metric_fn(threshold: float) -> Callable[[pd.DataFrame], float]:
            def _fn(frame: pd.DataFrame) -> float:
                return compute_hr_both(frame, threshold)

            return _fn

        metric_fn = _make_metric_fn(thr)

        # Pre-compute binary hit column for Binomial fast path
        for pair_id in config.pairs:
            for mode in config.modes:
                for df in raw_data.get(pair_id, {}).get(mode, {}).values():
                    aff_a = df["affinity_a"].to_numpy()
                    aff_b = df["affinity_b"].to_numpy()
                    df[_hit_col] = ((aff_a <= thr) & (aff_b <= thr)).astype(float)

        p_values: list[float] = []
        p_index: list[tuple[str, str]] = []

        for pair_id in config.pairs:
            pair_stats: dict[str, Any] = {}
            pair_metrics: dict[str, Any] = {}

            for mode in config.modes:
                seed_dfs = raw_data.get(pair_id, {}).get(mode, {})
                if not seed_dfs:
                    continue

                hr_both_vals = [float(df[_hit_col].mean()) for df in seed_dfs.values()]
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
                    n_bootstrap=config.n_bootstrap,
                    rng_seed=42,
                    fast_column=_hit_col,
                )
                vals_ctrl = [float(df[_hit_col].mean()) for df in group_ctrl]
                vals_treat = [float(df[_hit_col].mean()) for df in group_treat]

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

        # Flatten to rows
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
                for r in rows:
                    if r["threshold"] == thr and r["pair"] == pair_id and r["mode"] == "double":
                        r[f"{test_id}_diff"] = s["observed_diff"]
                        r[f"{test_id}_p_adj"] = s.get("p_value_adjusted", s["p_value_raw"])
                        r[f"{test_id}_cliffs_d"] = s["cliffs_delta"]
                        r[f"{test_id}_sig"] = s.get("significant", False)

    # ---- Robustness summary ----
    # Absent "set" defaults to "core" (same convention as compute_metrics/audit).
    core_pairs = [p for p, cfg in config.pairs.items() if cfg.get("set", "core") == "core"]
    summary: dict[str, dict[str, Any]] = {}
    for thr in config.thresholds:
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

    return SensitivityResult(json_out=json_out, rows=rows)
