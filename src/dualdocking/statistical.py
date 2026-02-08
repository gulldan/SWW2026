"""Statistical utilities for hierarchical bootstrap and effect sizes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Bootstrap summary statistics."""

    observed_diff: float
    p_value: float
    ci_lower: float
    ci_upper: float


def _as_dataframe_list(
    items: Iterable[Any],
    cache: dict[Path, pd.DataFrame],
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for item in items:
        if isinstance(item, pd.DataFrame):
            frames.append(item)
            continue
        path = Path(item)
        if path not in cache:
            cache[path] = pd.read_csv(path)
        frames.append(cache[path])
    return frames


def _column_mean(arr: np.ndarray, indices: np.ndarray) -> float:
    """Fast mean of arr[indices] — used as default fast_fn for column_mean mode."""
    return float(np.mean(arr[indices]))


def hierarchical_bootstrap_test(
    scored_csvs_a: Iterable[Any],
    scored_csvs_b: Iterable[Any],
    metric_fn: Callable[[pd.DataFrame], float],
    *,
    n_bootstrap: int = 10000,
    rng_seed: int = 42,
    fast_column: str | None = None,
    fast_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> BootstrapResult:
    """Two-level hierarchical bootstrap (goal.md lines 94-99).

    Level 1 (seeds): resample RL runs with replacement.
    Level 2 (molecules): within each resampled run, resample molecules
    with replacement before computing the metric.

    p-value: one-sided — fraction of bootstrap replicates with diff <= 0
    (goal.md line 99: "доля реплик с разностью ≤ 0").

    Parameters
    ----------
    scored_csvs_a, scored_csvs_b:
        Iterables of paths or DataFrames (one per seed).
    metric_fn:
        Function that computes scalar metric from a DataFrame.
    fast_column:
        If set, pre-extract this column as a numpy array and use numpy
        fancy indexing in the inner loop instead of df.iloc.  When set
        without *fast_fn*, computes mean(arr[indices]).
    fast_fn:
        Custom numpy-level function ``(arr, indices) -> float`` for use
        with *fast_column*.  Called with the pre-extracted array and the
        bootstrap row indices.  Ignored unless *fast_column* is set.
    """
    cache: dict[Path, pd.DataFrame] = {}
    group_a = _as_dataframe_list(scored_csvs_a, cache)
    group_b = _as_dataframe_list(scored_csvs_b, cache)

    if not group_a or not group_b:
        return BootstrapResult(observed_diff=float("nan"), p_value=float("nan"), ci_lower=float("nan"), ci_upper=float("nan"))

    rng = np.random.default_rng(rng_seed)

    # --- fast path: pre-extract numpy arrays --------------------------------
    if fast_column is not None:
        _fn = fast_fn if fast_fn is not None else _column_mean
        arrays_a = [df[fast_column].to_numpy(dtype=float, copy=False) for df in group_a]
        arrays_b = [df[fast_column].to_numpy(dtype=float, copy=False) for df in group_b]
        lens_a = np.array([len(a) for a in arrays_a])
        lens_b = np.array([len(b) for b in arrays_b])

        # Observed diff
        metrics_a_obs = np.array([_fn(arr, np.arange(len(arr))) for arr in arrays_a], dtype=float)
        metrics_b_obs = np.array([_fn(arr, np.arange(len(arr))) for arr in arrays_b], dtype=float)
        observed_diff = float(np.nanmean(metrics_b_obs) - np.nanmean(metrics_a_obs))

        n_a = len(arrays_a)
        n_b = len(arrays_b)
        diffs = np.empty(n_bootstrap, dtype=float)

        for i in range(n_bootstrap):
            idx_a = rng.integers(0, n_a, size=n_a)
            idx_b = rng.integers(0, n_b, size=n_b)

            boot_a = np.empty(n_a, dtype=float)
            for j, k in enumerate(idx_a):
                arr = arrays_a[k]
                row_idx = rng.integers(0, lens_a[k], size=lens_a[k])
                boot_a[j] = _fn(arr, row_idx)

            boot_b = np.empty(n_b, dtype=float)
            for j, k in enumerate(idx_b):
                arr = arrays_b[k]
                row_idx = rng.integers(0, lens_b[k], size=lens_b[k])
                boot_b[j] = _fn(arr, row_idx)

            diffs[i] = float(np.nanmean(boot_b) - np.nanmean(boot_a))

        p_value = float(np.mean(diffs <= 0))
        ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
        return BootstrapResult(
            observed_diff=observed_diff,
            p_value=p_value,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
        )

    # --- slow path: DataFrame-based (backward compat) -----------------------

    # Observed diff: per-seed metrics on original data.
    metrics_a_obs = np.array([metric_fn(df) for df in group_a], dtype=float)
    metrics_b_obs = np.array([metric_fn(df) for df in group_b], dtype=float)
    observed_diff = float(np.nanmean(metrics_b_obs) - np.nanmean(metrics_a_obs))

    n_a = len(group_a)
    n_b = len(group_b)
    diffs = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        # Level 1: resample seeds with replacement.
        idx_a = rng.integers(0, n_a, size=n_a)
        idx_b = rng.integers(0, n_b, size=n_b)

        # Level 2: resample molecules within each selected seed.
        boot_a = np.empty(n_a, dtype=float)
        for j, k in enumerate(idx_a):
            df = group_a[k]
            row_idx = rng.integers(0, len(df), size=len(df))
            boot_a[j] = metric_fn(df.iloc[row_idx])

        boot_b = np.empty(n_b, dtype=float)
        for j, k in enumerate(idx_b):
            df = group_b[k]
            row_idx = rng.integers(0, len(df), size=len(df))
            boot_b[j] = metric_fn(df.iloc[row_idx])

        diffs[i] = float(np.nanmean(boot_b) - np.nanmean(boot_a))

    # One-sided p-value: fraction of replicates where diff <= 0.
    p_value = float(np.mean(diffs <= 0))
    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])

    return BootstrapResult(
        observed_diff=observed_diff,
        p_value=p_value,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
    )


def fdr_correction(p_values: Iterable[float], alpha: float = 0.05) -> list[float]:
    """Benjamini-Hochberg FDR correction."""
    _ = alpha
    pvals = np.asarray(list(p_values), dtype=float)
    n = len(pvals)
    if n == 0:
        return []

    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(n, dtype=float)

    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = (ranked[i] * n) / rank
        prev = min(prev, val)
        adj[i] = prev

    adj = np.clip(adj, 0.0, 1.0)
    # Return in original order
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out.tolist()


def cliffs_delta(group_a: Iterable[float], group_b: Iterable[float]) -> float:
    """Cliff's delta effect size."""
    a = np.asarray(list(group_a), dtype=float)
    b = np.asarray(list(group_b), dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    n_greater = np.sum(diff > 0)
    n_less = np.sum(diff < 0)
    return float((n_greater - n_less) / (len(a) * len(b)))


def cohens_d(group_a: Iterable[float], group_b: Iterable[float]) -> float:
    """Cohen's d effect size (pooled standard deviation)."""
    a = np.asarray(list(group_a), dtype=float)
    b = np.asarray(list(group_b), dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    var_a = np.var(a, ddof=1) if len(a) > 1 else 0.0
    var_b = np.var(b, ddof=1) if len(b) > 1 else 0.0
    pooled = np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2)) if (len(a) + len(b) - 2) > 0 else 0.0
    if pooled == 0:
        return float("nan")
    return float((mean_b - mean_a) / pooled)


def bootstrap_ci(
    data: Iterable[float],
    statistic_fn: Callable[[np.ndarray], float],
    *,
    n_bootstrap: int = 10000,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval."""
    arr = np.asarray(list(data), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(rng_seed)
    stats = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, len(arr), size=len(arr))
        stats[i] = statistic_fn(arr[idx])
    lower, upper = np.percentile(stats, [2.5, 97.5])
    return float(lower), float(upper)
