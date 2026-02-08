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


def hierarchical_bootstrap_test(
    scored_csvs_a: Iterable[Any],
    scored_csvs_b: Iterable[Any],
    metric_fn: Callable[[pd.DataFrame], float],
    *,
    n_bootstrap: int = 10000,
    rng_seed: int = 42,
) -> BootstrapResult:
    """Cluster bootstrap: resample seeds (pre-computed per-seed metrics).

    Parameters
    ----------
    scored_csvs_a, scored_csvs_b:
        Iterables of paths or DataFrames (one per seed).
    metric_fn:
        Function that computes scalar metric from a DataFrame.
    """
    cache: dict[Path, pd.DataFrame] = {}
    group_a = _as_dataframe_list(scored_csvs_a, cache)
    group_b = _as_dataframe_list(scored_csvs_b, cache)

    if not group_a or not group_b:
        return BootstrapResult(observed_diff=float("nan"), p_value=float("nan"), ci_lower=float("nan"), ci_upper=float("nan"))

    rng = np.random.default_rng(rng_seed)

    # Pre-compute per-seed metric values (outside the loop).
    metrics_a_pre = np.array([metric_fn(df) for df in group_a], dtype=float)
    metrics_b_pre = np.array([metric_fn(df) for df in group_b], dtype=float)
    observed_diff = float(np.nanmean(metrics_b_pre) - np.nanmean(metrics_a_pre))

    n_a = len(group_a)
    n_b = len(group_b)
    diffs = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        idx_a = rng.integers(0, n_a, size=n_a)
        idx_b = rng.integers(0, n_b, size=n_b)
        diffs[i] = float(np.nanmean(metrics_b_pre[idx_b]) - np.nanmean(metrics_a_pre[idx_a]))

    p_value = float(np.mean(np.abs(diffs) >= abs(observed_diff)))
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
