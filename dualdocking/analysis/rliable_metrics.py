"""Rliable-style robust metrics (Agarwal et al., NeurIPS 2021).

Provides interquartile mean (IQM), performance profiles, and
probability of improvement (PoI) with optional seed-level aggregation.
"""

from __future__ import annotations

import numpy as np


def compute_iqm(values: np.ndarray) -> float:
    """Interquartile mean -- mean of values in [Q1, Q3]."""
    q1, q3 = np.percentile(values, [25, 75])
    mask = (values >= q1) & (values <= q3)
    return float(np.mean(values[mask])) if mask.any() else float(np.mean(values))


def compute_iqm_ci(
    values: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI for IQM."""
    rng = np.random.default_rng(seed)
    n = len(values)
    iqms = np.empty(n_boot)
    for b in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        iqms[b] = compute_iqm(sample)
    lo = float(np.percentile(iqms, 100 * alpha / 2))
    hi = float(np.percentile(iqms, 100 * (1 - alpha / 2)))
    return lo, hi


def performance_profile(
    scores_by_method: dict[str, np.ndarray],
    thresholds: np.ndarray | None = None,
    n_points: int = 101,
) -> dict[str, np.ndarray]:
    """Fraction of runs achieving each threshold (empirical CDF).

    Returns dict mapping method name -> array of fractions at each threshold.
    Also includes ``'thresholds'`` key with the threshold array.
    """
    all_scores = np.concatenate(list(scores_by_method.values()))
    if thresholds is None:
        thresholds = np.linspace(float(np.min(all_scores)), float(np.max(all_scores)), n_points)
    result: dict[str, np.ndarray] = {"thresholds": thresholds}
    for name, scores in scores_by_method.items():
        result[name] = np.array([float(np.mean(scores <= t)) for t in thresholds])
    return result


def probability_of_improvement(
    treatment: np.ndarray,
    control: np.ndarray,
    seed_ids: np.ndarray | None = None,
) -> float:
    """P(treatment > control) from pairwise comparisons.

    If *seed_ids* is provided, values are first averaged within each
    unique seed, then PoI is computed on seed-level means.  This
    respects the clustering structure (seeds are the independent unit).

    Args:
        treatment: Array of scores for the treatment condition.
        control: Array of scores for the control condition.
        seed_ids: Optional array of seed identifiers (same length as
            treatment and control).  If provided, aggregation is done
            per-seed before computing PoI.

    Returns:
        Probability of improvement in [0, 1].
    """
    if seed_ids is not None:
        unique_seeds = np.unique(seed_ids)
        t_means = np.array([np.mean(treatment[seed_ids == s]) for s in unique_seeds])
        c_means = np.array([np.mean(control[seed_ids == s]) for s in unique_seeds])
        treatment = t_means
        control = c_means

    wins = 0
    total = len(treatment) * len(control)
    for t in treatment:
        wins += int(np.sum(t > control))
    return wins / total if total > 0 else 0.5


def probability_of_improvement_ci(
    treatment: np.ndarray,
    control: np.ndarray,
    seed_ids: np.ndarray | None = None,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI for probability of improvement.

    Resamples treatment and control arrays (respecting seed structure
    if *seed_ids* is provided) and computes PoI on each resample.

    Returns:
        Tuple of (lower, upper) CI bounds.
    """
    rng = np.random.default_rng(seed)

    if seed_ids is not None:
        unique_seeds = np.unique(seed_ids)
        t_means = np.array([np.mean(treatment[seed_ids == s]) for s in unique_seeds])
        c_means = np.array([np.mean(control[seed_ids == s]) for s in unique_seeds])
    else:
        t_means = treatment
        c_means = control

    n_t = len(t_means)
    n_c = len(c_means)
    pois = np.empty(n_boot)
    for b in range(n_boot):
        bt = t_means[rng.integers(0, n_t, size=n_t)]
        bc = c_means[rng.integers(0, n_c, size=n_c)]
        wins = sum(int(np.sum(t > bc)) for t in bt)
        total = n_t * n_c
        pois[b] = wins / total if total > 0 else 0.5

    lo = float(np.percentile(pois, 100 * alpha / 2))
    hi = float(np.percentile(pois, 100 * (1 - alpha / 2)))
    return lo, hi
