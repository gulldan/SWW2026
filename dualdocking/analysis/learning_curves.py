"""Learning curve extraction from scored molecule data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class QualityProgression:
    """Quality progression over generation order.

    Attributes:
        quartile_means: Mean affinity for each quartile (Q1=earliest, Q4=latest).
        improvement_ratio: ``Q4_mean / Q1_mean``.  For lower-is-better metrics
            a ratio < 1 indicates the model improved over training.
        n_molecules: Number of non-NaN molecules used.
    """

    quartile_means: dict[str, float]
    improvement_ratio: float
    n_molecules: int


def extract_learning_proxy(
    scored_df: pd.DataFrame,
    affinity_col: str = "affinity_a",
    window: int = 50,
) -> pd.DataFrame:
    """Compute rolling mean affinity as proxy for learning trajectory.

    Assumes rows in *scored_df* are ordered by generation time (first rows
    = earlier in training).

    Args:
        scored_df: DataFrame with an affinity column.
        affinity_col: Column name for affinity scores.
        window: Rolling window size.

    Returns:
        DataFrame with columns ``generation_order``, ``rolling_mean``,
        ``rolling_std``.
    """
    values = scored_df[affinity_col].to_numpy().astype(float)
    n = len(values)
    series = pd.Series(values)

    return pd.DataFrame(
        {
            "generation_order": np.arange(n),
            "rolling_mean": series.rolling(window, min_periods=1).mean().to_numpy(),
            "rolling_std": series.rolling(window, min_periods=1).std().to_numpy(),
        }
    )


def compute_quality_progression(
    scored_df: pd.DataFrame,
    affinity_col: str = "affinity_a",
) -> QualityProgression:
    """Split generation order into quartiles and compute mean affinity per quartile.

    Args:
        scored_df: DataFrame with an affinity column, ordered by generation.
        affinity_col: Column name for affinity scores.

    Returns:
        QualityProgression with quartile means and improvement ratio.
    """
    values = scored_df[affinity_col].dropna().to_numpy().astype(float)
    n = len(values)

    if n < 4:
        return QualityProgression(
            quartile_means={
                "Q1": float("nan"),
                "Q2": float("nan"),
                "Q3": float("nan"),
                "Q4": float("nan"),
            },
            improvement_ratio=float("nan"),
            n_molecules=n,
        )

    q_size = n // 4
    quartiles = {
        "Q1": float(np.mean(values[:q_size])),
        "Q2": float(np.mean(values[q_size : 2 * q_size])),
        "Q3": float(np.mean(values[2 * q_size : 3 * q_size])),
        "Q4": float(np.mean(values[3 * q_size :])),
    }

    ratio = quartiles["Q4"] / quartiles["Q1"] if quartiles["Q1"] != 0 else float("nan")

    return QualityProgression(
        quartile_means=quartiles,
        improvement_ratio=ratio,
        n_molecules=n,
    )


def extract_per_seed_curves(
    scored_df: pd.DataFrame,
    seed_column: str = "seed",
    value_column: str = "affinity_a",
    window: int = 50,
) -> dict[int | str, pd.DataFrame]:
    """Extract learning curves separately for each seed.

    Args:
        scored_df: DataFrame with a seed column and an affinity column.
        seed_column: Column name identifying the seed.
        value_column: Column name for affinity scores.
        window: Rolling window size.

    Returns:
        Dict mapping seed -> DataFrame with columns:
        generation_order, rolling_mean, rolling_std.
    """
    result: dict[int | str, pd.DataFrame] = {}
    for seed, group in scored_df.groupby(seed_column):
        seed_key = int(seed) if isinstance(seed, (int, float)) else str(seed)
        result[seed_key] = extract_learning_proxy(group.reset_index(drop=True), value_column, window)
    return result


def detect_convergence(
    rolling_mean: np.ndarray,
    patience: int = 50,
    threshold: float = 0.01,
) -> int | None:
    """Detect convergence point where rolling mean plateaus.

    Returns the index where the rolling mean stops changing by more than
    *threshold* (relative to the mean) for *patience* consecutive steps,
    or ``None`` if not converged.

    Args:
        rolling_mean: Array of rolling-mean values.
        patience: Number of consecutive steps to check.
        threshold: Relative range threshold.

    Returns:
        Index of convergence onset, or None.
    """
    if len(rolling_mean) < patience:
        return None
    for i in range(patience, len(rolling_mean)):
        window = rolling_mean[i - patience : i]
        mean_val = np.mean(window)
        if mean_val == 0:
            if np.ptp(window) == 0:
                return i - patience
            continue
        if np.ptp(window) < threshold * abs(mean_val):
            return i - patience
    return None


def extract_multi_metric_curves(
    scored_df: pd.DataFrame,
    value_columns: list[str],
    window: int = 50,
) -> dict[str, pd.DataFrame]:
    """Extract learning curves for multiple metrics simultaneously.

    Args:
        scored_df: DataFrame with metric columns.
        value_columns: Column names for metrics.
        window: Rolling window size.

    Returns:
        Dict mapping column name -> DataFrame with generation_order,
        rolling_mean, rolling_std.
    """
    return {col: extract_learning_proxy(scored_df, col, window) for col in value_columns if col in scored_df.columns}
