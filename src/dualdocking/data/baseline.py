"""Baseline utilities for ZINC calibration and stability checks."""

from __future__ import annotations

from typing import Any

import numpy as np

from dualdocking.analysis.statistical import bootstrap_ci


def bootstrap_stability_check(scores: list[float] | np.ndarray, n_bootstrap: int = 1000) -> dict[str, Any]:
    """Check stability of mean/std via bootstrap confidence intervals."""
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci_mean": [float("nan"), float("nan")],
            "ci_std": [float("nan"), float("nan")],
            "stable_mean": False,
            "stable_std": False,
        }

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    ci_mean = bootstrap_ci(arr, lambda x: float(np.mean(x)), n_bootstrap=n_bootstrap, rng_seed=42)
    ci_std = bootstrap_ci(arr, lambda x: float(np.std(x, ddof=1)), n_bootstrap=n_bootstrap, rng_seed=42)

    mean_width = ci_mean[1] - ci_mean[0]
    std_width = ci_std[1] - ci_std[0]

    stable_mean = mean_width < max(0.10 * abs(mean), 0.5)
    stable_std = (std > 0) and (std_width / std < 0.15)

    return {
        "mean": mean,
        "std": std,
        "ci_mean": [float(ci_mean[0]), float(ci_mean[1])],
        "ci_std": [float(ci_std[0]), float(ci_std[1])],
        "stable_mean": bool(stable_mean),
        "stable_std": bool(stable_std),
    }
