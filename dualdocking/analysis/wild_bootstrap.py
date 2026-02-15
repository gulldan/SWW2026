"""Wild cluster bootstrap for small-m inference.

Implements Cameron, Gelbach & Miller (2008) wild cluster bootstrap with
Rademacher weights {-1, +1}.  Default: one-sided test to match the SAP
directional hypothesis (H1: treatment > control).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class WildBootstrapResult:
    """Result of a wild cluster bootstrap test.

    Attributes:
        observed_diff: Observed difference in means (treatment - control).
        p_value: p-value from bootstrap distribution.
        ci_lower: Lower bound of confidence interval.
        ci_upper: Upper bound of confidence interval.
        n_boot: Number of bootstrap replications used.
        alternative: Direction of the test.
    """

    observed_diff: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_boot: int
    alternative: str


def wild_cluster_bootstrap(
    treatment: np.ndarray,
    control: np.ndarray,
    n_boot: int = 9999,
    alpha: float = 0.05,
    seed: int = 42,
    alternative: Literal["less", "greater", "two-sided"] = "less",
) -> WildBootstrapResult:
    """Wild cluster bootstrap test comparing treatment vs control.

    Uses Rademacher weights {-1, +1} on cluster-level residuals under
    the null (grand mean).  Designed for small number of clusters (m >= 5).

    Args:
        treatment: Array of cluster-level summary statistics.
        control: Array of cluster-level summary statistics.
        n_boot: Number of bootstrap replications.
        alpha: Significance level for CI.
        seed: RNG seed.
        alternative: Direction of the test:
            - ``"less"``: p = P(boot_diff <= 0). H1: treatment > control.
            - ``"greater"``: p = P(boot_diff >= 0). H1: treatment < control.
            - ``"two-sided"``: p = P(|boot_diff| >= |observed|).

    Returns:
        WildBootstrapResult with p-value, CI, observed difference.
    """
    rng = np.random.default_rng(seed)
    observed = float(np.mean(treatment) - np.mean(control))
    pooled = np.concatenate([treatment, control])
    grand_mean = float(np.mean(pooled))
    residuals_t = treatment - grand_mean
    residuals_c = control - grand_mean

    # Vectorized Rademacher draws (much faster than Python loop at n_boot=O(1e4)).
    w_t = rng.choice([-1.0, 1.0], size=(n_boot, len(treatment)))
    w_c = rng.choice([-1.0, 1.0], size=(n_boot, len(control)))
    boot_t_means = np.mean(grand_mean + w_t * residuals_t, axis=1)
    boot_c_means = np.mean(grand_mean + w_c * residuals_c, axis=1)
    boot_diffs = boot_t_means - boot_c_means

    if alternative == "less":
        p_value = float(np.mean(boot_diffs <= 0))
    elif alternative == "greater":
        p_value = float(np.mean(boot_diffs >= 0))
    else:
        p_value = float(np.mean(np.abs(boot_diffs) >= np.abs(observed)))

    lo = float(np.percentile(boot_diffs, 100 * alpha / 2))
    hi = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))

    return WildBootstrapResult(observed, p_value, lo, hi, n_boot, alternative)
