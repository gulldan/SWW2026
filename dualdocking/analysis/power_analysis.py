"""Power analysis for Cliff's delta with hierarchical bootstrap.

Uses simulation-based power estimation for small-cluster designs.
Key reference: Cameron, Gelbach & Miller (2008) — bootstrap with m=5
clusters over-rejects; wild cluster bootstrap helps.

Cliff's delta thresholds: <0.147 negligible, <0.33 small,
<0.474 medium, >=0.474 large.
"""

from __future__ import annotations

import numpy as np


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta between two 1-D arrays."""
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def detectable_effect(
    n_seeds: int,
    alpha: float = 0.05,
    power: float = 0.80,
    n_simulations: int = 2000,
    n_molecules: int = 1000,
    n_bootstrap: int = 999,
    seed: int = 42,
) -> float:
    """Minimum detectable Cliff's delta at given n_seeds via binary search.

    Searches over candidate effect sizes and returns the smallest Cliff's
    delta for which ``simulate_power_hierarchical`` achieves the requested
    power.
    """
    lo, hi = 0.0, 1.0
    for _ in range(12):
        mid = (lo + hi) / 2
        pwr = simulate_power_hierarchical(
            n_seeds=n_seeds,
            n_molecules=n_molecules,
            effect_size=mid,
            n_simulations=n_simulations,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            seed=seed,
        )
        if pwr >= power:
            hi = mid
        else:
            lo = mid
    return round((lo + hi) / 2, 3)


def sensitivity_table(
    seed_range: list[int] | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
    n_simulations: int = 2000,
    n_molecules: int = 1000,
    n_bootstrap: int = 999,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Table of seeds vs minimum detectable Cliff's delta."""
    if seed_range is None:
        seed_range = [3, 5, 10, 15, 20]
    return [
        {
            "n_seeds": n,
            "min_cliffs_delta": detectable_effect(n, alpha, power, n_simulations, n_molecules, n_bootstrap, seed),
        }
        for n in seed_range
    ]


def simulate_power_hierarchical(
    n_seeds: int = 5,
    n_molecules: int = 1000,
    effect_size: float = 0.3,
    n_simulations: int = 2000,
    n_bootstrap: int = 999,
    alpha: float = 0.05,
    seed: int = 42,
) -> float:
    """Simulation-based power for hierarchical bootstrap test.

    Simulates *n_simulations* datasets where treatment has a known
    Cliff's delta *effect_size* over control.  For each dataset, runs a
    two-level hierarchical bootstrap and checks rejection at *alpha*.

    Returns estimated power (fraction of rejections).
    """
    rng = np.random.default_rng(seed)
    rejections = 0

    for _ in range(n_simulations):
        # Generate control/treatment: (n_seeds, n_molecules)
        control = rng.normal(0.0, 1.0, size=(n_seeds, n_molecules))
        shift = effect_size * 1.7  # empirical mapping: delta~0.3 -> shift~0.51
        treatment = rng.normal(shift, 1.0, size=(n_seeds, n_molecules))

        # Vectorized hierarchical bootstrap:
        # Level 1 — resample seeds with replacement
        idx_c = rng.integers(0, n_seeds, size=(n_bootstrap, n_seeds))
        idx_t = rng.integers(0, n_seeds, size=(n_bootstrap, n_seeds))
        # Level 2 — resample molecules within each selected seed
        mol_idx_c = rng.integers(0, n_molecules, size=(n_bootstrap, n_seeds, n_molecules))
        mol_idx_t = rng.integers(0, n_molecules, size=(n_bootstrap, n_seeds, n_molecules))

        sel_c = control[idx_c]       # (B, S, M)
        sel_t = treatment[idx_t]     # (B, S, M)
        resamp_c = np.take_along_axis(sel_c, mol_idx_c, axis=2)
        resamp_t = np.take_along_axis(sel_t, mol_idx_t, axis=2)

        diffs = resamp_t.mean(axis=2).mean(axis=1) - resamp_c.mean(axis=2).mean(axis=1)

        p_value = float(np.mean(diffs <= 0))
        if p_value < alpha:
            rejections += 1

    return rejections / n_simulations
