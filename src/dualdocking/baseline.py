"""Baseline utilities for ZINC calibration and stability checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dualdocking.statistical import bootstrap_ci

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ZINC_DIR = PROJECT_ROOT / "data" / "zinc"
BASELINE_DIR = PROJECT_ROOT / "data" / "baseline"


def fetch_zinc_subset(subset_name: str, n: int = 1000, seed: int = 42) -> list[str]:
    """Fetch a deterministic subset of ZINC SMILES from local cache.

    The function expects a local file: data/zinc/<subset_name>.smi
    with SMILES in the first column.
    """
    path = ZINC_DIR / f"{subset_name}.smi"
    if not path.exists():
        path = BASELINE_DIR / f"{subset_name}.smi"
    if not path.exists():
        raise FileNotFoundError(
            f"ZINC subset not found: {path}. Run scripts/fetch_zinc_baseline.sh or provide the file."
        )

    smiles = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        smiles.append(line.split()[0])

    if not smiles:
        raise ValueError(f"ZINC subset empty: {path}")

    rng = np.random.default_rng(seed)
    if n >= len(smiles):
        return smiles

    indices = rng.choice(len(smiles), size=n, replace=False)
    return [smiles[i] for i in indices]


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
