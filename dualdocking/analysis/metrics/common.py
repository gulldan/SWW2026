"""Shared helpers for metrics modules (computation/sensitivity/correlation)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def sem(values: list[float]) -> float:
    """Compute standard error of the mean over finite values.

    Args:
        values: Numeric observations.

    Returns:
        SEM, or ``0.0`` if fewer than two finite values are present.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def load_scored_affinity(path: Path) -> pd.DataFrame:
    """Load and normalize a ``scored_joined`` table.

    Normalization rules:
    - rename legacy ``score_a``/``score_b`` to ``affinity_a``/``affinity_b``
    - ensure ``smiles_canonical`` exists (derive from ``smiles`` when needed)
    - drop rows with empty canonical SMILES
    - deduplicate by ``smiles_canonical`` keeping the first row

    Args:
        path: Stem path (without extension) to ``scored_joined``.

    Returns:
        Normalized DataFrame with affinity columns and canonical SMILES.

    Raises:
        ValueError: If canonical SMILES cannot be resolved.
    """
    from dualdocking.data.io_utils import read_df

    df = read_df(path)
    if "affinity_a" not in df.columns and "score_a" in df.columns:
        df = df.rename(columns={"score_a": "affinity_a"})
    if "affinity_b" not in df.columns and "score_b" in df.columns:
        df = df.rename(columns={"score_b": "affinity_b"})

    if "smiles_canonical" not in df.columns:
        if "smiles" not in df.columns:
            raise ValueError(f"smiles_canonical missing in {path}")
        from dualdocking.data.chem_sanitize import canonicalize

        df["smiles_canonical"] = [canonicalize(s) or "" for s in df["smiles"].astype(str).tolist()]

    df = df[df["smiles_canonical"].astype(str).str.len() > 0]
    return df.drop_duplicates(subset=["smiles_canonical"], keep="first")


def add_hit_flags(df: pd.DataFrame, thr_a: float, thr_b: float) -> pd.DataFrame:
    """Add per-target and dual-target hit flags in-place when missing.

    Args:
        df: DataFrame with ``affinity_a`` and ``affinity_b`` columns.
        thr_a: Hit threshold for target A.
        thr_b: Hit threshold for target B.

    Returns:
        Same DataFrame object with ``is_hit_a``, ``is_hit_b``, ``is_hit_both``.
    """
    aff_a = df["affinity_a"].to_numpy()
    aff_b = df["affinity_b"].to_numpy()
    if "is_hit_a" not in df.columns:
        df["is_hit_a"] = aff_a <= thr_a
    if "is_hit_b" not in df.columns:
        df["is_hit_b"] = aff_b <= thr_b
    if "is_hit_both" not in df.columns:
        df["is_hit_both"] = (aff_a <= thr_a) & (aff_b <= thr_b)
    return df
