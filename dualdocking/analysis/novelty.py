"""Novelty and cross-condition overlap metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import DataStructs

from dualdocking.analysis.diversity import _fingerprints


@dataclass(frozen=True, slots=True)
class NoveltyResult:
    """Result of novelty-vs-baseline computation."""

    novelty_fraction: float
    n_novel: int
    n_total: int
    n_baseline: int
    threshold: float


def novelty_vs_baseline(
    generated_smiles: list[str],
    baseline_smiles: list[str],
    *,
    threshold: float = 0.4,
) -> NoveltyResult:
    """Fraction of generated molecules dissimilar to ALL baseline molecules.

    A molecule is *novel* if its maximum Tanimoto similarity to any
    baseline molecule is below *threshold*.

    Args:
        generated_smiles: SMILES from the generative model.
        baseline_smiles: Reference SMILES (e.g., ZINC calibration set).
        threshold: Max similarity to be considered novel (default 0.4).

    Returns:
        NoveltyResult with fraction and counts.
    """
    gen_fps = _fingerprints(generated_smiles)
    base_fps = _fingerprints(baseline_smiles)

    if not gen_fps or not base_fps:
        return NoveltyResult(0.0, 0, len(gen_fps), len(base_fps), threshold)

    n_novel = 0
    for gfp in gen_fps:
        sims = DataStructs.BulkTanimotoSimilarity(gfp, base_fps)
        if max(sims) < threshold:
            n_novel += 1

    return NoveltyResult(
        novelty_fraction=n_novel / len(gen_fps),
        n_novel=n_novel,
        n_total=len(gen_fps),
        n_baseline=len(base_fps),
        threshold=threshold,
    )


def cross_condition_overlap(
    smiles_by_condition: dict[str, list[str]],
    *,
    threshold: float = 0.85,
    max_per_condition: int = 500,
    seed: int = 42,
) -> dict[tuple[str, str], float]:
    """Pairwise overlap between conditions based on Tanimoto similarity.

    For each ordered pair (A, B) where A < B lexicographically, computes
    the fraction of molecules in A that have a nearest neighbour in B
    with Tanimoto >= *threshold*.

    Args:
        smiles_by_condition: ``{condition_name: [SMILES, ...]}``.
        threshold: Similarity threshold for counting as overlap.
        max_per_condition: Subsample to this size for efficiency.
        seed: RNG seed for subsampling.

    Returns:
        Dict mapping ``(cond_a, cond_b)`` to overlap fraction (A->B direction).
    """
    rng = np.random.default_rng(seed)

    fps_by_cond: dict[str, list] = {}
    for cond, smi_list in smiles_by_condition.items():
        subset = smi_list
        if len(smi_list) > max_per_condition:
            idx = rng.choice(len(smi_list), max_per_condition, replace=False)
            subset = [smi_list[int(i)] for i in idx]
        fps_by_cond[cond] = _fingerprints(subset)

    result: dict[tuple[str, str], float] = {}
    conds = sorted(fps_by_cond.keys())
    for i, ca in enumerate(conds):
        for cb in conds[i + 1 :]:
            fps_a = fps_by_cond[ca]
            fps_b = fps_by_cond[cb]
            if not fps_a or not fps_b:
                result[(ca, cb)] = 0.0
                continue
            n_overlap = sum(1 for fpa in fps_a if max(DataStructs.BulkTanimotoSimilarity(fpa, fps_b)) >= threshold)
            result[(ca, cb)] = n_overlap / len(fps_a)

    return result
