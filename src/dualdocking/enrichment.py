"""Docking-based enrichment metrics for virtual screening validation.

Computes AUROC, BEDROC(alpha=20), and enrichment factor to assess whether
UniDock docking scores can separate known actives from druglike decoys.

References:
    - BEDROC: Truchon & Bayly, J Chem Inf Model 2007;47(2):488-508
    - DUD-E: Mysinger et al., J Med Chem 2012;55(14):6582-94
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EnrichmentResult:
    """Result of enrichment benchmark for a single target."""

    target_id: str
    n_actives: int
    n_decoys: int
    auroc: float
    bedroc_alpha20: float
    ef_1pct: float
    ef_5pct: float
    passed: bool
    status: str  # "pass" | "fail" | "error" | "skipped:blocked"
    error: str | None = None


def compute_auroc(
    scores: NDArray,
    labels: NDArray,
    *,
    lower_is_better: bool = True,
) -> float:
    """Compute area under the ROC curve via trapezoidal rule.

    Parameters
    ----------
    scores:
        Docking scores (1-D array).
    labels:
        Binary labels (1=active, 0=decoy).
    lower_is_better:
        If True, more negative scores are considered better.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need at least one active and one decoy")

    # Sort: best scores first
    order = np.argsort(scores) if lower_is_better else np.argsort(-scores)
    sorted_labels = labels[order]

    # Walk through sorted list, accumulating TP/FP
    tp = 0
    fp = 0
    tp_prev = 0
    fp_prev = 0
    auc = 0.0
    prev_score = None

    for i in range(len(sorted_labels)):
        score_val = scores[order[i]]
        if prev_score is not None and score_val != prev_score:
            # Trapezoidal step
            auc += (fp - fp_prev) * (tp + tp_prev) / 2.0
            tp_prev = tp
            fp_prev = fp
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score_val

    # Final step
    auc += (fp - fp_prev) * (tp + tp_prev) / 2.0
    return auc / (n_pos * n_neg)


def compute_roc_curve(
    scores: NDArray,
    labels: NDArray,
    *,
    lower_is_better: bool = True,
) -> tuple[NDArray, NDArray]:
    """Compute ROC curve (FPR, TPR) arrays.

    Returns arrays suitable for matplotlib plotting. Includes (0,0) and (1,1).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need at least one active and one decoy")

    order = np.argsort(scores) if lower_is_better else np.argsort(-scores)
    sorted_labels = labels[order]

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0
    for lab in sorted_labels:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    return np.array(fpr_list), np.array(tpr_list)


def compute_bedroc(
    scores: NDArray,
    labels: NDArray,
    *,
    alpha: float = 20.0,
    lower_is_better: bool = True,
) -> float:
    """Compute BEDROC (Boltzmann-Enhanced Discrimination of ROC).

    Uses rdkit.ML.Scoring.Scoring.CalcBEDROC when available,
    with a numpy fallback implementing Truchon & Bayly (2007) eq. 6.

    Parameters
    ----------
    scores:
        Docking scores (1-D).
    labels:
        Binary labels (1=active, 0=decoy).
    alpha:
        Exponential weighting. alpha=20 places 80% weight on top 8%.
    lower_is_better:
        If True, more negative scores are considered better.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    n = len(scores)
    n_pos = int(labels.sum())
    if n_pos in (0, n):
        raise ValueError("Need at least one active and one decoy")

    # Sort by ascending score (best first for lower_is_better)
    order = np.argsort(scores) if lower_is_better else np.argsort(-scores)
    sorted_labels = labels[order]

    # Try RDKit implementation first
    try:
        from rdkit.ML.Scoring.Scoring import CalcBEDROC

        # CalcBEDROC expects list of (score, label) sorted so that
        # better scores come first (lower index = better prediction).
        # The score value itself is not used — only ordering and labels matter.
        # We pass rank as score (ascending = better).
        scored_list = [(float(i), int(sorted_labels[i])) for i in range(n)]
        return CalcBEDROC(scored_list, 1, alpha)
    except ImportError:
        pass

    # Numpy fallback: Truchon & Bayly eq. 6
    ra = float(n_pos) / n
    active_ranks = np.where(sorted_labels == 1)[0]
    ri = (active_ranks + 1) / n  # normalized ranks in (0, 1]
    s = np.sum(np.exp(-alpha * ri))

    rie_max = (1 - np.exp(-alpha * ra)) / (ra * (1 - np.exp(-alpha)))
    rie_min = (1 - np.exp(alpha * ra)) / (ra * (1 - np.exp(alpha)))
    rie_val = s / (n_pos * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1))

    denom = rie_max - rie_min
    if abs(denom) < 1e-12:
        return 0.0
    return float((rie_val - rie_min) / denom)


def compute_enrichment_factor(
    scores: NDArray,
    labels: NDArray,
    *,
    fraction: float = 0.01,
    lower_is_better: bool = True,
) -> float:
    """Compute enrichment factor at a given fraction of the ranked list.

    EF@x% = (actives_in_top_x% / n_top_x%) / (total_actives / N)

    Parameters
    ----------
    scores:
        Docking scores.
    labels:
        Binary labels (1=active, 0=decoy).
    fraction:
        Fraction of the ranked list to consider (0.01 = top 1%).
    lower_is_better:
        If True, more negative scores are considered better.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    n = len(scores)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return 0.0

    order = np.argsort(scores) if lower_is_better else np.argsort(-scores)
    sorted_labels = labels[order]

    k = max(1, int(math.ceil(n * fraction)))
    hits_in_top = int(sorted_labels[:k].sum())
    baseline_rate = n_pos / n
    top_rate = hits_in_top / k
    if baseline_rate == 0:
        return 0.0
    return top_rate / baseline_rate
