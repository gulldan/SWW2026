"""Dual-target objective functions for scalarising two-objective scores.

Scalar objectives:
    - **min**: worst-case (max of two scores when lower_is_better).
    - **harmonic**: harmonic mean via utility with positive-domain shift.
    - **geometric**: geometric mean via utility with positive-domain shift.
    - **weighted_sum**: ``w * u_a + (1-w) * u_b`` linear scalarisation.
    - **weighted_product**: ``u_a^w * u_b^(1-w)`` (requires positive utility).
    - **tchebycheff**: ``max(w*|s_a - z_a|, (1-w)*|s_b - z_b|)`` for
      non-convex Pareto fronts.

Pareto utilities:
    - **pareto_mask**: boolean mask of non-dominated points (O(n log n)).
    - **hypervolume_contributions**: per-point 2-D hypervolume contribution.

All functions are vectorised over numpy arrays.
Default convention: ``lower_is_better=True`` (minimisation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ObjectiveKind(StrEnum):
    """Enumeration of supported dual-target scalarisation methods."""

    MIN = "min"
    HARMONIC = "harmonic"
    WEIGHTED_SUM = "weighted_sum"
    GEOMETRIC = "geometric"
    WEIGHTED_PRODUCT = "weighted_product"
    TCHEBYCHEFF = "tchebycheff"


@dataclass(frozen=True)
class ObjectiveSpec:
    """Specification for a single objective function parsed from experiment config.

    Attributes:
        kind: The scalarisation method to use.
        column_name: Column name for this objective in results DataFrames.
        is_primary: Whether this is the primary (baseline) objective.
        params: Extra parameters (e.g. ``weight_a``, ``utopia_a``, ``utopia_b``).
    """

    kind: ObjectiveKind
    column_name: str
    is_primary: bool
    params: dict[str, float] = field(default_factory=dict)


def _column_name_for(kind: ObjectiveKind, params: dict[str, float]) -> str:
    """Build a DataFrame column name for the given objective kind.

    Args:
        kind: Objective scalarisation method.
        params: Parameter dict; ``weight_a`` is appended for disambiguation.

    Returns:
        Column name like ``combined_min`` or ``combined_weighted_sum_0.7``.
    """
    base = f"combined_{kind.value}"
    # Only include the first relevant param value for disambiguation
    for key in ("weight_a",):
        if key in params:
            base = f"{base}_{params[key]}"
    return base


def parse_objectives_config(exp_cfg: dict[str, Any]) -> list[ObjectiveSpec]:
    """Parse the ``objectives`` section of experiment config into specs.

    Falls back to a single MIN objective when the section is missing or empty.

    Args:
        exp_cfg: Full experiment configuration dict (expects an ``objectives`` key).

    Returns:
        Ordered list of objective specs; the first is always the primary baseline.
    """
    obj_section = exp_cfg.get("objectives")
    if not obj_section or not isinstance(obj_section, dict):
        return [ObjectiveSpec(kind=ObjectiveKind.MIN, column_name="combined_min", is_primary=True)]

    specs: list[ObjectiveSpec] = []
    seen_columns: set[str] = set()

    baseline_kind = ObjectiveKind(str(obj_section.get("baseline", "min")))
    baseline_col = _column_name_for(baseline_kind, {})
    specs.append(ObjectiveSpec(kind=baseline_kind, column_name=baseline_col, is_primary=True))
    seen_columns.add(baseline_col)

    # Alternatives
    alternatives = obj_section.get("alternatives")
    if not alternatives or not isinstance(alternatives, list):
        return specs

    for alt in alternatives:
        if not isinstance(alt, dict):
            continue
        kind = ObjectiveKind(str(alt["kind"]))
        params: dict[str, float] = {}
        for pkey in ("weight_a", "utopia_a", "utopia_b"):
            if pkey in alt:
                params[pkey] = float(alt[pkey])
        col = _column_name_for(kind, params)
        if col in seen_columns:
            continue
        specs.append(ObjectiveSpec(kind=kind, column_name=col, is_primary=False, params=params))
        seen_columns.add(col)

    return specs


def _to_utility(scores: NDArray[np.float64], lower_is_better: bool) -> NDArray[np.float64]:
    """Convert raw scores to utility space (higher is better).

    Args:
        scores: Raw objective scores.
        lower_is_better: If ``True``, negates scores so that higher = better.

    Returns:
        Utility values with the same shape as *scores*.
    """
    return -scores if lower_is_better else scores


def _from_utility(utility: NDArray[np.float64], lower_is_better: bool) -> NDArray[np.float64]:
    """Convert utility values back to the original score direction.

    Args:
        utility: Values in higher-is-better space.
        lower_is_better: If ``True``, negates back to lower-is-better scores.

    Returns:
        Scores with the original directionality.
    """
    return -utility if lower_is_better else utility


def combined_min(
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    *,
    lower_is_better: bool = True,
) -> NDArray[np.float64]:
    """Worst-case scalarisation: score determined by the worse target.

    When ``lower_is_better=True``, returns ``max(scores_a, scores_b)``
    (higher score = worse binding).

    Args:
        scores_a: Scores for target A.
        scores_b: Scores for target B.
        lower_is_better: Score direction convention.

    Returns:
        Element-wise worst score across both targets.
    """
    if lower_is_better:
        return np.maximum(scores_a, scores_b)
    return np.minimum(scores_a, scores_b)


def combined_harmonic(
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    *,
    lower_is_better: bool = True,
) -> NDArray[np.float64]:
    """Harmonic mean scalarisation via utility space.

    Shifts utilities to the positive domain before computing the harmonic
    mean to preserve discriminability when utilities are non-positive.

    Args:
        scores_a: Scores for target A.
        scores_b: Scores for target B.
        lower_is_better: Score direction convention.

    Returns:
        Harmonic-mean combined score (same direction as input).
    """
    util_a = _to_utility(scores_a, lower_is_better)
    util_b = _to_utility(scores_b, lower_is_better)
    # Shift to positive domain for harmonic mean
    shift = max(0.0, -min(np.min(util_a), np.min(util_b))) + 1.0
    sa, sb = util_a + shift, util_b + shift
    with np.errstate(divide="ignore", invalid="ignore"):
        hm = 2.0 * sa * sb / (sa + sb)
    return _from_utility(hm - shift, lower_is_better)


def combined_geometric(
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    *,
    lower_is_better: bool = True,
) -> NDArray[np.float64]:
    """Geometric mean scalarisation: ``sqrt(u_a * u_b)`` in utility space.

    Balances both scores more gently than ``combined_min``.  Shifts utilities
    to the positive domain before computing the geometric mean.

    Args:
        scores_a: Scores for target A.
        scores_b: Scores for target B.
        lower_is_better: Score direction convention.

    Returns:
        Geometric-mean combined score (same direction as input).
    """
    util_a = _to_utility(scores_a, lower_is_better)
    util_b = _to_utility(scores_b, lower_is_better)
    # Shift to positive domain for geometric mean
    shift = max(0.0, -min(np.min(util_a), np.min(util_b))) + 1.0
    sa, sb = util_a + shift, util_b + shift
    gm = np.sqrt(sa * sb)
    return _from_utility(gm - shift, lower_is_better)


def pareto_mask(
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    *,
    lower_is_better: bool = True,
) -> NDArray[np.bool_]:
    """Boolean mask of Pareto-optimal (non-dominated) points.

    Point *i* is Pareto-optimal iff no other point *j* dominates it.
    Complexity: O(n log n) via lexsort + sweep.

    Args:
        scores_a: Scores for target A.
        scores_b: Scores for target B.
        lower_is_better: Score direction convention.

    Returns:
        Boolean mask of length ``n``; ``True`` for non-dominated points.
    """
    n = len(scores_a)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask

    sa = np.asarray(scores_a, dtype=np.float64)
    sb = np.asarray(scores_b, dtype=np.float64)
    if not lower_is_better:
        sa, sb = -sa, -sb

    # Sort by first objective ascending, break ties by second ascending
    # (best sb first among ties → dominated tied points won't pass sb < best_b)
    order = np.lexsort((sb, sa))

    best_b = np.inf
    for idx in order:
        if sb[idx] <= best_b:
            mask[idx] = True
            best_b = sb[idx]
    return mask


def combined_weighted_sum(
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    weight_a: float = 0.5,
    *,
    lower_is_better: bool = True,
) -> NDArray[np.float64]:
    """Weighted-sum scalarisation: ``w * u_a + (1 - w) * u_b``.

    Finds solutions only on convex parts of the Pareto front.

    Args:
        scores_a: Scores for target A.
        scores_b: Scores for target B.
        weight_a: Weight for target A (target B gets ``1 - weight_a``).
        lower_is_better: Score direction convention.

    Returns:
        Weighted-sum combined score (same direction as input).
    """
    w = float(weight_a)
    util_a = _to_utility(scores_a, lower_is_better)
    util_b = _to_utility(scores_b, lower_is_better)
    util = w * util_a + (1.0 - w) * util_b
    return _from_utility(util, lower_is_better)


def combined_weighted_product(
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    weight_a: float = 0.5,
    *,
    lower_is_better: bool = True,
) -> NDArray[np.float64]:
    """Weighted-product scalarisation: ``u_a^w * u_b^(1-w)``.

    Softer imbalance penalty than weighted sum.  Utilities are clamped
    to zero to satisfy the positive-domain requirement.

    Args:
        scores_a: Scores for target A.
        scores_b: Scores for target B.
        weight_a: Weight for target A (target B gets ``1 - weight_a``).
        lower_is_better: Score direction convention.

    Returns:
        Weighted-product combined score (same direction as input).
    """
    w = float(weight_a)
    util_a = _to_utility(scores_a, lower_is_better)
    util_b = _to_utility(scores_b, lower_is_better)
    safe_a = np.maximum(util_a, 0.0)
    safe_b = np.maximum(util_b, 0.0)
    util = np.power(safe_a, w) * np.power(safe_b, 1.0 - w)
    return _from_utility(util, lower_is_better)


def combined_tchebycheff(
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    weight_a: float = 0.5,
    utopia_a: float = 0.0,
    utopia_b: float = 0.0,
    *,
    lower_is_better: bool = True,  # noqa: ARG001
) -> NDArray[np.float64]:
    """Tchebycheff decomposition: ``max(w*|s_a - z_a|, (1-w)*|s_b - z_b|)``.

    Finds solutions on non-convex Pareto fronts (MOEA/D, NSGA-III).
    Returns non-negative deviation (0 = at utopia).  Works directly in
    score-space (symmetric ``|score - utopia|``), so no utility conversion.

    Args:
        scores_a: Scores for target A.
        scores_b: Scores for target B.
        weight_a: Weight for target A deviation.
        utopia_a: Ideal score value for target A.
        utopia_b: Ideal score value for target B.
        lower_is_better: Unused (symmetric formulation); kept for API parity.

    Returns:
        Element-wise maximum weighted deviation from the utopia point.
    """
    w = float(weight_a)
    dev_a = w * np.abs(scores_a - utopia_a)
    dev_b = (1.0 - w) * np.abs(scores_b - utopia_b)
    return np.maximum(dev_a, dev_b)


def hypervolume_contributions(
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    ref_a: float = 0.0,
    ref_b: float = 0.0,
    *,
    lower_is_better: bool = True,
) -> NDArray[np.float64]:
    """Per-point 2-D hypervolume contribution relative to a reference point.

    Standard Pareto-front quality metric (EHVI, MOBO).  Computed in
    O(n log n).  Points dominated by the reference get contribution 0.

    Args:
        scores_a: Scores for target A.
        scores_b: Scores for target B.
        ref_a: Reference (nadir) score for target A.
        ref_b: Reference (nadir) score for target B.
        lower_is_better: Score direction convention.

    Returns:
        Non-negative contribution of each point to the hypervolume.
    """
    n = len(scores_a)
    contributions = np.zeros(n, dtype=np.float64)

    util_a = _to_utility(scores_a, lower_is_better)
    util_b = _to_utility(scores_b, lower_is_better)
    ref_u_a = float(_to_utility(np.array([ref_a], dtype=np.float64), lower_is_better)[0])
    ref_u_b = float(_to_utility(np.array([ref_b], dtype=np.float64), lower_is_better)[0])

    # Фильтруем точки, лучше реферальной (больше = лучше для обоих)
    valid = (util_a > ref_u_a) & (util_b > ref_u_b)
    if not np.any(valid):
        return contributions

    # Сортируем по score_a по убыванию
    indices = np.where(valid)[0]
    order = np.argsort(-util_a[indices])
    sorted_idx = indices[order]

    sorted_a = util_a[sorted_idx]
    sorted_b = util_b[sorted_idx]

    # Только не-доминируемые точки вносят вклад
    # Проходим слева направо (убывание a), отслеживаем макс b
    max_b_so_far = ref_u_b
    front_indices = []
    front_a = []
    front_b = []
    for k, idx in enumerate(sorted_idx):
        if sorted_b[k] > max_b_so_far:
            front_indices.append(idx)
            front_a.append(sorted_a[k])
            front_b.append(sorted_b[k])
            max_b_so_far = sorted_b[k]

    if not front_indices:
        return contributions

    fa = np.array(front_a)
    fb = np.array(front_b)
    m = len(fa)

    for k in range(m):
        # Ширина по a: от текущей до следующей (или до ref_a)
        right_a = fa[k + 1] if k + 1 < m else ref_u_a
        # Высота по b: от ref_b до текущей
        lower_b = fb[k - 1] if k > 0 else ref_u_b
        contributions[front_indices[k]] = (fa[k] - right_a) * (fb[k] - lower_b)

    return contributions


def compute_combined(
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    kind: ObjectiveKind = ObjectiveKind.MIN,
    *,
    weight_a: float = 0.5,
    utopia_a: float = 0.0,
    utopia_b: float = 0.0,
    lower_is_better: bool = True,
) -> NDArray[np.float64]:
    """Dispatch to the chosen scalarisation function by *kind*.

    Args:
        scores_a: Scores for target A.
        scores_b: Scores for target B.
        kind: Scalarisation method to apply.
        weight_a: Weight for target A (used by weighted/tchebycheff kinds).
        utopia_a: Ideal score for target A (tchebycheff only).
        utopia_b: Ideal score for target B (tchebycheff only).
        lower_is_better: Score direction convention.

    Returns:
        Combined score array with the same length as the inputs.

    Raises:
        ValueError: If *kind* is not a recognised ``ObjectiveKind``.
    """
    if kind == ObjectiveKind.MIN:
        return combined_min(scores_a, scores_b, lower_is_better=lower_is_better)
    if kind == ObjectiveKind.HARMONIC:
        return combined_harmonic(scores_a, scores_b, lower_is_better=lower_is_better)
    if kind == ObjectiveKind.GEOMETRIC:
        return combined_geometric(scores_a, scores_b, lower_is_better=lower_is_better)
    if kind == ObjectiveKind.WEIGHTED_SUM:
        return combined_weighted_sum(scores_a, scores_b, weight_a, lower_is_better=lower_is_better)
    if kind == ObjectiveKind.WEIGHTED_PRODUCT:
        return combined_weighted_product(scores_a, scores_b, weight_a, lower_is_better=lower_is_better)
    if kind == ObjectiveKind.TCHEBYCHEFF:
        return combined_tchebycheff(
            scores_a,
            scores_b,
            weight_a,
            utopia_a,
            utopia_b,
            lower_is_better=lower_is_better,
        )
    msg = f"Неизвестный тип целевой функции: {kind}"
    raise ValueError(msg)
