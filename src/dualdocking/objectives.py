"""Целевые функции для двойной мишени.

Стратегия
---------
При оптимизации сразу по двум мишеням нужен один скаляр, отражающий
"хорош по *обеим* мишеням". Реализованы варианты:

Базовые (скалярные):

1. **min** - worst-case скаляризация. Для задач *минимизации*
   (меньше = лучше, как для docking/affinity) эквивалентна
   ``max(score_a, score_b)`` — оценивает по *худшей* мишени.
2. **harmonic** - гармоническое среднее - плавно штрафует дисбаланс,
   но мягче, чем ``min``.
3. **weighted_sum** - ``w * score_a + (1 - w) * score_b`` - линейная
   скаляризация; позволяет задать приоритет мишени.
4. **weighted_product** - ``score_a^w * score_b^(1-w)`` - мягче штрафует
   дисбаланс, чем взвешенная сумма. Требует положительных utility.
5. **tchebycheff** - ``max(w * |score_a - z_a|, (1-w) * |score_b - z_b|)`` -
   минимизирует максимальное взвешенное отклонение от утопической точки.
   Находит решения и на невыпуклых участках фронта Парето.

Продвинутые (множественные / Парето):

6. **pareto** - не скаляр; возвращает маску Парето-оптимальных молекул
   в двумерном пространстве скоров. Полезно для пост-анализа/отбора.
7. **hypervolume** - вклад каждой точки в гиперобъём, покрываемый
   набором относительно реферальной точки. Метрика качества фронта Парето.

Литература:
- POLYGON (Nature Comm 2024) - RL + VAE для полифармакологии
- ParetoDrug (Comm Bio 2024) - Pareto MCTS для multi-target генерации
- REINVENT 4 - multi-objective RL со скаляризацией и агрегацией
- MOBO / EHVI - Парето-оптимальный байесовский поиск

Все функции работают с numpy-массивами и векторизованы.
Конвенция по умолчанию: **lower_is_better=True** (минимизация).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ObjectiveKind(StrEnum):
    """Допустимые типы целевых функций."""

    MIN = "min"
    HARMONIC = "harmonic"
    WEIGHTED_SUM = "weighted_sum"
    GEOMETRIC = "geometric"
    WEIGHTED_PRODUCT = "weighted_product"
    TCHEBYCHEFF = "tchebycheff"


@dataclass(frozen=True)
class ObjectiveSpec:
    """Specification for a single objective function from config."""

    kind: ObjectiveKind
    column_name: str
    is_primary: bool
    params: dict[str, float] = field(default_factory=dict)


def _column_name_for(kind: ObjectiveKind, params: dict[str, float]) -> str:
    """Build column name: ``combined_{kind}`` or ``combined_{kind}_{param}``."""
    base = f"combined_{kind.value}"
    # Only include the first relevant param value for disambiguation
    for key in ("weight_a",):
        if key in params:
            base = f"{base}_{params[key]}"
    return base


def parse_objectives_config(exp_cfg: dict[str, Any]) -> list[ObjectiveSpec]:
    """Parse objectives section from experiment config into ObjectiveSpec list.

    Falls back to a single MIN objective when the section is missing or empty.
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
    """Перевести score в utility (выше = лучше)."""
    return -scores if lower_is_better else scores


def _from_utility(utility: NDArray[np.float64], lower_is_better: bool) -> NDArray[np.float64]:
    """Вернуть utility обратно в score (с исходной направленностью)."""
    return -utility if lower_is_better else utility


def combined_min(
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    *,
    lower_is_better: bool = True,
) -> NDArray[np.float64]:
    """Worst-case скаляризация.

    Если lower_is_better=True, то худшая цель = наибольший (максимальный) скор.
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
    """Гармоническое среднее (через utility).

    Shifts utilities to positive domain before computing harmonic mean
    to preserve discriminability when utilities are non-positive.
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
    """Геометрическое среднее (через utility): ``sqrt(u_a * u_b)``.

    Простейшее улучшение по сравнению с min: балансирует оба скора,
    но мягче штрафует дисбаланс. Рекомендовано как первый шаг от baseline.
    Shifts utilities to positive domain before computing geometric mean
    to preserve discriminability when utilities are non-positive.
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
    """Булева маска Парето-оптимальных (недоминируемых) точек.

    Точка *i* Парето-оптимальна, если нет другой точки *j*, которая
    доминирует её. Для minimization: ``scores_a[j] <= scores_a[i]`` **и**
    ``scores_b[j] <= scores_b[i]`` хотя бы с одним строгим неравенством.

    Сложность: O(n log n).
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
    """Взвешенная сумма: ``w * score_a + (1 - w) * score_b``.

    Простейшая скаляризация (REINVENT 4, DrugEx v3). Находит решения
    только на выпуклых участках фронта Парето.
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
    """Взвешенное произведение: ``score_a^w * score_b^(1-w)``.

    Мягче штрафует дисбаланс, чем взвешенная сумма.
    Требует строго положительных utility.
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
    """Декомпозиция Чебышёва: ``max(w*|s_a - z_a|, (1-w)*|s_b - z_b|)``.

    В отличие от взвешенной суммы, находит решения и на невыпуклых
    участках фронта Парето (MOEA/D, NSGA-III). Утопическая точка
    (utopia_a, utopia_b) - идеальные значения по каждой мишени.

    Returns non-negative deviation (0 = at utopia). Works directly in
    score-space — the ``|score - utopia|`` formulation is symmetric,
    so no utility conversion is needed.
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
    """Вклад каждой точки в 2D-гиперобъём.

    Гиперобъём (HV) - стандартная метрика качества фронта Парето
    (EHVI, MOBO). Для 2D вычисляется за O(n log n).

    Точки, доминируемые реферальной (ref_a, ref_b), получают вклад 0.
    Больший вклад = точка более ценна для разнообразия фронта.
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
    """Диспетчер: вызывает выбранную целевую функцию."""
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
