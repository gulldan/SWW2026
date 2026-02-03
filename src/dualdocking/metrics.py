"""Метрики двойных хитов и статистические сравнения.

Вычисляет hit-rate, enrichment@K и статистические тесты single vs double.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats as sp_stats

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class DualHitResult:
    """Сводка анализа двойных хитов."""

    n_total: int
    n_hit_a: int
    n_hit_b: int
    n_hit_both: int

    @property
    def hit_rate_a(self) -> float:
        """Доля молекул, являющихся хитами по мишени A."""
        return self.n_hit_a / self.n_total if self.n_total else 0.0

    @property
    def hit_rate_b(self) -> float:
        """Доля молекул, являющихся хитами по мишени B."""
        return self.n_hit_b / self.n_total if self.n_total else 0.0

    @property
    def dual_hit_rate(self) -> float:
        """Доля молекул, являющихся хитами одновременно по A и B."""
        return self.n_hit_both / self.n_total if self.n_total else 0.0

    def to_dict(self) -> dict:
        """Сериализовать результат в словарь для отчёта."""
        return {
            "n_total": self.n_total,
            "n_hit_a": self.n_hit_a,
            "n_hit_b": self.n_hit_b,
            "n_hit_both": self.n_hit_both,
            "hit_rate_a": round(self.hit_rate_a, 4),
            "hit_rate_b": round(self.hit_rate_b, 4),
            "dual_hit_rate": round(self.dual_hit_rate, 4),
        }


def compute_dual_hits(
    scores_a: NDArray[np.float64],
    scores_b: NDArray[np.float64],
    threshold_a: float,
    threshold_b: float,
) -> DualHitResult:
    """Подсчёт single- и dual-target хитов.

    Соглашение: молекула является *хитом*, если её скор **<=** порога
    (чем ниже, тем лучше - типично для docking-скоров / энергий связывания).
    """
    valid = np.isfinite(scores_a) & np.isfinite(scores_b)
    if not np.any(valid):
        return DualHitResult(n_total=0, n_hit_a=0, n_hit_b=0, n_hit_both=0)

    scores_a = scores_a[valid]
    scores_b = scores_b[valid]

    hit_a = scores_a <= threshold_a
    hit_b = scores_b <= threshold_b
    hit_both = hit_a & hit_b
    return DualHitResult(
        n_total=len(scores_a),
        n_hit_a=int(hit_a.sum()),
        n_hit_b=int(hit_b.sum()),
        n_hit_both=int(hit_both.sum()),
    )


def enrichment_at_k(
    scores: NDArray[np.float64],
    is_active: NDArray[np.bool_],
    k: int,
) -> float:
    """Фактор обогащения при top-K.

    EF@K = (хиты_в_top_k / k) / (всего_хитов / N)
    """
    if k <= 0:
        return 0.0
    valid = np.isfinite(scores)
    scores = scores[valid]
    is_active = is_active[valid]
    n = len(scores)
    if n == 0:
        return 0.0
    k = min(k, n)

    order = np.argsort(scores)  # по возрастанию = лучшие первыми для докинга
    top_k = is_active[order[:k]]
    hits_top_k = top_k.sum()
    total_hits = is_active.sum()
    if total_hits == 0 or k == 0:
        return 0.0
    expected_rate = total_hits / n
    observed_rate = hits_top_k / k
    return float(observed_rate / expected_rate)


def compare_distributions(
    single_scores: NDArray[np.float64],
    double_scores: NDArray[np.float64],
) -> dict:
    """KS-тест и Mann-Whitney U между single и double запусками."""
    single_scores = single_scores[np.isfinite(single_scores)]
    double_scores = double_scores[np.isfinite(double_scores)]

    n1, n2 = len(single_scores), len(double_scores)
    if n1 == 0 or n2 == 0:
        return {
            "ks_statistic": float("nan"),
            "ks_pvalue": float("nan"),
            "mannwhitney_statistic": float("nan"),
            "mannwhitney_pvalue": float("nan"),
            "rank_biserial_r": float("nan"),
            "n_single": n1,
            "n_double": n2,
        }

    ks_stat, ks_p = sp_stats.ks_2samp(single_scores, double_scores)
    mw_stat, mw_p = sp_stats.mannwhitneyu(single_scores, double_scores, alternative="two-sided")

    # Размер эффекта: ранговая бисериальная корреляция
    r = 1 - (2 * mw_stat) / (n1 * n2)

    return {
        "ks_statistic": round(float(ks_stat), 4),
        "ks_pvalue": float(ks_p),
        "mannwhitney_statistic": float(mw_stat),
        "mannwhitney_pvalue": float(mw_p),
        "rank_biserial_r": round(float(r), 4),
        "n_single": n1,
        "n_double": n2,
    }
