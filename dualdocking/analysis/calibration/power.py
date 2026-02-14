"""Statistical power analysis for the dual-target experiment.

Computes minimum detectable effect size (MDE) given the experimental
design (m=5 RL seeds per mode), reports Cohen's d thresholds, and
calculates the number of seeds required to detect a medium effect
(d=0.5) at 80% power.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from loguru import logger
from scipy import stats


def mde_paired_ttest(
    m: int,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "two-sided",
) -> float:
    """Minimum detectable effect size (Cohen's d) for a paired t-test.

    Uses the non-central t-distribution to find the smallest effect size
    detectable at the given alpha and power for *m* paired observations.

    Args:
        m: Number of paired observations (seed count).
        alpha: Significance level.
        power: Desired statistical power (1 - beta).
        alternative: ``"two-sided"`` or ``"one-sided"``.

    Returns:
        Minimum detectable Cohen's d.
    """
    df = m - 1
    t_crit = stats.t.ppf(1 - alpha, df) if alternative == "one-sided" else stats.t.ppf(1 - alpha / 2, df)

    lo, hi = 0.0, 20.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if alternative == "one-sided":
            achieved_power = float(stats.nct.sf(t_crit, df, mid))
        else:
            achieved_power = float(stats.nct.sf(t_crit, df, mid) + stats.nct.cdf(-t_crit, df, mid))
        if achieved_power < power:
            lo = mid
        else:
            hi = mid
    ncp = (lo + hi) / 2
    return ncp / np.sqrt(m)


def required_seeds(
    d: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "two-sided",
    max_m: int = 200,
) -> int:
    """Minimum number of seeds to detect effect size *d* at given power.

    Args:
        d: Target Cohen's d.
        alpha: Significance level.
        power: Desired power.
        alternative: ``"two-sided"`` or ``"one-sided"``.
        max_m: Upper search bound.

    Returns:
        Required number of seeds (paired observations).
    """
    for m in range(2, max_m + 1):
        if mde_paired_ttest(m, alpha, power, alternative) <= d:
            return m
    return max_m


def power_at_effect(
    m: int,
    d: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """Achieved power for a given effect size and sample count.

    Args:
        m: Number of paired observations.
        d: Cohen's d effect size.
        alpha: Significance level.
        alternative: ``"two-sided"`` or ``"one-sided"``.

    Returns:
        Statistical power (probability of rejecting H0 when H1 is true).
    """
    df = m - 1
    ncp = d * np.sqrt(m)
    if alternative == "one-sided":
        t_crit = stats.t.ppf(1 - alpha, df)
        return float(stats.nct.sf(t_crit, df, ncp))
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp))


def power_curve(
    d: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    m_range: tuple[int, int] = (3, 50),
) -> list[dict[str, float]]:
    """Power as a function of seed count for a fixed effect size.

    Args:
        d: Target Cohen's d.
        alpha: Significance level.
        alternative: Test sidedness.
        m_range: ``(min_m, max_m)`` range of seed counts.

    Returns:
        List of ``{"m": ..., "power": ...}`` dicts.
    """
    return [{"m": m, "power": power_at_effect(m, d, alpha, alternative)} for m in range(m_range[0], m_range[1] + 1)]


def cliffs_delta_to_cohens_d(delta: float) -> float:
    """Approximate conversion: Cliff's delta -> Cohen's d.

    Uses the normal-theory relation: d = delta / sqrt(1 - delta^2).
    Accurate for moderate deltas; less precise near |delta| -> 1.

    Args:
        delta: Cliff's delta (-1 to +1).

    Returns:
        Approximate Cohen's d.
    """
    delta = np.clip(delta, -0.999, 0.999)
    return float(delta / np.sqrt(1 - delta**2))


def run_power_analysis(
    project_root: Path,
    output: str = "output/data/calibration/power_analysis.json",
    seeds: int = 5,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """Run power analysis and write results to JSON.

    Args:
        project_root: Project root directory.
        output: Output JSON path relative to project_root.
        seeds: Number of RL seeds (m).
        alpha: Significance level.
        power: Desired statistical power.

    Returns:
        The power analysis manifest dict.
    """
    m = seeds

    mde_two = mde_paired_ttest(m, alpha, power, "two-sided")
    mde_one = mde_paired_ttest(m, alpha, power, "one-sided")

    effect_sizes = {
        "small": 0.2,
        "medium": 0.5,
        "large": 0.8,
    }
    seeds_required = {label: required_seeds(d, alpha, power) for label, d in effect_sizes.items()}
    power_at_m = {label: power_at_effect(m, d, alpha) for label, d in effect_sizes.items()}

    cliffs_medium = 0.33
    cliffs_degradation = -0.47
    d_from_cliffs_medium = cliffs_delta_to_cohens_d(cliffs_medium)
    d_from_cliffs_degradation = cliffs_delta_to_cohens_d(abs(cliffs_degradation))

    seeds_for_cliffs_medium = required_seeds(d_from_cliffs_medium, alpha, power)
    power_for_cliffs_medium = power_at_effect(m, d_from_cliffs_medium, alpha)

    curve = power_curve(0.5, alpha, "two-sided", (3, 50))

    manifest = {
        "date": datetime.now(UTC).date().isoformat(),
        "parameters": {
            "m_seeds": m,
            "alpha": alpha,
            "target_power": power,
        },
        "minimum_detectable_effect": {
            "two_sided_cohens_d": round(mde_two, 3),
            "one_sided_cohens_d": round(mde_one, 3),
            "interpretation": (
                f"With m={m} seeds, the experiment can only detect effects "
                f"of Cohen's d >= {mde_two:.2f} (two-sided) at alpha={alpha}, "
                f"power={power:.0%}. This is a {'large' if mde_two > 0.8 else 'very large'} "
                f"effect — smaller but practically meaningful effects will be missed."
            ),
        },
        "power_at_standard_effects": {
            label: {
                "cohens_d": d,
                "power": round(power_at_m[label], 3),
                "seeds_for_80pct_power": seeds_required[label],
            }
            for label, d in effect_sizes.items()
        },
        "cliff_delta_mapping": {
            "medium_cliff_0.33": {
                "cliffs_delta": cliffs_medium,
                "approx_cohens_d": round(d_from_cliffs_medium, 3),
                "power_at_m5": round(power_for_cliffs_medium, 3),
                "seeds_for_80pct": seeds_for_cliffs_medium,
            },
            "degradation_cliff_0.47": {
                "cliffs_delta": cliffs_degradation,
                "approx_cohens_d": round(d_from_cliffs_degradation, 3),
                "note": "Success criterion: Cliff's delta > -0.47 for no per-target degradation",
            },
        },
        "power_curve_d0.5": curve,
        "recommendations": [
            f"m={m} seeds yields very low power ({power_at_m['medium']:.1%}) for medium effects (d=0.5).",
            f"At least {seeds_required['medium']} seeds are needed for 80% power at d=0.5.",
            f"At least {seeds_required['large']} seeds are needed for 80% power at d=0.8 (large effect).",
            "Primary evidence should rely on effect size (Cliff's delta) and cross-pair consistency, not p-values.",
            "The two-level bootstrap CI should be interpreted as a rough bound, not a precise interval.",
        ],
    }

    out_path = project_root / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Power analysis saved to {}", out_path)

    return manifest
