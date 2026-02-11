"""Mixed-effects model for hierarchical experimental data.

Sensitivity analysis: fits ``outcome ~ C(mode) + (1|pair)`` using
statsmodels MixedLM with REML.  With m=5 seeds per group, random
effects estimates will have wide CIs -- this is intended as exploratory
sensitivity analysis, not primary inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger


@dataclass(frozen=True, slots=True)
class MixedModelResult:
    """Result of mixed-effects model fit.

    Attributes:
        coefficients: Fixed-effect coefficient estimates keyed by term name.
        std_errors: Standard errors for each coefficient.
        p_values: p-values for each coefficient (NaN if not converged).
        random_effects_var: Estimated variance of the random intercept.
        n_groups: Number of groups (clusters).
        n_obs: Total number of observations.
        aic: Akaike information criterion (None if unavailable).
        converged: Whether the model converged.
        summary_text: Human-readable summary string.
    """

    coefficients: dict[str, float]
    std_errors: dict[str, float]
    p_values: dict[str, float]
    random_effects_var: float
    n_groups: int
    n_obs: int
    aic: float | None
    converged: bool
    summary_text: str


def fit_mixed_model(
    df: pd.DataFrame,
    outcome: str = "hr_both",
    fixed: str = "mode",
    group: str = "pair",
) -> MixedModelResult:
    """Fit a linear mixed-effects model: outcome ~ C(fixed) + (1|group).

    Uses statsmodels MixedLM with REML estimation.  If the fit fails or
    statsmodels is unavailable, returns a result with ``converged=False``
    and NaN p-values -- the caller decides how to handle this.

    Args:
        df: DataFrame with columns [outcome, fixed, group].
            Should be seed-level aggregates.
        outcome: Response variable column name.
        fixed: Fixed effect column name (categorical).
        group: Grouping variable for random intercept.

    Returns:
        MixedModelResult with coefficients, p-values, random effects variance.
    """
    n_groups = df[group].nunique()
    n_obs = len(df)

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        logger.warning("statsmodels not installed; returning unconverged result")
        return _unconverged_result(n_groups, n_obs, "statsmodels not installed")

    formula = f"{outcome} ~ C({fixed})"
    try:
        model = smf.mixedlm(formula, df, groups=df[group])
        result = model.fit(reml=True)

        coefs = {str(k): float(v) for k, v in result.fe_params.items()}
        ses = {str(k): float(v) for k, v in result.bse_fe.items()}
        pvals = {str(k): float(v) for k, v in result.pvalues.items()}
        re_var = float(result.cov_re.iloc[0, 0]) if hasattr(result.cov_re, "iloc") else float(result.cov_re)

        return MixedModelResult(
            coefficients=coefs,
            std_errors=ses,
            p_values=pvals,
            random_effects_var=re_var,
            n_groups=n_groups,
            n_obs=n_obs,
            aic=float(result.aic) if hasattr(result, "aic") else None,
            converged=result.converged,
            summary_text=str(result.summary()),
        )
    except Exception as exc:
        logger.warning("MixedLM failed: {}", exc)
        return _unconverged_result(n_groups, n_obs, str(exc))


def _unconverged_result(
    n_groups: int,
    n_obs: int,
    reason: str,
) -> MixedModelResult:
    """Return a result with converged=False and NaN p-values."""
    return MixedModelResult(
        coefficients={},
        std_errors={},
        p_values={},
        random_effects_var=float("nan"),
        n_groups=n_groups,
        n_obs=n_obs,
        aic=None,
        converged=False,
        summary_text=f"Model did not converge: {reason}",
    )
