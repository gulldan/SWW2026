"""Tests for mixed_effects module."""

import math

import numpy as np
import pandas as pd

from dualdocking.analysis.mixed_effects import (
    MixedModelResult,
    fit_mixed_model,
)


def _make_synthetic_df(
    n_modes: int = 3,
    n_pairs: int = 3,
    n_seeds: int = 5,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic seed-level DataFrame for testing."""
    rng = np.random.default_rng(rng_seed)
    modes = [f"mode_{i}" for i in range(n_modes)]
    pairs = [f"pair_{i}" for i in range(n_pairs)]
    rows = []
    for mode in modes:
        for pair in pairs:
            for _ in range(n_seeds):
                rows.append(
                    {
                        "mode": mode,
                        "pair": pair,
                        "hr_both": rng.normal(0.3, 0.1),
                    }
                )
    return pd.DataFrame(rows)


class TestFitMixedModel:
    def test_returns_mixed_model_result(self):
        df = _make_synthetic_df()
        result = fit_mixed_model(df)
        assert isinstance(result, MixedModelResult)

    def test_converged_result_has_coefficients(self):
        df = _make_synthetic_df()
        result = fit_mixed_model(df)
        if result.converged:
            assert len(result.coefficients) > 0
            assert len(result.p_values) > 0

    def test_n_obs_correct(self):
        df = _make_synthetic_df(n_modes=2, n_pairs=2, n_seeds=5)
        result = fit_mixed_model(df)
        assert result.n_obs == 20

    def test_n_groups_correct(self):
        df = _make_synthetic_df(n_modes=2, n_pairs=3, n_seeds=5)
        result = fit_mixed_model(df)
        assert result.n_groups == 3

    def test_custom_column_names(self):
        df = _make_synthetic_df()
        df = df.rename(columns={"hr_both": "score", "mode": "condition", "pair": "cluster"})
        result = fit_mixed_model(df, outcome="score", fixed="condition", group="cluster")
        assert isinstance(result, MixedModelResult)
        assert result.n_obs == len(df)


class TestUnconvergedResult:
    def test_statsmodels_import_failure(self, monkeypatch):
        """When statsmodels is missing, returns unconverged with NaN p-values."""
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("statsmodels"):
                raise ImportError("mocked: no statsmodels")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        df = _make_synthetic_df()
        result = fit_mixed_model(df)
        assert isinstance(result, MixedModelResult)
        assert not result.converged
        assert result.aic is None
        assert len(result.p_values) == 0  # empty, not fake p=1.0

    def test_unconverged_has_nan_re_var(self, monkeypatch):
        """Unconverged result has NaN random effects variance."""
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("statsmodels"):
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        df = _make_synthetic_df()
        result = fit_mixed_model(df)
        assert math.isnan(result.random_effects_var)
