"""Tests for learning curve extraction."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from dualdocking.analysis.learning_curves import (
    QualityProgression,
    compute_quality_progression,
    detect_convergence,
    extract_learning_proxy,
    extract_multi_metric_curves,
    extract_per_seed_curves,
)


def _make_df(values: list[float], col: str = "affinity_a") -> pd.DataFrame:
    return pd.DataFrame({col: values})


class TestExtractLearningProxy:
    def test_columns_and_length(self) -> None:
        df = _make_df([1.0, 2.0, 3.0, 4.0, 5.0])
        result = extract_learning_proxy(df, window=2)
        assert list(result.columns) == ["generation_order", "rolling_mean", "rolling_std"]
        assert len(result) == 5

    def test_generation_order(self) -> None:
        df = _make_df([10.0, 20.0, 30.0])
        result = extract_learning_proxy(df, window=2)
        np.testing.assert_array_equal(result["generation_order"].values, [0, 1, 2])

    def test_monotonically_decreasing_scores(self) -> None:
        values = list(np.linspace(-2.0, -8.0, 100))
        df = _make_df(values)
        result = extract_learning_proxy(df, window=10)
        means = result["rolling_mean"].to_numpy()
        # After the window fills, rolling mean should be monotonically decreasing
        filled = means[9:]
        assert all(filled[i] >= filled[i + 1] for i in range(len(filled) - 1))

    def test_constant_values_zero_std(self) -> None:
        df = _make_df([5.0] * 20)
        result = extract_learning_proxy(df, window=5)
        stds = result["rolling_std"].to_numpy()
        # After first element, std of constant values should be 0
        assert all(s == 0.0 or math.isnan(s) for s in stds)

    def test_custom_column(self) -> None:
        df = pd.DataFrame({"my_score": [1.0, 2.0, 3.0]})
        result = extract_learning_proxy(df, affinity_col="my_score", window=2)
        assert len(result) == 3


class TestComputeQualityProgression:
    def test_improving_sequence(self) -> None:
        # Scores decrease over time (lower = better)
        values = list(np.linspace(-2.0, -8.0, 100))
        df = _make_df(values)
        result = compute_quality_progression(df)
        assert isinstance(result, QualityProgression)
        # Q4 should be more negative than Q1 → ratio > 1 (both negative, Q4/Q1 > 1)
        assert result.improvement_ratio > 1.0
        assert result.quartile_means["Q4"] < result.quartile_means["Q1"]
        assert result.n_molecules == 100

    def test_constant_values_ratio_one(self) -> None:
        df = _make_df([3.0] * 40)
        result = compute_quality_progression(df)
        assert result.improvement_ratio == pytest.approx(1.0)

    def test_too_few_molecules_nan(self) -> None:
        df = _make_df([1.0, 2.0])
        result = compute_quality_progression(df)
        assert math.isnan(result.improvement_ratio)
        assert all(math.isnan(v) for v in result.quartile_means.values())
        assert result.n_molecules == 2

    def test_zero_q1_nan_ratio(self) -> None:
        # Q1 mean = 0 should give NaN ratio
        values = [0.0] * 10 + [1.0] * 10 + [2.0] * 10 + [3.0] * 10
        df = _make_df(values)
        result = compute_quality_progression(df)
        assert math.isnan(result.improvement_ratio)

    def test_quartile_order(self) -> None:
        result = compute_quality_progression(_make_df(list(range(20))))
        assert set(result.quartile_means.keys()) == {"Q1", "Q2", "Q3", "Q4"}
        assert result.quartile_means["Q1"] < result.quartile_means["Q4"]

    def test_nan_values_dropped(self) -> None:
        values = [1.0, float("nan"), 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        df = _make_df(values)
        result = compute_quality_progression(df)
        # 7 non-NaN values
        assert result.n_molecules == 7

    def test_custom_column(self) -> None:
        df = pd.DataFrame({"score_b": list(range(20))})
        result = compute_quality_progression(df, affinity_col="score_b")
        assert result.n_molecules == 20


class TestExtractPerSeedCurves:
    def test_per_seed_separation(self) -> None:
        df = pd.DataFrame(
            {
                "seed": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "affinity_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            }
        )
        result = extract_per_seed_curves(df, window=2)
        assert len(result) == 3
        assert set(result.keys()) == {1, 2, 3}

    def test_per_seed_correct_length(self) -> None:
        df = pd.DataFrame(
            {
                "seed": [1, 1, 1, 1, 2, 2],
                "affinity_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        result = extract_per_seed_curves(df, window=2)
        assert len(result[1]) == 4
        assert len(result[2]) == 2

    def test_per_seed_columns(self) -> None:
        df = pd.DataFrame(
            {
                "seed": [1, 1, 1],
                "affinity_a": [1.0, 2.0, 3.0],
            }
        )
        result = extract_per_seed_curves(df, window=2)
        assert list(result[1].columns) == ["generation_order", "rolling_mean", "rolling_std"]

    def test_custom_column(self) -> None:
        df = pd.DataFrame(
            {
                "seed": [1, 1],
                "score_b": [1.0, 2.0],
            }
        )
        result = extract_per_seed_curves(df, value_column="score_b", window=2)
        assert 1 in result


class TestDetectConvergence:
    def test_convergence_detected(self) -> None:
        # Decreasing then flat
        rolling = np.concatenate([np.linspace(-2.0, -8.0, 100), np.full(60, -8.0)])
        idx = detect_convergence(rolling, patience=50, threshold=0.01)
        assert idx is not None
        # Should detect convergence somewhere after the flat region starts
        assert 90 <= idx <= 110

    def test_no_convergence(self) -> None:
        # Always changing
        rolling = np.linspace(0.0, 100.0, 200)
        idx = detect_convergence(rolling, patience=50, threshold=0.001)
        assert idx is None

    def test_convergence_short_series(self) -> None:
        rolling = np.array([1.0, 2.0, 3.0])
        idx = detect_convergence(rolling, patience=50, threshold=0.01)
        assert idx is None

    def test_already_converged(self) -> None:
        rolling = np.full(100, 5.0)
        idx = detect_convergence(rolling, patience=50, threshold=0.01)
        assert idx is not None
        assert idx == 0


class TestExtractMultiMetricCurves:
    def test_multi_metric_curves(self) -> None:
        df = pd.DataFrame(
            {
                "affinity_a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "affinity_b": [5.0, 4.0, 3.0, 2.0, 1.0],
            }
        )
        result = extract_multi_metric_curves(df, ["affinity_a", "affinity_b"], window=2)
        assert "affinity_a" in result
        assert "affinity_b" in result
        assert len(result["affinity_a"]) == 5

    def test_missing_column_skipped(self) -> None:
        df = pd.DataFrame({"affinity_a": [1.0, 2.0]})
        result = extract_multi_metric_curves(df, ["affinity_a", "nonexistent"], window=2)
        assert "affinity_a" in result
        assert "nonexistent" not in result

    def test_empty_columns_list(self) -> None:
        df = pd.DataFrame({"affinity_a": [1.0]})
        result = extract_multi_metric_curves(df, [], window=2)
        assert result == {}
