"""Tests for dualdocking.analysis.metrics.computation module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dualdocking.analysis.metrics.computation import (
    ComputeMetricsConfig,
    ComputeMetricsResult,
    balance_fast_fn,
    compute_degradation_guard,
    compute_summary_metrics,
    load_scored,
    metric_balance,
    metric_hr_a,
    metric_hr_b,
    metric_hr_both,
    sem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(
    values_a: list[float],
    values_b: list[float],
    thr_a: float = 0.0,
    thr_b: float = 0.0,
) -> pd.DataFrame:
    """Create a scored DataFrame with affinity columns and hit flags."""
    n = len(values_a)
    return pd.DataFrame(
        {
            "smiles_canonical": [f"C{'C' * i}" for i in range(n)],
            "smiles": [f"C{'C' * i}" for i in range(n)],
            "affinity_a": values_a,
            "affinity_b": values_b,
            "is_hit_a": [v <= thr_a for v in values_a],
            "is_hit_b": [v <= thr_b for v in values_b],
            "is_hit_both": [a <= thr_a and b <= thr_b for a, b in zip(values_a, values_b)],
        }
    )


# ---------------------------------------------------------------------------
# sem
# ---------------------------------------------------------------------------


class TestSem:
    def test_basic(self) -> None:
        result = sem([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = float(np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5))
        assert abs(result - expected) < 1e-10

    def test_single_value(self) -> None:
        assert sem([5.0]) == 0.0

    def test_empty(self) -> None:
        assert sem([]) == 0.0

    def test_with_nan(self) -> None:
        result = sem([1.0, float("nan"), 3.0])
        expected = float(np.std([1.0, 3.0], ddof=1) / np.sqrt(2))
        assert abs(result - expected) < 1e-10


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


class TestMetricFunctions:
    def test_hr_a(self) -> None:
        df = _make_df([-1.0, 0.5, -0.5], [-0.8, -1.2, 0.3])
        assert abs(metric_hr_a(df) - 2 / 3) < 1e-10

    def test_hr_b(self) -> None:
        df = _make_df([-1.0, 0.5, -0.5], [-0.8, -1.2, 0.3])
        assert abs(metric_hr_b(df) - 2 / 3) < 1e-10

    def test_hr_both(self) -> None:
        df = _make_df([-1.0, 0.5, -0.5], [-0.8, -1.2, 0.3])
        # both hit: only row 0 (-1.0 <= 0.0 and -0.8 <= 0.0)
        assert abs(metric_hr_both(df) - 1 / 3) < 1e-10

    def test_balance(self) -> None:
        df = _make_df([-1.0, 0.5, -0.5], [-0.8, -1.2, 0.3])
        diffs = np.abs(np.array([-1.0, 0.5, -0.5]) - np.array([-0.8, -1.2, 0.3]))
        expected = float(np.median(diffs))
        assert abs(metric_balance(df) - expected) < 1e-10

    def test_empty_df(self) -> None:
        empty = pd.DataFrame(columns=["is_hit_a", "is_hit_b", "is_hit_both", "affinity_a", "affinity_b"])
        assert np.isnan(metric_hr_a(empty))
        assert np.isnan(metric_hr_b(empty))
        assert np.isnan(metric_hr_both(empty))
        assert np.isnan(metric_balance(empty))


class TestBalanceFastFn:
    def test_basic(self) -> None:
        arr = np.array([0.2, 1.7, 0.8, 0.5, 1.0])
        indices = np.array([0, 2, 4])
        result = balance_fast_fn(arr, indices)
        assert abs(result - np.median(arr[indices])) < 1e-10


# ---------------------------------------------------------------------------
# load_scored
# ---------------------------------------------------------------------------


class TestLoadScored:
    def test_with_affinity_columns(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "smiles_canonical": ["CCO", "CCN", "CCC"],
                "affinity_a": [-1.0, 0.5, -0.5],
                "affinity_b": [-0.8, -1.2, 0.3],
            }
        )
        stem = tmp_path / "scored_joined"
        df.to_parquet(stem.with_suffix(".parquet"), index=False)

        result = load_scored(stem, thr_a=0.0, thr_b=0.0)
        assert "is_hit_a" in result.columns
        assert "is_hit_b" in result.columns
        assert "is_hit_both" in result.columns
        assert len(result) == 3

    def test_legacy_score_columns(self, tmp_path: Path) -> None:
        """Legacy score_a/score_b columns get renamed to affinity_a/affinity_b."""
        df = pd.DataFrame(
            {
                "smiles_canonical": ["CCO", "CCN"],
                "score_a": [-1.0, 0.5],
                "score_b": [-0.8, -1.2],
            }
        )
        stem = tmp_path / "scored_joined"
        df.to_parquet(stem.with_suffix(".parquet"), index=False)

        result = load_scored(stem, thr_a=0.0, thr_b=0.0)
        assert "affinity_a" in result.columns
        assert "affinity_b" in result.columns

    def test_deduplication(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "smiles_canonical": ["CCO", "CCO", "CCN"],
                "affinity_a": [-1.0, -2.0, 0.5],
                "affinity_b": [-0.8, -0.9, -1.2],
            }
        )
        stem = tmp_path / "scored_joined"
        df.to_parquet(stem.with_suffix(".parquet"), index=False)

        result = load_scored(stem, thr_a=0.0, thr_b=0.0)
        assert len(result) == 2  # deduplicated


# ---------------------------------------------------------------------------
# compute_degradation_guard
# ---------------------------------------------------------------------------


class TestDegradationGuard:
    def test_no_degradation(self) -> None:
        """Double as good as or better than single -> not degraded."""
        pairs = {"P1": {"set": "core"}}
        data = {
            "P1": {
                "double": {
                    42: _make_df([-1.0] * 50 + [0.5] * 50, [-0.8] * 50 + [0.3] * 50),
                    137: _make_df([-1.0] * 50 + [0.5] * 50, [-0.8] * 50 + [0.3] * 50),
                },
                "single_a": {
                    42: _make_df([0.5] * 80 + [-1.0] * 20, [0.5] * 100),
                    137: _make_df([0.5] * 80 + [-1.0] * 20, [0.5] * 100),
                },
                "single_b": {
                    42: _make_df([0.5] * 100, [0.5] * 80 + [-1.0] * 20),
                    137: _make_df([0.5] * 100, [0.5] * 80 + [-1.0] * 20),
                },
            }
        }

        from dualdocking.analysis.statistical import cliffs_delta

        result = compute_degradation_guard(pairs, data, cliffs_delta)
        for comparison in result.get("P1", {}).values():
            assert not comparison["degraded"]

    def test_degradation_detected(self) -> None:
        """Double much worse than single -> degraded flag set."""
        pairs = {"P1": {"set": "core"}}
        data = {
            "P1": {
                "double": {
                    42: _make_df([0.5] * 100, [0.5] * 100),  # 0% hit rate
                    137: _make_df([0.5] * 100, [0.5] * 100),
                },
                "single_a": {
                    42: _make_df([-1.0] * 100, [0.5] * 100),  # 100% hit rate
                    137: _make_df([-1.0] * 100, [0.5] * 100),
                },
            }
        }

        from dualdocking.analysis.statistical import cliffs_delta

        result = compute_degradation_guard(pairs, data, cliffs_delta)
        assert result["P1"]["hr_a_double_vs_single_a"]["degraded"]


# ---------------------------------------------------------------------------
# compute_summary_metrics
# ---------------------------------------------------------------------------


class TestComputeSummaryMetrics:
    def test_basic_metrics(self) -> None:
        pairs = {"P1": {"set": "core"}}
        modes = ["double"]
        seeds = [42]
        data = {
            "P1": {
                "double": {42: _make_df([-1.0, 0.5, -0.5], [-0.8, -1.2, 0.3])},
            }
        }
        metrics_cfg = {"thresholds": {"P1": {"target_a": 0.0, "target_b": 0.0}}}
        exp_cfg = {}

        result = compute_summary_metrics(pairs, modes, seeds, data, metrics_cfg, exp_cfg)
        assert "P1" in result
        assert "double" in result["P1"]
        m = result["P1"]["double"]
        assert "hr_a_mean" in m
        assert "hr_b_mean" in m
        assert "hr_both_mean" in m
        assert "balance_mean" in m
        assert m["n_valid_runs"] == 1
        assert m["n_expected_runs"] == 1
        assert m["missing_rate"] == 0.0

    def test_missing_runs_counted(self) -> None:
        pairs = {"P1": {"set": "core"}}
        modes = ["double"]
        seeds = [42, 137]
        data = {
            "P1": {
                "double": {42: _make_df([-1.0, 0.5], [-0.8, -1.2])},
                # seed 137 missing
            }
        }
        metrics_cfg = {"thresholds": {}}
        exp_cfg = {}

        result = compute_summary_metrics(pairs, modes, seeds, data, metrics_cfg, exp_cfg)
        m = result["P1"]["double"]
        assert m["n_valid_runs"] == 1
        assert m["n_expected_runs"] == 2
        assert m["missing_rate"] == 0.5
