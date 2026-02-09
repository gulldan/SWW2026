"""Tests for dualdocking.statistical module."""

import numpy as np
import pandas as pd
import pytest

from dualdocking.analysis.statistical import (
    BootstrapResult,
    bootstrap_ci,
    cliffs_delta,
    cohens_d,
    fdr_correction,
    hierarchical_bootstrap_test,
)


# ---------------------------------------------------------------------------
# hierarchical_bootstrap_test: two-level + one-sided p-value
# ---------------------------------------------------------------------------


def _make_df(values_a: list[float], values_b: list[float]) -> pd.DataFrame:
    """Create a scored DataFrame with affinity columns and hit flags."""
    n = len(values_a)
    return pd.DataFrame({
        "smiles_canonical": [f"C{i}" for i in range(n)],
        "affinity_a": values_a,
        "affinity_b": values_b,
        "is_hit_a": [v <= 0.0 for v in values_a],
        "is_hit_b": [v <= 0.0 for v in values_b],
        "is_hit_both": [a <= 0.0 and b <= 0.0 for a, b in zip(values_a, values_b)],
    })


def _metric_hr_both(df: pd.DataFrame) -> float:
    return float(df["is_hit_both"].mean()) if not df.empty else float("nan")


class TestTwoLevelBootstrap:
    """Verify two-level bootstrap (seeds + molecules) and one-sided p-value."""

    def test_clear_improvement_low_p(self):
        """When group B is clearly better, one-sided p-value should be near 0."""
        rng = np.random.default_rng(42)
        # Group A: low hit rate (~20%)
        group_a = [
            _make_df(
                rng.normal(0.5, 0.3, 100).tolist(),
                rng.normal(0.5, 0.3, 100).tolist(),
            )
            for _ in range(5)
        ]
        # Group B: high hit rate (~80%)
        group_b = [
            _make_df(
                rng.normal(-1.0, 0.3, 100).tolist(),
                rng.normal(-1.0, 0.3, 100).tolist(),
            )
            for _ in range(5)
        ]

        result = hierarchical_bootstrap_test(
            group_a, group_b, _metric_hr_both,
            n_bootstrap=2000, rng_seed=42,
        )
        assert isinstance(result, BootstrapResult)
        assert result.observed_diff > 0.3
        assert result.p_value < 0.05
        assert result.ci_lower > 0

    def test_no_improvement_high_p(self):
        """When group A >= group B, one-sided p-value should be high."""
        rng = np.random.default_rng(42)
        # Both groups similar
        group_a = [
            _make_df(
                rng.normal(-1.0, 0.3, 100).tolist(),
                rng.normal(-1.0, 0.3, 100).tolist(),
            )
            for _ in range(5)
        ]
        group_b = [
            _make_df(
                rng.normal(-1.0, 0.3, 100).tolist(),
                rng.normal(-1.0, 0.3, 100).tolist(),
            )
            for _ in range(5)
        ]

        result = hierarchical_bootstrap_test(
            group_a, group_b, _metric_hr_both,
            n_bootstrap=2000, rng_seed=42,
        )
        # p-value should be high (no clear advantage)
        assert result.p_value > 0.1

    def test_reversed_direction_p_near_1(self):
        """When group A is better than B, one-sided p should be near 1."""
        rng = np.random.default_rng(42)
        # Group A much better
        group_a = [
            _make_df(
                rng.normal(-1.5, 0.2, 100).tolist(),
                rng.normal(-1.5, 0.2, 100).tolist(),
            )
            for _ in range(5)
        ]
        # Group B worse
        group_b = [
            _make_df(
                rng.normal(1.0, 0.3, 100).tolist(),
                rng.normal(1.0, 0.3, 100).tolist(),
            )
            for _ in range(5)
        ]

        result = hierarchical_bootstrap_test(
            group_a, group_b, _metric_hr_both,
            n_bootstrap=2000, rng_seed=42,
        )
        assert result.observed_diff < -0.3
        assert result.p_value > 0.95

    def test_empty_groups_return_nan(self):
        result = hierarchical_bootstrap_test([], [], _metric_hr_both)
        assert np.isnan(result.observed_diff)
        assert np.isnan(result.p_value)

    def test_one_empty_group_return_nan(self):
        rng = np.random.default_rng(42)
        group = [_make_df(rng.normal(0, 1, 10).tolist(), rng.normal(0, 1, 10).tolist())]
        result = hierarchical_bootstrap_test(group, [], _metric_hr_both)
        assert np.isnan(result.p_value)

    def test_molecule_resampling_adds_variance(self):
        """CI width with 2-level should reflect molecule-level variance.

        With identical per-seed metrics but variable molecules,
        the CI should not collapse to zero width.
        """
        rng = np.random.default_rng(42)

        # Create DataFrames where per-seed means are identical
        # but individual molecules vary.
        group_a = []
        group_b = []
        for _ in range(5):
            # 50% hits in both groups — same mean, different molecules
            vals = np.concatenate([rng.normal(-1, 0.1, 50), rng.normal(1, 0.1, 50)])
            rng.shuffle(vals)
            group_a.append(_make_df(vals.tolist(), vals.tolist()))
            vals2 = np.concatenate([rng.normal(-1.2, 0.1, 60), rng.normal(1, 0.1, 40)])
            rng.shuffle(vals2)
            group_b.append(_make_df(vals2.tolist(), vals2.tolist()))

        result = hierarchical_bootstrap_test(
            group_a, group_b, _metric_hr_both,
            n_bootstrap=2000, rng_seed=42,
        )
        ci_width = result.ci_upper - result.ci_lower
        assert ci_width > 0.001, "CI should have non-trivial width from molecule resampling"

    def test_deterministic_with_same_seed(self):
        """Same rng_seed should produce identical results."""
        rng = np.random.default_rng(42)
        group_a = [_make_df(rng.normal(0, 1, 50).tolist(), rng.normal(0, 1, 50).tolist()) for _ in range(3)]
        group_b = [_make_df(rng.normal(-0.5, 1, 50).tolist(), rng.normal(-0.5, 1, 50).tolist()) for _ in range(3)]

        r1 = hierarchical_bootstrap_test(group_a, group_b, _metric_hr_both, n_bootstrap=500, rng_seed=99)
        r2 = hierarchical_bootstrap_test(group_a, group_b, _metric_hr_both, n_bootstrap=500, rng_seed=99)
        assert r1.p_value == r2.p_value
        assert r1.ci_lower == r2.ci_lower


# ---------------------------------------------------------------------------
# Existing function tests (unchanged API)
# ---------------------------------------------------------------------------


class TestFdrCorrection:
    def test_empty(self):
        assert fdr_correction([]) == []

    def test_single(self):
        assert fdr_correction([0.03]) == [0.03]

    def test_monotone(self):
        adj = fdr_correction([0.01, 0.04, 0.1])
        assert all(0 <= v <= 1 for v in adj)
        # Adjusted p-values should be >= raw
        assert adj[0] >= 0.01
        assert adj[1] >= 0.04


class TestCliffsDelta:
    def test_identical_groups(self):
        assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0

    def test_perfectly_separated(self):
        d = cliffs_delta([10, 20, 30], [1, 2, 3])
        assert d == 1.0

    def test_empty(self):
        assert np.isnan(cliffs_delta([], [1, 2]))

    def test_nan_filtered(self):
        d = cliffs_delta([1, float("nan"), 3], [0, 1, 2])
        assert np.isfinite(d)


class TestCohensD:
    def test_identical(self):
        assert np.isnan(cohens_d([5, 5, 5], [5, 5, 5]))

    def test_clear_difference(self):
        d = cohens_d([0, 1, 2], [10, 11, 12])
        assert d > 1.0

    def test_empty(self):
        assert np.isnan(cohens_d([], [1]))


class TestBootstrapCi:
    def test_basic(self):
        lo, hi = bootstrap_ci([1, 2, 3, 4, 5], np.mean, n_bootstrap=1000, rng_seed=42)
        assert lo < hi
        assert lo >= 1.0
        assert hi <= 5.0

    def test_empty(self):
        lo, hi = bootstrap_ci([], np.mean)
        assert np.isnan(lo)
        assert np.isnan(hi)


# ---------------------------------------------------------------------------
# Edge-case tests for boundary conditions and failure modes
# ---------------------------------------------------------------------------


class TestBootstrapEdgeCases:
    """Edge cases: 0-row DFs, 1-row DFs, all-NaN, metric exceptions."""

    def test_zero_row_dataframe_produces_nan(self):
        """DataFrame with 0 rows should not crash; metric returns NaN."""
        empty_df = _make_df([], [])
        group_a = [empty_df]
        group_b = [_make_df([-1.0, -2.0], [-1.0, -2.0])]
        result = hierarchical_bootstrap_test(
            group_a, group_b, _metric_hr_both,
            n_bootstrap=100, rng_seed=42,
        )
        # Should not crash; observed_diff will be nan because group_a metric is nan
        assert np.isnan(result.observed_diff)

    def test_single_row_dataframe(self):
        """DataFrame with 1 row: resampling always picks the same row."""
        one_row = _make_df([-1.0], [-1.0])  # is_hit_both = True
        group_a = [one_row, one_row, one_row]
        group_b = [_make_df([1.0], [1.0])]  # is_hit_both = False
        result = hierarchical_bootstrap_test(
            group_a, group_b, _metric_hr_both,
            n_bootstrap=200, rng_seed=42,
        )
        # group_a: all seeds have hr_both=1.0; group_b: hr_both=0.0
        # observed_diff = mean(0.0) - mean(1.0) = -1.0
        assert result.observed_diff == pytest.approx(-1.0)
        # One-sided: diff <= 0 always true
        assert result.p_value == pytest.approx(1.0)

    def test_all_nan_affinities_in_metric(self):
        """When all affinity values are NaN, metric should not crash."""
        nan_df = pd.DataFrame({
            "smiles_canonical": ["C0", "C1"],
            "affinity_a": [float("nan"), float("nan")],
            "affinity_b": [float("nan"), float("nan")],
            "is_hit_a": [False, False],
            "is_hit_b": [False, False],
            "is_hit_both": [False, False],
        })
        group_a = [nan_df]
        group_b = [_make_df([-1.0, -2.0], [-1.0, -2.0])]
        result = hierarchical_bootstrap_test(
            group_a, group_b, _metric_hr_both,
            n_bootstrap=100, rng_seed=42,
        )
        # NaN affinities produce is_hit_both=False -> hr_both=0.0 (not NaN)
        assert np.isfinite(result.observed_diff)

    def test_metric_fn_exception_propagates(self):
        """If metric_fn raises, the exception should propagate (no silent swallow)."""
        def _bad_metric(df: pd.DataFrame) -> float:
            raise ValueError("intentional test error")

        group_a = [_make_df([-1.0], [-1.0])]
        group_b = [_make_df([1.0], [1.0])]
        with pytest.raises(ValueError, match="intentional test error"):
            hierarchical_bootstrap_test(
                group_a, group_b, _bad_metric,
                n_bootstrap=10, rng_seed=42,
            )

    def test_single_seed_per_group(self):
        """With n=1 seed per group, bootstrap resampling picks the only seed."""
        group_a = [_make_df([-1.0, -2.0, 1.0], [-1.0, -2.0, 1.0])]
        group_b = [_make_df([-1.0, -2.0, -3.0], [-1.0, -2.0, -3.0])]
        result = hierarchical_bootstrap_test(
            group_a, group_b, _metric_hr_both,
            n_bootstrap=200, rng_seed=42,
        )
        assert np.isfinite(result.observed_diff)
        assert np.isfinite(result.p_value)
        # CI should still have width from molecule resampling
        assert result.ci_upper >= result.ci_lower


class TestCliffsDeltaEdgeCases:
    def test_single_element_groups(self):
        d = cliffs_delta([1.0], [2.0])
        assert d == -1.0  # 1 > 2 never, 1 < 2 always -> (0-1)/(1*1) = -1

    def test_all_nan_returns_nan(self):
        d = cliffs_delta([float("nan")], [float("nan")])
        assert np.isnan(d)

    def test_both_empty(self):
        d = cliffs_delta([], [])
        assert np.isnan(d)

    def test_all_equal(self):
        d = cliffs_delta([5, 5, 5], [5, 5, 5])
        assert d == 0.0


class TestCohensD_EdgeCases:
    def test_single_element_each(self):
        """One element per group: pooled std = 0, returns NaN."""
        d = cohens_d([1.0], [2.0])
        assert np.isnan(d)

    def test_all_nan(self):
        d = cohens_d([float("nan")], [float("nan")])
        assert np.isnan(d)


class TestFdrCorrectionEdgeCases:
    def test_all_ones(self):
        adj = fdr_correction([1.0, 1.0, 1.0])
        assert all(v == 1.0 for v in adj)

    def test_all_zeros(self):
        adj = fdr_correction([0.0, 0.0, 0.0])
        assert all(v == 0.0 for v in adj)

    def test_single_nan_propagates(self):
        """NaN in p-values: BH correction should not crash."""
        adj = fdr_correction([0.01, float("nan"), 0.05])
        # NaN comparison in argsort puts NaN at end
        assert len(adj) == 3
