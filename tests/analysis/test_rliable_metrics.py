"""Tests for rliable_metrics module."""

import numpy as np

from dualdocking.analysis.rliable_metrics import (
    compute_iqm,
    compute_iqm_ci,
    performance_profile,
    probability_of_improvement,
    probability_of_improvement_ci,
)


class TestComputeIQM:
    def test_simple_values(self):
        # [1,2,3,4,5,6,7,8]: Q1=2.75, Q3=6.25 -> values in [2.75, 6.25] = [3,4,5,6]
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        iqm = compute_iqm(values)
        assert abs(iqm - 4.5) < 1e-10

    def test_single_value(self):
        assert abs(compute_iqm(np.array([5.0])) - 5.0) < 1e-10

    def test_constant_array(self):
        assert abs(compute_iqm(np.array([3.0, 3.0, 3.0, 3.0])) - 3.0) < 1e-10


class TestComputeIQMCI:
    def test_ci_contains_iqm(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        iqm = compute_iqm(values)
        lo, hi = compute_iqm_ci(values, n_boot=5000)
        assert lo <= iqm <= hi

    def test_narrow_ci_for_constant(self):
        values = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        lo, hi = compute_iqm_ci(values, n_boot=1000)
        assert abs(hi - lo) < 1e-10


class TestPerformanceProfile:
    def test_monotonic_cdf(self):
        scores = {"method": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        result = performance_profile(scores, n_points=51)
        fracs = result["method"]
        for i in range(len(fracs) - 1):
            assert fracs[i] <= fracs[i + 1] + 1e-10

    def test_starts_at_zero_ends_at_one(self):
        scores = {"A": np.array([1.0, 2.0, 3.0])}
        result = performance_profile(scores, n_points=101)
        assert result["A"][0] >= 0.0
        assert abs(result["A"][-1] - 1.0) < 1e-10

    def test_custom_thresholds(self):
        scores = {"X": np.array([1.0, 2.0, 3.0])}
        thresholds = np.array([0.0, 1.5, 3.0, 5.0])
        result = performance_profile(scores, thresholds=thresholds)
        assert len(result["thresholds"]) == 4


class TestProbabilityOfImprovement:
    def test_clear_winner(self):
        assert probability_of_improvement(np.array([10.0, 20.0]), np.array([1.0, 2.0])) == 1.0

    def test_clear_loser(self):
        assert probability_of_improvement(np.array([1.0, 2.0]), np.array([10.0, 20.0])) == 0.0

    def test_identical(self):
        assert probability_of_improvement(np.array([5.0, 5.0]), np.array([5.0, 5.0])) == 0.0

    def test_seed_level_aggregation(self):
        # With seed_ids: aggregate per seed first, then compute PoI
        treatment = np.array([1.0, 2.0, 10.0, 11.0])
        control = np.array([5.0, 6.0, 5.0, 6.0])
        seeds = np.array([0, 0, 1, 1])
        # Seed means: treatment=[1.5, 10.5], control=[5.5, 5.5]
        # PoI: 1.5>5.5=F, 1.5>5.5=F, 10.5>5.5=T, 10.5>5.5=T -> 2/4=0.5
        poi = probability_of_improvement(treatment, control, seed_ids=seeds)
        assert abs(poi - 0.5) < 1e-10

    def test_seed_ids_none_backward_compat(self):
        # Without seed_ids should behave like original
        t = np.array([3.0, 7.0])
        c = np.array([5.0])
        poi = probability_of_improvement(t, c, seed_ids=None)
        assert abs(poi - 0.5) < 1e-10


class TestProbabilityOfImprovementCI:
    def test_returns_valid_interval(self):
        t = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        c = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        lo, hi = probability_of_improvement_ci(t, c, n_boot=1000)
        assert 0.0 <= lo <= hi <= 1.0

    def test_ci_contains_point_estimate(self):
        t = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        c = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        poi = probability_of_improvement(t, c)
        lo, hi = probability_of_improvement_ci(t, c, n_boot=5000)
        assert lo <= poi <= hi

    def test_seed_level_ci(self):
        t = np.array([1.0, 2.0, 10.0, 11.0])
        c = np.array([5.0, 6.0, 5.0, 6.0])
        seeds = np.array([0, 0, 1, 1])
        lo, hi = probability_of_improvement_ci(t, c, seed_ids=seeds, n_boot=1000)
        assert 0.0 <= lo <= hi <= 1.0
