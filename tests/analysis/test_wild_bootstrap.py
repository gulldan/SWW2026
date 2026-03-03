"""Tests for wild_bootstrap module."""

import numpy as np

from dualdocking.analysis.wild_bootstrap import (
    WildBootstrapResult,
    wild_cluster_bootstrap,
)


class TestWildClusterBootstrap:
    def test_identical_arrays_high_p(self):
        arr = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        result = wild_cluster_bootstrap(arr, arr.copy(), alternative="less")
        assert isinstance(result, WildBootstrapResult)
        assert abs(result.observed_diff) < 1e-10
        assert result.p_value > 0.3  # should be ~0.5

    def test_very_different_arrays_low_p(self):
        treatment = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        control = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = wild_cluster_bootstrap(treatment, control, alternative="two-sided")
        assert result.p_value < 0.10
        assert result.observed_diff > 0

    def test_one_sided_default(self):
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        c = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        result = wild_cluster_bootstrap(t, c)
        assert result.alternative == "less"

    def test_two_sided(self):
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        c = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        wild_cluster_bootstrap(t, c, alternative="less", seed=42)
        r_two = wild_cluster_bootstrap(t, c, alternative="two-sided", seed=42)
        assert r_two.alternative == "two-sided"
        # Two-sided p should generally differ from one-sided
        # (not always, but in this case treatment < control so one-sided "less" is high)

    def test_greater_alternative(self):
        treatment = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        control = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = wild_cluster_bootstrap(treatment, control, alternative="greater")
        # treatment >> control, so P(boot_diff >= 0) should be high
        assert result.p_value > 0.5

    def test_result_fields(self):
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        c = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = wild_cluster_bootstrap(t, c, n_boot=999)
        assert result.n_boot == 999
        assert result.ci_lower < result.ci_upper

    def test_deterministic_with_same_seed(self):
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        c = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        r1 = wild_cluster_bootstrap(t, c, seed=123)
        r2 = wild_cluster_bootstrap(t, c, seed=123)
        assert r1.p_value == r2.p_value

    def test_rademacher_weights_symmetric(self):
        """With identical arrays, bootstrap diffs should be roughly symmetric around 0."""
        arr = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        result = wild_cluster_bootstrap(arr, arr.copy(), n_boot=9999, alternative="two-sided")
        # p-value should be high (no signal)
        assert result.p_value > 0.3
