"""Tests for dual-target objective functions."""

import numpy as np

from dualdocking.objectives import (
    ObjectiveKind,
    combined_geometric,
    combined_harmonic,
    combined_min,
    combined_tchebycheff,
    combined_weighted_product,
    combined_weighted_sum,
    compute_combined,
    hypervolume_contributions,
    pareto_mask,
)


def test_combined_min_basic():
    a = np.array([-8.0, -5.0, -3.0])
    b = np.array([-7.0, -6.0, -4.0])
    result = combined_min(a, b)
    # worst-case for minimization = max
    np.testing.assert_array_equal(result, [-7.0, -5.0, -3.0])


def test_combined_harmonic_basic():
    a = np.array([-2.0, -4.0])
    b = np.array([-2.0, -4.0])
    result = combined_harmonic(a, b)
    np.testing.assert_array_almost_equal(result, [-2.0, -4.0])


def test_combined_harmonic_zero_guard():
    a = np.array([0.0, 5.0])
    b = np.array([3.0, 0.0])
    result = combined_harmonic(a, b)
    np.testing.assert_array_equal(result, [0.0, 0.0])


def test_pareto_mask():
    a = np.array([-3.0, -2.0, -1.0, -1.5])
    b = np.array([-1.0, -2.0, -3.0, -1.5])
    mask = pareto_mask(a, b)
    # Points (-3,-1), (-2,-2), (-1,-3) are Pareto-optimal; (-1.5, -1.5) dominated by (-2, -2)
    assert mask[0]  # (-3, -1)
    assert mask[1]  # (-2, -2)
    assert mask[2]  # (-1, -3)
    assert not mask[3]  # (-1.5, -1.5) dominated by (-2, -2)


def test_geometric_mean_basic():
    a = np.array([-4.0, -9.0])
    b = np.array([-4.0, -9.0])
    result = combined_geometric(a, b)
    np.testing.assert_array_almost_equal(result, [-4.0, -9.0])


def test_geometric_mean_zero():
    a = np.array([0.0, 5.0])
    b = np.array([3.0, -1.0])
    result = combined_geometric(a, b)
    np.testing.assert_array_equal(result, [0.0, 0.0])


def test_weighted_sum_equal_weights():
    a = np.array([-2.0, -6.0])
    b = np.array([-4.0, -2.0])
    result = combined_weighted_sum(a, b, weight_a=0.5)
    np.testing.assert_array_almost_equal(result, [-3.0, -4.0])


def test_weighted_sum_asymmetric():
    a = np.array([-10.0])
    b = np.array([0.0])
    result = combined_weighted_sum(a, b, weight_a=0.3)
    np.testing.assert_array_almost_equal(result, [-3.0])


def test_weighted_product_equal():
    a = np.array([-4.0, -9.0])
    b = np.array([-4.0, -9.0])
    result = combined_weighted_product(a, b, weight_a=0.5)
    np.testing.assert_array_almost_equal(result, [-4.0, -9.0])


def test_weighted_product_zero_safe():
    a = np.array([0.0])
    b = np.array([-5.0])
    result = combined_weighted_product(a, b, weight_a=0.5)
    assert result[0] == 0.0


def test_tchebycheff_symmetric():
    a = np.array([-3.0, -1.0])
    b = np.array([-1.0, -3.0])
    result = combined_tchebycheff(a, b, weight_a=0.5, utopia_a=0.0, utopia_b=0.0)
    # max(0.5*3, 0.5*1) = 1.5; max(0.5*1, 0.5*3) = 1.5
    np.testing.assert_array_almost_equal(result, [1.5, 1.5])


def test_tchebycheff_with_utopia():
    a = np.array([-5.0])
    b = np.array([-5.0])
    result = combined_tchebycheff(a, b, weight_a=0.5, utopia_a=-5.0, utopia_b=-5.0)
    np.testing.assert_array_almost_equal(result, [0.0])


def test_hypervolume_contributions_basic():
    a = np.array([-3.0, -1.0, -2.0])
    b = np.array([-1.0, -3.0, -2.0])
    hv = hypervolume_contributions(a, b, ref_a=0.0, ref_b=0.0)
    # Все три точки на фронте Парето, каждая должна иметь вклад > 0
    assert np.all(hv > 0)


def test_hypervolume_dominated_point():
    a = np.array([-3.0, -1.0])
    b = np.array([-3.0, -1.0])
    hv = hypervolume_contributions(a, b, ref_a=0.0, ref_b=0.0)
    # (-3,-3) доминирует (-1,-1), поэтому (-1,-1) не на фронте -> вклад 0
    assert hv[0] > 0
    assert hv[1] == 0.0


def test_compute_combined_dispatch():
    a = np.array([-1.0, -5.0])
    b = np.array([-3.0, -2.0])
    result_min = compute_combined(a, b, ObjectiveKind.MIN)
    result_harm = compute_combined(a, b, ObjectiveKind.HARMONIC)
    result_geom = compute_combined(a, b, ObjectiveKind.GEOMETRIC)
    result_wsum = compute_combined(a, b, ObjectiveKind.WEIGHTED_SUM)
    result_wprod = compute_combined(a, b, ObjectiveKind.WEIGHTED_PRODUCT)
    result_tch = compute_combined(a, b, ObjectiveKind.TCHEBYCHEFF)
    np.testing.assert_array_equal(result_min, [-1.0, -2.0])
    assert result_harm[0] <= 0
    assert len(result_geom) == 2
    assert len(result_wsum) == 2
    assert len(result_wprod) == 2
    assert len(result_tch) == 2
