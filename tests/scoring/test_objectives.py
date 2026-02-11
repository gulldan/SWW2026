"""Tests for dual-target objective functions."""

import numpy as np
import pytest

from dualdocking.scoring.objectives import (
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
    parse_objectives_config,
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


def test_combined_harmonic_shift():
    """With shift, harmonic mean preserves discriminability for non-positive utilities."""
    a = np.array([0.0, 5.0])
    b = np.array([3.0, 0.0])
    result = combined_harmonic(a, b)
    # After shift, these should produce finite non-NaN values
    assert np.all(np.isfinite(result))


def test_pareto_mask():
    a = np.array([-3.0, -2.0, -1.0, -1.5])
    b = np.array([-1.0, -2.0, -3.0, -1.5])
    mask = pareto_mask(a, b)
    # Points (-3,-1), (-2,-2), (-1,-3) are Pareto-optimal; (-1.5, -1.5) dominated by (-2, -2)
    assert mask[0]  # (-3, -1)
    assert mask[1]  # (-2, -2)
    assert mask[2]  # (-1, -3)
    assert not mask[3]  # (-1.5, -1.5) dominated by (-2, -2)


def test_pareto_mask_tied_score_a():
    """When score_a is tied, only the point with best score_b is non-dominated."""
    a = np.array([-3.0, -3.0])
    b = np.array([-2.0, -5.0])
    mask = pareto_mask(a, b, lower_is_better=True)
    assert not mask[0]  # (-3, -2) dominated by (-3, -5)
    assert mask[1]  # (-3, -5) non-dominated


def test_pareto_mask_identical_points_all_nondominated():
    """Identical points do not dominate each other and should stay on the frontier."""
    a = np.array([-3.0, -3.0, -2.0])
    b = np.array([-2.0, -2.0, -1.0])
    mask = pareto_mask(a, b, lower_is_better=True)
    assert mask[0]
    assert mask[1]
    assert not mask[2]


def test_geometric_mean_basic():
    a = np.array([-4.0, -9.0])
    b = np.array([-4.0, -9.0])
    result = combined_geometric(a, b)
    np.testing.assert_array_almost_equal(result, [-4.0, -9.0])


def test_geometric_mean_shift():
    """With shift, geometric mean preserves discriminability for non-positive utilities."""
    a = np.array([0.0, 5.0])
    b = np.array([3.0, -1.0])
    result = combined_geometric(a, b)
    # After shift, these should produce finite non-NaN values
    assert np.all(np.isfinite(result))


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


def test_tchebycheff_return_domain_documented():
    """Tchebycheff returns non-negative deviation (0 = at utopia)."""
    a = np.array([-5.0, -3.0, -7.0])
    b = np.array([-5.0, -7.0, -3.0])
    result = combined_tchebycheff(a, b, weight_a=0.5, utopia_a=-7.0, utopia_b=-7.0)
    # All deviations must be >= 0
    assert np.all(result >= 0)
    # Point at utopia on both axes should have deviation 0
    a_at_utopia = np.array([-7.0])
    b_at_utopia = np.array([-7.0])
    result_utopia = combined_tchebycheff(a_at_utopia, b_at_utopia, utopia_a=-7.0, utopia_b=-7.0)
    np.testing.assert_almost_equal(result_utopia[0], 0.0)


def test_harmonic_with_positive_scores():
    """Harmonic mean should discriminate even when scores > 0 (utility < 0)."""
    a = np.array([2.0, 5.0, 10.0])
    b = np.array([2.0, 5.0, 10.0])
    result = combined_harmonic(a, b, lower_is_better=True)
    # Should still be able to rank: lower score = better
    # With shift, all three should be distinguishable
    assert result[0] < result[1] < result[2]


def test_pareto_mask_sorted_algorithm():
    """O(n log n) pareto_mask gives same result as brute-force reference."""
    rng = np.random.default_rng(42)
    a = rng.uniform(-10, 0, size=200)
    b = rng.uniform(-10, 0, size=200)

    # Brute-force reference
    n = len(a)
    ref_mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not ref_mask[i]:
            continue
        for j in range(n):
            if i == j or not ref_mask[j]:
                continue
            if a[j] <= a[i] and b[j] <= b[i] and (a[j] < a[i] or b[j] < b[i]):
                ref_mask[i] = False
                break

    result = pareto_mask(a, b, lower_is_better=True)
    np.testing.assert_array_equal(result, ref_mask)


# --- parse_objectives_config tests ---


def _full_experiment_cfg():
    return {
        "objectives": {
            "baseline": "min",
            "alternatives": [
                {"kind": "geometric"},
                {"kind": "harmonic"},
                {"kind": "weighted_sum", "weight_a": 0.5},
                {"kind": "weighted_product", "weight_a": 0.5},
                {"kind": "tchebycheff", "weight_a": 0.5, "utopia_a": -3.0, "utopia_b": -3.0},
            ],
        }
    }


def test_parse_objectives_config_full():
    specs = parse_objectives_config(_full_experiment_cfg())
    assert len(specs) == 6
    assert specs[0].is_primary
    assert specs[0].kind == ObjectiveKind.MIN
    kinds = [s.kind for s in specs]
    assert ObjectiveKind.GEOMETRIC in kinds
    assert ObjectiveKind.HARMONIC in kinds
    assert ObjectiveKind.WEIGHTED_SUM in kinds
    assert ObjectiveKind.WEIGHTED_PRODUCT in kinds
    assert ObjectiveKind.TCHEBYCHEFF in kinds


def test_parse_objectives_config_missing_section():
    specs = parse_objectives_config({})
    assert len(specs) == 1
    assert specs[0].kind == ObjectiveKind.MIN
    assert specs[0].is_primary


def test_parse_objectives_config_empty_alternatives():
    specs = parse_objectives_config({"objectives": {"baseline": "min", "alternatives": []}})
    assert len(specs) == 1
    assert specs[0].is_primary


def test_parse_objectives_config_params_passed():
    cfg = {
        "objectives": {
            "baseline": "min",
            "alternatives": [
                {"kind": "tchebycheff", "weight_a": 0.5, "utopia_a": -3.0, "utopia_b": -3.0},
            ],
        }
    }
    specs = parse_objectives_config(cfg)
    tch = [s for s in specs if s.kind == ObjectiveKind.TCHEBYCHEFF][0]
    assert tch.params["weight_a"] == 0.5
    assert tch.params["utopia_a"] == -3.0
    assert tch.params["utopia_b"] == -3.0


def test_parse_objectives_config_duplicate_column_skipped():
    cfg = {
        "objectives": {
            "baseline": "min",
            "alternatives": [
                {"kind": "min"},
            ],
        }
    }
    specs = parse_objectives_config(cfg)
    assert len(specs) == 1


def test_parse_objectives_config_same_kind_different_params():
    cfg = {
        "objectives": {
            "baseline": "min",
            "alternatives": [
                {"kind": "weighted_sum", "weight_a": 0.3},
                {"kind": "weighted_sum", "weight_a": 0.7},
            ],
        }
    }
    specs = parse_objectives_config(cfg)
    ws_specs = [s for s in specs if s.kind == ObjectiveKind.WEIGHTED_SUM]
    assert len(ws_specs) == 2
    cols = {s.column_name for s in ws_specs}
    assert "combined_weighted_sum_0.3" in cols
    assert "combined_weighted_sum_0.7" in cols


def test_parse_objectives_config_unknown_kind_raises():
    cfg = {"objectives": {"baseline": "min", "alternatives": [{"kind": "nonexistent"}]}}
    with pytest.raises(ValueError):
        parse_objectives_config(cfg)


def test_parse_objectives_config_no_alternatives_key():
    cfg = {"objectives": {"baseline": "harmonic"}}
    specs = parse_objectives_config(cfg)
    assert len(specs) == 1
    assert specs[0].kind == ObjectiveKind.HARMONIC
    assert specs[0].is_primary
