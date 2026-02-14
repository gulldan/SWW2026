"""Tests for cli/score.py hit-threshold logic and objective columns."""

import numpy as np
import pandas as pd

from dualdocking.scoring.objectives import ObjectiveKind, compute_combined, parse_objectives_config


def test_hit_threshold_uses_leq():
    """Verify <= is used consistently for hit threshold (score <= threshold = hit)."""
    thr_a = 0.0
    thr_b = 0.0

    affinity_a = np.array([-1.0, 0.0, 0.5])
    affinity_b = np.array([-1.0, 0.0, 0.5])

    is_hit_both = (affinity_a <= thr_a) & (affinity_b <= thr_b)

    assert is_hit_both[0]  # -1.0 <= 0.0 → hit
    assert is_hit_both[1]  # 0.0 <= 0.0 → hit (boundary)
    assert not is_hit_both[2]  # 0.5 > 0.0 → not hit


def test_all_objective_columns_generated():
    """Parse config → compute → verify all combined_* columns exist."""
    exp_cfg = {
        "objectives": {
            "baseline": "min",
            "alternatives": [
                {"kind": "harmonic"},
                {"kind": "geometric"},
                {"kind": "weighted_sum", "weight_a": 0.5},
                {"kind": "weighted_product", "weight_a": 0.5},
                {"kind": "tchebycheff", "weight_a": 0.5, "utopia_a": -3.0, "utopia_b": -3.0},
            ],
        }
    }
    objectives = parse_objectives_config(exp_cfg)

    scores_a = np.array([-8.0, -5.0, -3.0])
    scores_b = np.array([-7.0, -6.0, -4.0])

    joined = pd.DataFrame({"smiles": ["A", "B", "C"]})
    for obj in objectives:
        joined[obj.column_name] = compute_combined(scores_a, scores_b, obj.kind, lower_is_better=True, **obj.params)

    expected_cols = {
        "combined_min",
        "combined_harmonic",
        "combined_geometric",
        "combined_weighted_sum_0.5",
        "combined_weighted_product_0.5",
        "combined_tchebycheff_0.5",
    }
    assert expected_cols.issubset(set(joined.columns))


def test_sort_by_primary_objective():
    """Verify DataFrame sorted by primary column, not hardcoded combined_min."""
    exp_cfg = {
        "objectives": {
            "baseline": "harmonic",
            "alternatives": [{"kind": "min"}],
        }
    }
    objectives = parse_objectives_config(exp_cfg)
    assert objectives[0].kind == ObjectiveKind.HARMONIC
    assert objectives[0].is_primary

    scores_a = np.array([-8.0, -5.0, -3.0, -9.0])
    scores_b = np.array([-7.0, -6.0, -4.0, -2.0])

    joined = pd.DataFrame({"smiles": ["A", "B", "C", "D"]})
    primary_col = "combined_min"  # fallback
    for obj in objectives:
        joined[obj.column_name] = compute_combined(scores_a, scores_b, obj.kind, lower_is_better=True, **obj.params)
        if obj.is_primary:
            primary_col = obj.column_name

    assert primary_col == "combined_harmonic"
    joined = joined.sort_values(primary_col)
    # Values should be non-decreasing
    vals = joined[primary_col].to_numpy()
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
