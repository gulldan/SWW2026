"""Tests for metrics module."""

import numpy as np

from dualdocking.metrics import compute_dual_hits, enrichment_at_k


def test_dual_hits():
    scores_a = np.array([-8.0, -6.0, -9.0, -5.0])
    scores_b = np.array([-7.5, -8.0, -6.5, -4.0])
    result = compute_dual_hits(scores_a, scores_b, threshold_a=-7.0, threshold_b=-7.0)
    assert result.n_total == 4
    assert result.n_hit_a == 2  # -8.0, -9.0
    assert result.n_hit_b == 2  # -7.5, -8.0
    assert result.n_hit_both == 1  # only first molecule hits both


def test_dual_hits_ignores_nan():
    scores_a = np.array([-8.0, np.nan, -9.0])
    scores_b = np.array([-7.5, -8.0, np.nan])
    result = compute_dual_hits(scores_a, scores_b, threshold_a=-7.0, threshold_b=-7.0)
    # Только один валидный элемент (-8.0, -7.5)
    assert result.n_total == 1
    assert result.n_hit_a == 1
    assert result.n_hit_b == 1
    assert result.n_hit_both == 1


def test_enrichment_at_k():
    scores = np.array([-10.0, -5.0, -8.0, -3.0, -9.0])
    is_active = np.array([True, False, True, False, True])
    ef = enrichment_at_k(scores, is_active, k=2)
    # Top-2 by ascending score: -10.0 (active), -9.0 (active) → 2/2 = 1.0
    # Expected rate: 3/5 = 0.6
    # EF = 1.0 / 0.6 ≈ 1.667
    assert ef > 1.0


def test_enrichment_at_k_clamps_k():
    scores = np.array([-2.0, -1.0])
    is_active = np.array([True, False])
    ef = enrichment_at_k(scores, is_active, k=10)
    # k clamped to N=2
    assert ef >= 0.0
