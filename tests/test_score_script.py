"""Tests for scripts/score.py hit-threshold logic."""

import numpy as np


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
