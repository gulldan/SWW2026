"""Tests for dualdocking.enrichment — docking-based enrichment metrics."""

from __future__ import annotations

import numpy as np
import pytest

from dualdocking.analysis.enrichment import (
    EnrichmentResult,
    compute_auroc,
    compute_bedroc,
    compute_enrichment_factor,
    compute_roc_curve,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perfect_separation(n_pos: int = 50, n_neg: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Actives get scores (-10..-8), decoys get (-4..-2). Perfect separation."""
    rng = np.random.RandomState(42)
    active_scores = rng.uniform(-10.0, -8.0, size=n_pos)
    decoy_scores = rng.uniform(-4.0, -2.0, size=n_neg)
    scores = np.concatenate([active_scores, decoy_scores])
    labels = np.array([1] * n_pos + [0] * n_neg)
    return scores, labels


def _random_separation(n_pos: int = 50, n_neg: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Both actives and decoys sample from the same distribution."""
    rng = np.random.RandomState(42)
    scores = rng.uniform(-8.0, -2.0, size=n_pos + n_neg)
    labels = np.array([1] * n_pos + [0] * n_neg)
    return scores, labels


def _inverted_separation(n_pos: int = 50, n_neg: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Actives score worse than decoys (inverted)."""
    rng = np.random.RandomState(42)
    active_scores = rng.uniform(-4.0, -2.0, size=n_pos)
    decoy_scores = rng.uniform(-10.0, -8.0, size=n_neg)
    scores = np.concatenate([active_scores, decoy_scores])
    labels = np.array([1] * n_pos + [0] * n_neg)
    return scores, labels


# ===========================================================================
# TestComputeAuroc
# ===========================================================================


class TestComputeAuroc:
    """Tests for compute_auroc."""

    def test_perfect_separation(self) -> None:
        scores, labels = _perfect_separation()
        auroc = compute_auroc(scores, labels)
        assert auroc == pytest.approx(1.0, abs=1e-6)

    def test_random_separation(self) -> None:
        scores, labels = _random_separation(n_pos=500, n_neg=2000)
        auroc = compute_auroc(scores, labels)
        assert 0.35 < auroc < 0.65

    def test_inverted_separation(self) -> None:
        scores, labels = _inverted_separation()
        auroc = compute_auroc(scores, labels)
        assert auroc < 0.05

    def test_single_active(self) -> None:
        scores = np.array([-10.0, -5.0, -4.0, -3.0])
        labels = np.array([1, 0, 0, 0])
        auroc = compute_auroc(scores, labels)
        assert auroc == pytest.approx(1.0, abs=1e-6)

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            compute_auroc(np.array([1.0, 2.0]), np.array([1]))

    def test_no_positives(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            compute_auroc(np.array([1.0, 2.0]), np.array([0, 0]))

    def test_no_negatives(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            compute_auroc(np.array([1.0, 2.0]), np.array([1, 1]))

    def test_lower_is_better_false(self) -> None:
        # When lower_is_better=False, higher score = better
        scores = np.array([10.0, 8.0, 4.0, 2.0])
        labels = np.array([1, 1, 0, 0])
        auroc = compute_auroc(scores, labels, lower_is_better=False)
        assert auroc == pytest.approx(1.0, abs=1e-6)

    def test_tied_scores(self) -> None:
        scores = np.array([-5.0, -5.0, -5.0, -5.0])
        labels = np.array([1, 1, 0, 0])
        auroc = compute_auroc(scores, labels)
        assert auroc == pytest.approx(0.5, abs=1e-6)


# ===========================================================================
# TestComputeBedroc
# ===========================================================================


class TestComputeBedroc:
    """Tests for compute_bedroc."""

    def test_perfect_early_enrichment(self) -> None:
        scores, labels = _perfect_separation()
        bedroc = compute_bedroc(scores, labels, alpha=20.0)
        assert bedroc > 0.9

    def test_range_zero_to_one(self) -> None:
        scores, labels = _random_separation(n_pos=100, n_neg=1000)
        bedroc = compute_bedroc(scores, labels, alpha=20.0)
        assert 0.0 <= bedroc <= 1.0

    def test_inverted_low_bedroc(self) -> None:
        scores, labels = _inverted_separation()
        bedroc = compute_bedroc(scores, labels, alpha=20.0)
        assert bedroc < 0.1

    def test_alpha_sensitivity(self) -> None:
        """Higher alpha should be more sensitive to early enrichment."""
        scores, labels = _perfect_separation()
        bedroc_20 = compute_bedroc(scores, labels, alpha=20.0)
        bedroc_5 = compute_bedroc(scores, labels, alpha=5.0)
        # Both should be high for perfect separation
        assert bedroc_20 > 0.9
        assert bedroc_5 > 0.9

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            compute_bedroc(np.array([1.0, 2.0]), np.array([1]))

    def test_no_positives(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            compute_bedroc(np.array([1.0, 2.0]), np.array([0, 0]))


# ===========================================================================
# TestComputeEnrichmentFactor
# ===========================================================================


class TestComputeEnrichmentFactor:
    """Tests for compute_enrichment_factor."""

    def test_perfect_separation(self) -> None:
        scores, labels = _perfect_separation(n_pos=50, n_neg=950)
        ef = compute_enrichment_factor(scores, labels, fraction=0.05)
        # Perfect: all top 5% are actives, baseline rate = 50/1000 = 5%
        assert ef == pytest.approx(20.0, rel=0.1)

    def test_random_near_one(self) -> None:
        scores, labels = _random_separation(n_pos=500, n_neg=2000)
        ef = compute_enrichment_factor(scores, labels, fraction=0.10)
        assert 0.5 < ef < 2.0

    def test_no_actives(self) -> None:
        scores = np.array([-5.0, -4.0, -3.0])
        labels = np.array([0, 0, 0])
        ef = compute_enrichment_factor(scores, labels, fraction=0.5)
        assert ef == 0.0

    def test_fraction_1pct(self) -> None:
        scores, labels = _perfect_separation(n_pos=10, n_neg=990)
        ef = compute_enrichment_factor(scores, labels, fraction=0.01)
        # Top 1% = 10 molecules, all 10 should be actives = EF = 1.0/(10/1000) = 100
        assert ef == pytest.approx(100.0, rel=0.1)

    def test_lower_is_better_false(self) -> None:
        scores = np.array([10.0, 8.0, -5.0, -6.0])
        labels = np.array([1, 1, 0, 0])
        ef = compute_enrichment_factor(scores, labels, fraction=0.5, lower_is_better=False)
        assert ef == pytest.approx(2.0, rel=0.1)


# ===========================================================================
# TestComputeRocCurve
# ===========================================================================


class TestComputeRocCurve:
    """Tests for compute_roc_curve."""

    def test_endpoints(self) -> None:
        scores, labels = _perfect_separation()
        fpr, tpr = compute_roc_curve(scores, labels)
        assert fpr[0] == pytest.approx(0.0)
        assert tpr[0] == pytest.approx(0.0)
        assert fpr[-1] == pytest.approx(1.0)
        assert tpr[-1] == pytest.approx(1.0)

    def test_monotonic_tpr(self) -> None:
        scores, labels = _random_separation()
        _fpr, tpr = compute_roc_curve(scores, labels)
        assert np.all(np.diff(tpr) >= 0)

    def test_monotonic_fpr(self) -> None:
        scores, labels = _random_separation()
        fpr, _tpr = compute_roc_curve(scores, labels)
        assert np.all(np.diff(fpr) >= 0)

    def test_length(self) -> None:
        scores, labels = _perfect_separation(n_pos=10, n_neg=20)
        fpr, tpr = compute_roc_curve(scores, labels)
        # One point per sample + the (0,0) start
        assert len(fpr) == 10 + 20 + 1
        assert len(tpr) == len(fpr)

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            compute_roc_curve(np.array([1.0, 2.0]), np.array([1]))


# ===========================================================================
# TestEnrichmentResult
# ===========================================================================


class TestEnrichmentResult:
    """Tests for EnrichmentResult dataclass."""

    def test_creation(self) -> None:
        r = EnrichmentResult(
            target_id="CDK2",
            n_actives=100,
            n_decoys=1000,
            auroc=0.85,
            bedroc_alpha20=0.55,
            ef_1pct=15.0,
            ef_5pct=8.0,
            passed=True,
            status="pass",
        )
        assert r.target_id == "CDK2"
        assert r.passed is True
        assert r.error is None

    def test_error_result(self) -> None:
        r = EnrichmentResult(
            target_id="CDK5",
            n_actives=0,
            n_decoys=0,
            auroc=0.0,
            bedroc_alpha20=0.0,
            ef_1pct=0.0,
            ef_5pct=0.0,
            passed=False,
            status="error",
            error="Docking failed",
        )
        assert r.status == "error"
        assert r.error == "Docking failed"

    def test_skipped_blocked(self) -> None:
        r = EnrichmentResult(
            target_id="COX1",
            n_actives=0,
            n_decoys=0,
            auroc=0.0,
            bedroc_alpha20=0.0,
            ef_1pct=0.0,
            ef_5pct=0.0,
            passed=False,
            status="skipped:blocked",
        )
        assert r.status == "skipped:blocked"

    def test_frozen(self) -> None:
        r = EnrichmentResult(
            target_id="CDK2",
            n_actives=100,
            n_decoys=1000,
            auroc=0.85,
            bedroc_alpha20=0.55,
            ef_1pct=15.0,
            ef_5pct=8.0,
            passed=True,
            status="pass",
        )
        with pytest.raises(AttributeError):
            r.auroc = 0.9  # type: ignore[misc]


# ===========================================================================
# TestRerunStability
# ===========================================================================


class TestRerunStability:
    """Verify deterministic results for identical inputs (DoD requirement)."""

    def test_auroc_deterministic(self) -> None:
        scores, labels = _perfect_separation()
        a1 = compute_auroc(scores, labels)
        a2 = compute_auroc(scores, labels)
        assert a1 == a2

    def test_bedroc_deterministic(self) -> None:
        scores, labels = _perfect_separation()
        b1 = compute_bedroc(scores, labels, alpha=20.0)
        b2 = compute_bedroc(scores, labels, alpha=20.0)
        assert b1 == b2

    def test_ef_deterministic(self) -> None:
        scores, labels = _perfect_separation()
        e1 = compute_enrichment_factor(scores, labels, fraction=0.01)
        e2 = compute_enrichment_factor(scores, labels, fraction=0.01)
        assert e1 == e2

    def test_roc_curve_deterministic(self) -> None:
        scores, labels = _perfect_separation()
        fpr1, tpr1 = compute_roc_curve(scores, labels)
        fpr2, tpr2 = compute_roc_curve(scores, labels)
        np.testing.assert_array_equal(fpr1, fpr2)
        np.testing.assert_array_equal(tpr1, tpr2)

    def test_sampling_deterministic(self) -> None:
        """Verify that random.sample with fixed seed is deterministic."""
        import random

        smiles = [f"C{'C' * i}" for i in range(500)]
        rng1 = random.Random(137)  # noqa: S311
        s1 = rng1.sample(smiles, 200)
        rng2 = random.Random(137)  # noqa: S311
        s2 = rng2.sample(smiles, 200)
        assert s1 == s2
