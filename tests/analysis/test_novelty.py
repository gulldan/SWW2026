"""Tests for novelty and cross-condition overlap metrics."""

from __future__ import annotations

import pytest

from dualdocking.analysis.novelty import (
    NoveltyResult,
    cross_condition_overlap,
    novelty_vs_baseline,
)

# Well-known, structurally distinct SMILES
ETHANOL = "CCO"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
NAPHTHALENE = "c1ccc2ccccc2c1"
BENZENE = "c1ccccc1"


class TestNoveltyVsBaseline:
    def test_identical_sets_low_novelty(self) -> None:
        mols = [ETHANOL, ASPIRIN, CAFFEINE, IBUPROFEN]
        result = novelty_vs_baseline(mols, mols, threshold=0.4)
        assert isinstance(result, NoveltyResult)
        # Identical molecules have Tanimoto 1.0 >= 0.4, so none are novel
        assert result.novelty_fraction == pytest.approx(0.0)
        assert result.n_novel == 0
        assert result.n_total == 4
        assert result.n_baseline == 4

    def test_disjoint_sets_high_novelty(self) -> None:
        gen = [CAFFEINE, NAPHTHALENE]
        base = [ETHANOL]
        result = novelty_vs_baseline(gen, base, threshold=0.4)
        # These are structurally very different from ethanol
        assert result.novelty_fraction == pytest.approx(1.0)
        assert result.n_novel == 2

    def test_threshold_effect(self) -> None:
        gen = [ASPIRIN]
        base = [BENZENE]
        low = novelty_vs_baseline(gen, base, threshold=0.2)
        high = novelty_vs_baseline(gen, base, threshold=0.99)
        # With a very high threshold, everything is novel
        assert high.novelty_fraction >= low.novelty_fraction

    def test_empty_generated(self) -> None:
        result = novelty_vs_baseline([], [ETHANOL])
        assert result.novelty_fraction == 0.0
        assert result.n_total == 0

    def test_empty_baseline(self) -> None:
        result = novelty_vs_baseline([ETHANOL], [])
        assert result.novelty_fraction == 0.0
        assert result.n_baseline == 0

    def test_invalid_smiles_skipped(self) -> None:
        result = novelty_vs_baseline(
            ["NOT_A_SMILES", CAFFEINE],
            [ETHANOL],
            threshold=0.4,
        )
        # Only caffeine should be processed (1 valid generated)
        assert result.n_total == 1

    def test_result_fields(self) -> None:
        result = novelty_vs_baseline([ETHANOL], [ASPIRIN], threshold=0.5)
        assert isinstance(result.novelty_fraction, float)
        assert isinstance(result.n_novel, int)
        assert isinstance(result.n_total, int)
        assert isinstance(result.n_baseline, int)
        assert isinstance(result.threshold, float)
        assert result.threshold == 0.5


class TestCrossConditionOverlap:
    def test_identical_conditions_high_overlap(self) -> None:
        mols = [ETHANOL, ASPIRIN, CAFFEINE, IBUPROFEN]
        result = cross_condition_overlap(
            {"A": mols, "B": mols},
            threshold=0.85,
        )
        # Identical sets: every molecule matches itself at Tanimoto=1.0
        assert result[("A", "B")] == pytest.approx(1.0)

    def test_disjoint_conditions_low_overlap(self) -> None:
        result = cross_condition_overlap(
            {"X": [ETHANOL], "Y": [CAFFEINE]},
            threshold=0.85,
        )
        # Very different molecules shouldn't overlap at high threshold
        assert result[("X", "Y")] == pytest.approx(0.0)

    def test_three_conditions(self) -> None:
        result = cross_condition_overlap(
            {"A": [ETHANOL], "B": [ASPIRIN], "C": [CAFFEINE]},
            threshold=0.85,
        )
        # Should have 3 pairs
        assert len(result) == 3
        assert ("A", "B") in result
        assert ("A", "C") in result
        assert ("B", "C") in result

    def test_empty_condition(self) -> None:
        result = cross_condition_overlap(
            {"A": [ETHANOL], "B": []},
            threshold=0.85,
        )
        assert result[("A", "B")] == 0.0

    def test_invalid_smiles_handled(self) -> None:
        result = cross_condition_overlap(
            {"A": ["INVALID"], "B": [ETHANOL]},
            threshold=0.85,
        )
        assert result[("A", "B")] == 0.0

    def test_keys_sorted(self) -> None:
        result = cross_condition_overlap(
            {"Z": [ETHANOL], "A": [ASPIRIN]},
            threshold=0.85,
        )
        # Keys should be sorted: ("A", "Z") not ("Z", "A")
        assert ("A", "Z") in result
        assert ("Z", "A") not in result

    def test_subsampling(self) -> None:
        many = [ETHANOL, ASPIRIN, CAFFEINE, IBUPROFEN, NAPHTHALENE, BENZENE]
        result = cross_condition_overlap(
            {"A": many, "B": many},
            threshold=0.85,
            max_per_condition=3,
            seed=42,
        )
        # Should still produce a result (subsampled to 3)
        assert ("A", "B") in result
        assert 0.0 <= result[("A", "B")] <= 1.0
