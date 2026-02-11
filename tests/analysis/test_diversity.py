"""Tests for dualdocking.analysis.diversity."""

from __future__ import annotations

import pytest

from dualdocking.analysis.diversity import (
    DiversityResult,
    compute_all_diversity_metrics,
    internal_diversity_tanimoto,
    scaffold_diversity,
    tanimoto_snn,
    uniqueness,
)

# Well-known drug SMILES for testing
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
NAPROXEN = "COc1ccc2cc(C(C)C(=O)O)ccc2c1"
ETHANOL = "CCO"
BENZENE = "c1ccccc1"
PYRIDINE = "c1ccncc1"
INDOLE = "c1ccc2[nH]ccc2c1"
NAPHTHALENE = "c1ccc2ccccc2c1"


class TestUniqueness:
    def test_all_unique(self) -> None:
        smiles = [ASPIRIN, IBUPROFEN, PARACETAMOL, CAFFEINE, NAPROXEN]
        assert uniqueness(smiles) == pytest.approx(1.0)

    def test_all_duplicates(self) -> None:
        smiles = [ASPIRIN] * 5
        assert uniqueness(smiles) == pytest.approx(1 / 5)

    def test_mixed(self) -> None:
        smiles = [ASPIRIN, IBUPROFEN, PARACETAMOL, ASPIRIN, IBUPROFEN]
        assert uniqueness(smiles) == pytest.approx(3 / 5)

    def test_empty_list(self) -> None:
        assert uniqueness([]) == 0.0

    def test_invalid_smiles_skipped(self) -> None:
        smiles = [ASPIRIN, "NOT_A_SMILES", IBUPROFEN]
        result = uniqueness(smiles)
        # 2 valid unique out of 3 total
        assert result == pytest.approx(2 / 3)

    def test_single_molecule(self) -> None:
        assert uniqueness([ASPIRIN]) == pytest.approx(1.0)

    def test_canonicalization(self) -> None:
        # Aspirin in non-canonical form should be recognized as same
        smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(O)=O"]
        result = uniqueness(smiles)
        # Both canonicalize to the same SMILES
        assert result == pytest.approx(1 / 2)


class TestScaffoldDiversity:
    def test_same_scaffold(self) -> None:
        # Benzene derivatives share the benzene scaffold
        smiles = [BENZENE, "c1ccccc1O", "c1ccccc1N", "c1ccccc1C"]
        ratio = scaffold_diversity(smiles)
        assert 0.0 <= ratio <= 1.0
        # All are simple benzene derivatives -> ratio should be low (few scaffolds)
        assert ratio < 1.0

    def test_different_scaffolds(self) -> None:
        smiles = [PYRIDINE, INDOLE, NAPHTHALENE, CAFFEINE]
        ratio = scaffold_diversity(smiles)
        assert 0.0 <= ratio <= 1.0
        # Diverse scaffolds -> ratio should be high
        assert ratio > 0.5

    def test_single_molecule(self) -> None:
        ratio = scaffold_diversity([ASPIRIN])
        assert ratio == pytest.approx(1.0)

    def test_empty(self) -> None:
        assert scaffold_diversity([]) == 0.0

    def test_all_invalid(self) -> None:
        assert scaffold_diversity(["INVALID", "ALSO_INVALID"]) == 0.0


class TestInternalDiversity:
    def test_identical_molecules(self) -> None:
        smiles = [ASPIRIN] * 10
        div, stderr = internal_diversity_tanimoto(smiles)
        assert div == pytest.approx(0.0, abs=0.01)
        assert stderr is None  # n <= 316, exact computation

    def test_dissimilar_molecules(self) -> None:
        smiles = [ASPIRIN, CAFFEINE, ETHANOL, NAPHTHALENE, INDOLE]
        div, stderr = internal_diversity_tanimoto(smiles)
        assert div > 0.5
        assert stderr is None  # n <= 316

    def test_range_zero_one(self) -> None:
        smiles = [ASPIRIN, IBUPROFEN, CAFFEINE, PARACETAMOL]
        div, _ = internal_diversity_tanimoto(smiles)
        assert 0.0 <= div <= 1.0

    def test_single_molecule(self) -> None:
        div, stderr = internal_diversity_tanimoto([ASPIRIN])
        assert div == 0.0
        assert stderr is None

    def test_empty(self) -> None:
        div, stderr = internal_diversity_tanimoto([])
        assert div == 0.0
        assert stderr is None

    def test_sampling_mode(self) -> None:
        # >316 molecules triggers sampling path (with stderr)
        # Use duplicates to keep the test fast
        smiles = [ASPIRIN, CAFFEINE, ETHANOL, IBUPROFEN] * 100  # 400 molecules
        div, stderr = internal_diversity_tanimoto(smiles, max_pairs=500)
        assert 0.0 <= div <= 1.0
        assert stderr is not None
        assert stderr >= 0.0

    def test_higher_diversity_for_more_diverse_set(self) -> None:
        similar = [ASPIRIN] * 5 + [IBUPROFEN] * 5
        diverse = [ASPIRIN, CAFFEINE, ETHANOL, NAPHTHALENE, INDOLE, PYRIDINE, BENZENE, IBUPROFEN, PARACETAMOL, NAPROXEN]
        div_similar, _ = internal_diversity_tanimoto(similar)
        div_diverse, _ = internal_diversity_tanimoto(diverse)
        assert div_diverse > div_similar


class TestTanimotoSNN:
    def test_identical_molecules(self) -> None:
        smiles = [ASPIRIN] * 5
        snn = tanimoto_snn(smiles, k=3)
        assert snn == pytest.approx(1.0, abs=0.01)

    def test_dissimilar_molecules(self) -> None:
        smiles = [ASPIRIN, CAFFEINE, ETHANOL, NAPHTHALENE]
        snn = tanimoto_snn(smiles, k=2)
        # Dissimilar molecules should have lower SNN
        assert snn < 0.8

    def test_k_parameter(self) -> None:
        smiles = [ASPIRIN, IBUPROFEN, CAFFEINE, ETHANOL, NAPHTHALENE]
        snn_k1 = tanimoto_snn(smiles, k=1)
        snn_k3 = tanimoto_snn(smiles, k=3)
        # k=1 uses only the most similar neighbor, should be >= k=3
        assert snn_k1 >= snn_k3 - 0.01  # small tolerance

    def test_empty(self) -> None:
        assert tanimoto_snn([], k=3) == 0.0

    def test_single_molecule(self) -> None:
        assert tanimoto_snn([ASPIRIN], k=3) == 0.0

    def test_two_molecules(self) -> None:
        snn = tanimoto_snn([ASPIRIN, CAFFEINE], k=1)
        assert 0.0 <= snn <= 1.0


class TestComputeAll:
    def test_returns_all_metrics(self) -> None:
        smiles = [ASPIRIN, IBUPROFEN, PARACETAMOL, CAFFEINE, NAPROXEN]
        result = compute_all_diversity_metrics(smiles)
        assert isinstance(result, DiversityResult)
        assert 0.0 <= result.uniqueness <= 1.0
        assert 0.0 <= result.scaffold_diversity <= 1.0
        assert 0.0 <= result.internal_diversity <= 1.0
        assert 0.0 <= result.tanimoto_snn <= 1.0

    def test_with_known_molecules(self) -> None:
        smiles = [ASPIRIN, IBUPROFEN, PARACETAMOL]
        result = compute_all_diversity_metrics(smiles)
        # All unique -> uniqueness = 1.0
        assert result.uniqueness == pytest.approx(1.0)
        # Different scaffolds -> scaffold diversity > 0
        assert result.scaffold_diversity > 0.0
        # Different molecules -> some internal diversity
        assert result.internal_diversity > 0.0

    def test_to_dict(self) -> None:
        smiles = [ASPIRIN, IBUPROFEN, CAFFEINE]
        result = compute_all_diversity_metrics(smiles)
        d = result.to_dict()
        assert "uniqueness" in d
        assert "scaffold_diversity" in d
        assert "internal_diversity" in d
        assert "tanimoto_snn" in d
        assert all(isinstance(v, float) for v in d.values())

    def test_empty_input(self) -> None:
        result = compute_all_diversity_metrics([])
        assert result.uniqueness == 0.0
        assert result.scaffold_diversity == 0.0
        assert result.internal_diversity == 0.0

    def test_single_molecule(self) -> None:
        result = compute_all_diversity_metrics([ASPIRIN])
        assert result.uniqueness == pytest.approx(1.0)
        assert result.scaffold_diversity == pytest.approx(1.0)
