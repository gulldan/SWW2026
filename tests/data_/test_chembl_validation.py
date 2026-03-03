"""Tests for chembl_validation module."""

import json

import numpy as np
import pytest

from dualdocking.data.chembl_validation import (
    TARGET_CHEMBL_MAP,
    compute_nearest_tanimoto,
    enrichment_factor,
    find_dual_actives,
)


class TestTargetChemblMap:
    def test_all_core_targets_mapped(self):
        core_targets = {"CDK2", "ERK2", "AChE", "BChE", "BACE1", "GSK3B", "CDK5"}
        for t in core_targets:
            assert t in TARGET_CHEMBL_MAP, f"Missing ChEMBL mapping for {t}"

    def test_all_values_start_with_chembl(self):
        for target_id, chembl_id in TARGET_CHEMBL_MAP.items():
            assert chembl_id.startswith("CHEMBL"), f"{target_id} -> {chembl_id}"


class TestFindDualActives:
    def test_intersection(self):
        a = ["CCO", "CC(=O)O", "c1ccccc1"]
        b = ["CC(=O)O", "c1ccc(O)cc1", "c1ccccc1"]
        dual = find_dual_actives(a, b)
        assert set(dual) == {"CC(=O)O", "c1ccccc1"}

    def test_no_overlap(self):
        assert find_dual_actives(["CCO"], ["c1ccccc1"]) == []

    def test_empty_input(self):
        assert find_dual_actives([], ["CCO"]) == []
        assert find_dual_actives(["CCO"], []) == []

    def test_sorted_output(self):
        result = find_dual_actives(["ZZZ", "AAA", "MMM"], ["MMM", "AAA"])
        assert result == sorted(result)


class TestComputeNearestTanimoto:
    def test_identical_molecule(self):
        smi = "c1ccccc1"
        result = compute_nearest_tanimoto([smi], [smi])
        assert len(result) == 1
        assert result[0] == pytest.approx(1.0)

    def test_different_molecules(self):
        query = ["c1ccccc1"]  # benzene
        ref = ["CCCCCCCCCCCCCCCC"]  # hexadecane
        result = compute_nearest_tanimoto(query, ref)
        assert len(result) == 1
        assert 0.0 <= result[0] < 1.0

    def test_invalid_smiles_returns_nan(self):
        result = compute_nearest_tanimoto(["INVALID_SMILES"], ["c1ccccc1"])
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_empty_reference(self):
        result = compute_nearest_tanimoto(["c1ccccc1"], [])
        assert np.isnan(result[0])

    def test_multiple_queries(self):
        queries = ["c1ccccc1", "CCO", "CC(=O)O"]
        refs = ["c1ccccc1", "CC(=O)O"]
        result = compute_nearest_tanimoto(queries, refs)
        assert len(result) == 3
        assert result[0] == pytest.approx(1.0)  # benzene == benzene
        assert result[2] == pytest.approx(1.0)  # acetic acid == acetic acid
        assert 0.0 <= result[1] <= 1.0  # ethanol has some similarity


class TestEnrichmentFactor:
    def test_perfect_enrichment(self):
        # All hits are in top-K
        sims = np.array([0.9, 0.8, 0.7, 0.1, 0.05, 0.02])
        ef = enrichment_factor(sims, k=3, threshold=0.5)
        # 3 hits total, all in top-3: EF = (3/3) / (3/6) = 2.0
        assert ef == pytest.approx(2.0)

    def test_no_hits(self):
        sims = np.array([0.1, 0.2, 0.3])
        ef = enrichment_factor(sims, k=2, threshold=0.5)
        assert ef == 0.0

    def test_all_hits(self):
        sims = np.array([0.9, 0.8, 0.7, 0.6])
        ef = enrichment_factor(sims, k=2, threshold=0.5)
        # 4 hits total, 2 in top-2: EF = (2/2) / (4/4) = 1.0
        assert ef == pytest.approx(1.0)

    def test_nan_values_excluded(self):
        sims = np.array([0.9, np.nan, 0.1, np.nan])
        ef = enrichment_factor(sims, k=1, threshold=0.5)
        # Valid: [0.9, 0.1], 1 hit total, 1 in top-1: EF = (1/1) / (1/2) = 2.0
        assert ef == pytest.approx(2.0)

    def test_k_zero(self):
        sims = np.array([0.9, 0.8])
        assert enrichment_factor(sims, k=0, threshold=0.5) == 0.0

    def test_empty_input(self):
        assert enrichment_factor(np.array([]), k=5, threshold=0.5) == 0.0


class TestFetchActives:
    """Tests for fetch_actives (network-dependent, use cache for CI)."""

    def test_unknown_target_raises(self):
        from dualdocking.data.chembl_validation import fetch_actives

        with pytest.raises(ValueError, match="Unknown target_id"):
            fetch_actives("NONEXISTENT_TARGET")

    def test_cached_fetch(self, tmp_path):
        from dualdocking.data.chembl_validation import fetch_actives

        # Pre-populate cache
        cache_data = ["CCO", "c1ccccc1"]
        cache_file = tmp_path / "BACE1_actives.json"
        cache_file.write_text(json.dumps(cache_data))

        result = fetch_actives("BACE1", cache_dir=tmp_path)
        assert result == cache_data
