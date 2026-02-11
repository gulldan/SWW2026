"""Tests for drug-likeness filters."""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from dualdocking.data.filters import (
    FilterResult,
    FilterStats,
    compute_filters,
    filter_one,
    lipinski,
    load_filter_config,
    pains_check,
    qed_check,
    sa_check,
    veber_check,
)

# Aspirin: small, drug-like, no PAINS
_ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"

# Large molecule (MW > 500): hexapeptide-like
_LARGE_MOL = "CC(C)CC(NC(=O)C(CC(=O)O)NC(=O)C(CC(C)C)NC(=O)C(CC(C)C)NC(=O)C(CC(C)C)NC(=O)C(CC(C)C)NC(=O)C(CC(C)C)N)C(=O)O"

# Known PAINS substructure: rhodanine (2-thioxothiazolidin-4-one)
_PAINS_MOL = "O=C1CSC(=S)N1"

_DEFAULT_CFG = {
    "lipinski": {"enabled": True, "mw_max": 500.0, "logp_max": 5.0, "hbd_max": 5, "hba_max": 10},
    "qed": {"enabled": True, "threshold_min": 0.5},
    "pains": {"enabled": True},
    "sa_score": {"enabled": True, "threshold_max": 5.0},
    "veber": {"enabled": True, "rotatable_bonds_max": 10, "tpsa_max": 140.0},
}


class TestLipinski:
    def test_aspirin_passes(self):
        mol = Chem.MolFromSmiles(_ASPIRIN)
        assert lipinski(mol, _DEFAULT_CFG["lipinski"]) is True

    def test_large_mol_fails_mw(self):
        mol = Chem.MolFromSmiles(_LARGE_MOL)
        assert mol is not None
        from rdkit.Chem import Descriptors

        mw = Descriptors.MolWt(mol)
        assert mw > 500.0
        assert lipinski(mol, _DEFAULT_CFG["lipinski"]) is False

    def test_disabled_always_passes(self):
        mol = Chem.MolFromSmiles(_LARGE_MOL)
        assert lipinski(mol, {"enabled": False}) is True


class TestQED:
    def test_aspirin_passes(self):
        mol = Chem.MolFromSmiles(_ASPIRIN)
        assert qed_check(mol, _DEFAULT_CFG["qed"]) is True

    def test_disabled_always_passes(self):
        mol = Chem.MolFromSmiles(_ASPIRIN)
        assert qed_check(mol, {"enabled": False}) is True


class TestPAINS:
    def test_aspirin_no_pains(self):
        mol = Chem.MolFromSmiles(_ASPIRIN)
        assert pains_check(mol) is True

    def test_rhodanine_is_pains(self):
        mol = Chem.MolFromSmiles(_PAINS_MOL)
        assert mol is not None
        assert pains_check(mol) is False


class TestSAScore:
    def test_aspirin_sa_range(self):
        mol = Chem.MolFromSmiles(_ASPIRIN)
        from rdkit.Contrib.SA_Score import sascorer

        score = sascorer.calculateScore(mol)
        assert 1.0 <= score <= 10.0

    def test_aspirin_passes(self):
        mol = Chem.MolFromSmiles(_ASPIRIN)
        assert sa_check(mol, _DEFAULT_CFG["sa_score"]) is True

    def test_disabled_always_passes(self):
        mol = Chem.MolFromSmiles(_ASPIRIN)
        assert sa_check(mol, {"enabled": False}) is True


class TestFilterOne:
    def test_aspirin_passes_all(self):
        r = filter_one(_ASPIRIN, _DEFAULT_CFG)
        assert isinstance(r, FilterResult)
        assert r.passes_lipinski is True
        assert r.passes_qed is True
        assert r.passes_pains is True
        assert r.passes_sa is True
        assert r.passes_all is True
        assert r.mw > 0
        assert r.qed_score > 0
        assert 1.0 <= r.sa_score <= 10.0

    def test_invalid_smiles(self):
        r = filter_one("NOT_A_MOLECULE", _DEFAULT_CFG)
        assert r.passes_all is False
        assert r.passes_lipinski is False

    def test_pains_mol_fails(self):
        r = filter_one(_PAINS_MOL, _DEFAULT_CFG)
        assert r.passes_pains is False
        assert r.pains_matches > 0
        assert r.passes_all is False


class TestComputeFilters:
    def test_batch(self):
        smiles = [_ASPIRIN, "NOT_VALID", _PAINS_MOL]
        results, stats = compute_filters(smiles, _DEFAULT_CFG)
        assert len(results) == 3
        assert isinstance(stats, FilterStats)
        assert stats.n_total == 3
        assert stats.n_valid == 2  # aspirin + rhodanine
        # Aspirin passes all; rhodanine fails PAINS; invalid fails all
        assert stats.n_pass_all >= 1
        assert stats.n_pass_all <= 2

    def test_stats_to_dict(self):
        _, stats = compute_filters([_ASPIRIN], _DEFAULT_CFG)
        d = stats.to_dict()
        assert "n_total" in d
        assert "n_pass_all" in d
        assert d["n_total"] == 1


class TestVeber:
    def test_aspirin_passes(self):
        mol = Chem.MolFromSmiles(_ASPIRIN)
        assert veber_check(mol, _DEFAULT_CFG["veber"]) is True

    def test_rotatable_bonds_boundary_pass(self):
        """A molecule with exactly 10 rotatable bonds should pass."""
        # Decane: CH3-(CH2)8-CH3 has ~7 rotatable bonds; use a longer chain
        # Undecanol: long aliphatic chain
        from rdkit.Chem import Descriptors

        # Build a molecule with exactly 10 rotatable bonds by trial
        # n-tridecane (C13) has 10 rotatable bonds
        smi = "CCCCCCCCCCCCC"
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        rot = Descriptors.NumRotatableBonds(mol)
        assert rot == 10
        assert veber_check(mol, {"enabled": True, "rotatable_bonds_max": 10, "tpsa_max": 140.0}) is True

    def test_rotatable_bonds_boundary_fail(self):
        """A molecule with 11 rotatable bonds should fail."""
        from rdkit.Chem import Descriptors

        # n-tetradecane (C14) has 11 rotatable bonds
        smi = "CCCCCCCCCCCCCC"
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        rot = Descriptors.NumRotatableBonds(mol)
        assert rot == 11
        assert veber_check(mol, {"enabled": True, "rotatable_bonds_max": 10, "tpsa_max": 140.0}) is False

    def test_tpsa_boundary(self):
        """Molecule with TPSA > 140 should fail Veber."""
        from rdkit.Chem import Descriptors

        # Sucrose: disaccharide with many hydroxyl groups, TPSA ~189
        smi = "OC[C@H]1OC(O[C@@]2(CO)OC[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O"
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None
        tpsa = Descriptors.TPSA(mol)
        assert tpsa > 140.0
        assert veber_check(mol, {"enabled": True, "rotatable_bonds_max": 10, "tpsa_max": 140.0}) is False

    def test_disabled_always_passes(self):
        """When veber.enabled=false, always passes regardless of properties."""
        from rdkit.Chem import Descriptors

        smi = "CCCCCCCCCCCCCC"  # 11 rotatable bonds
        mol = Chem.MolFromSmiles(smi)
        assert Descriptors.NumRotatableBonds(mol) == 11
        assert veber_check(mol, {"enabled": False}) is True


# Imatinib: a real drug that should pass all filters including Veber
_IMATINIB = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"


class TestImatinib:
    def test_imatinib_passes_veber_and_lipinski(self):
        """Imatinib passes Lipinski, Veber, PAINS, SA — but QED < 0.5."""
        r = filter_one(_IMATINIB, _DEFAULT_CFG)
        assert r.passes_lipinski is True
        assert r.passes_veber is True
        assert r.passes_pains is True
        assert r.passes_sa is True
        # Imatinib has QED ~0.39, below 0.5 threshold
        assert r.passes_qed is False
        assert r.passes_all is False

    def test_imatinib_passes_all_with_relaxed_qed(self):
        """Imatinib passes all filters when QED threshold is lowered."""
        cfg = {**_DEFAULT_CFG, "qed": {"enabled": True, "threshold_min": 0.3}}
        r = filter_one(_IMATINIB, cfg)
        assert r.passes_all is True


class TestVeberInFilterResult:
    def test_passes_veber_in_result(self):
        r = filter_one(_ASPIRIN, _DEFAULT_CFG)
        assert hasattr(r, "passes_veber")
        assert isinstance(r.passes_veber, bool)
        assert hasattr(r, "rotatable_bonds")
        assert hasattr(r, "tpsa")
        assert r.rotatable_bonds >= 0
        assert r.tpsa >= 0

    def test_invalid_smiles_veber_fields(self):
        import math

        r = filter_one("NOT_A_MOLECULE", _DEFAULT_CFG)
        assert r.passes_veber is False
        assert math.isnan(r.rotatable_bonds)
        assert math.isnan(r.tpsa)


class TestChargedSpecies:
    def test_glycine_zwitterion(self):
        """Charged species (glycine zwitterion) should be parseable and filtered."""
        smi = "[NH3+]CC(=O)[O-]"
        r = filter_one(smi, _DEFAULT_CFG)
        assert isinstance(r, FilterResult)
        assert r.mw > 0
        assert isinstance(r.passes_veber, bool)


class TestLoadConfig:
    def test_load_from_file(self, tmp_path: Path):
        cfg_path = tmp_path / "filters.yaml"
        cfg_path.write_text("filters:\n  lipinski:\n    enabled: true\n    mw_max: 600.0\n  qed:\n    enabled: false\n")
        cfg = load_filter_config(cfg_path)
        assert cfg["lipinski"]["mw_max"] == 600.0
        assert cfg["qed"]["enabled"] is False
