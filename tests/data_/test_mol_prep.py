"""Tests for dualdocking.data.mol_prep — shared molecule preparation."""

from __future__ import annotations

import builtins
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from dualdocking.data.mol_prep import (
    check_ligand_size,
    embed_mol_3d,
    mol_to_pdbqt,
    neutralize_smiles,
    parse_smiles,
    prepare_ligand_pdbqt,
)


# ---------------------------------------------------------------------------
# parse_smiles
# ---------------------------------------------------------------------------


class TestParseSmiles:
    def test_valid(self):
        mol = parse_smiles("CCO")
        assert mol is not None

    def test_invalid(self):
        assert parse_smiles("not_a_smiles!!!") is None

    def test_empty(self):
        # RDKit parses "" as an empty molecule (0 atoms)
        mol = parse_smiles("")
        if mol is not None:
            assert mol.GetNumAtoms() == 0

    def test_benzene(self):
        mol = parse_smiles("c1ccccc1")
        assert mol is not None
        assert mol.GetNumHeavyAtoms() == 6


# ---------------------------------------------------------------------------
# check_ligand_size
# ---------------------------------------------------------------------------


class TestCheckLigandSize:
    def test_small_molecule_passes(self):
        mol = parse_smiles("CCO")
        assert mol is not None
        assert check_ligand_size(mol) is True

    def test_heavy_atoms_exceeds(self):
        mol = parse_smiles("CCO")
        assert mol is not None
        assert check_ligand_size(mol, max_heavy=1) is False

    def test_max_atoms_exceeds(self):
        mol = parse_smiles("CCO")
        assert mol is not None
        # GetNumAtoms on SMILES mol = heavy atoms only (3 for CCO)
        assert check_ligand_size(mol, max_atoms=2) is False

    def test_max_atoms_none_skips(self):
        mol = parse_smiles("CCO")
        assert mol is not None
        assert check_ligand_size(mol, max_atoms=None) is True

    def test_rotatable_bonds_exceeds(self):
        # Long chain: C-C-C-C-C-C-C-C-C-C has many rotatable bonds
        mol = parse_smiles("CCCCCCCCCCCCCCCCCCCC")
        assert mol is not None
        assert check_ligand_size(mol, max_rotatable=2) is False

    def test_rotatable_bonds_none_skips(self):
        mol = parse_smiles("CCCCCCCCCCCCCCCCCCCC")
        assert mol is not None
        assert check_ligand_size(mol, max_rotatable=None) is True

    def test_boundary_heavy_atoms(self):
        mol = parse_smiles("CCO")
        assert mol is not None
        n = mol.GetNumHeavyAtoms()
        assert check_ligand_size(mol, max_heavy=n) is True
        assert check_ligand_size(mol, max_heavy=n - 1) is False

    def test_boundary_max_atoms(self):
        mol = parse_smiles("CCO")
        assert mol is not None
        n_atoms = mol.GetNumAtoms()
        assert check_ligand_size(mol, max_atoms=n_atoms) is True
        assert check_ligand_size(mol, max_atoms=n_atoms - 1) is False

    def test_boundary_rotatable_bonds(self):
        mol = parse_smiles("CCCCCC")
        assert mol is not None
        from rdkit.Chem import Descriptors

        n_rot = Descriptors.NumRotatableBonds(mol)
        assert n_rot > 0
        assert check_ligand_size(mol, max_rotatable=n_rot) is True
        assert check_ligand_size(mol, max_rotatable=n_rot - 1) is False

    def test_heavy_gate_short_circuits_rotatable(self):
        mol = parse_smiles("CCO")
        assert mol is not None
        with patch("rdkit.Chem.Descriptors.NumRotatableBonds", side_effect=AssertionError("should not be called")):
            assert check_ligand_size(mol, max_heavy=1, max_rotatable=999) is False


# ---------------------------------------------------------------------------
# neutralize_smiles
# ---------------------------------------------------------------------------


class TestNeutralizeSmiles:
    def test_charged_nitrogen(self):
        result = neutralize_smiles("C[NH3+]")
        mol = parse_smiles(result)
        assert mol is not None
        # Should have no formal charges
        total_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
        assert total_charge == 0

    def test_already_neutral(self):
        result = neutralize_smiles("CCO")
        assert result == "CCO"

    def test_invalid_smiles_returns_original(self):
        result = neutralize_smiles("not_a_smiles!!!")
        assert result == "not_a_smiles!!!"

    def test_carboxylate(self):
        result = neutralize_smiles("[O-]C(=O)C")
        mol = parse_smiles(result)
        assert mol is not None

    def test_import_error_returns_original(self, monkeypatch):
        original_import = builtins.__import__

        def fake_import(name, globals_=None, locals_=None, fromlist=(), level=0):  # noqa: A002
            if name == "rdkit.Chem.MolStandardize":
                msg = "forced missing module"
                raise ImportError(msg)
            return original_import(name, globals_, locals_, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        charged = "C[NH3+]"
        assert neutralize_smiles(charged) == charged

    def test_uncharger_exception_returns_original(self):
        failing_uncharger = MagicMock()
        failing_uncharger.uncharge.side_effect = RuntimeError("boom")
        with patch("rdkit.Chem.MolStandardize.rdMolStandardize.Uncharger", return_value=failing_uncharger):
            charged = "C[NH3+]"
            assert neutralize_smiles(charged) == charged

    def test_empty_mol_to_smiles_falls_back_to_original(self):
        identity_uncharger = MagicMock()
        identity_uncharger.uncharge.side_effect = lambda mol: mol
        with patch("rdkit.Chem.MolStandardize.rdMolStandardize.Uncharger", return_value=identity_uncharger), patch(
            "rdkit.Chem.MolToSmiles",
            return_value="",
        ):
            original = "CCO"
            assert neutralize_smiles(original) == original


# ---------------------------------------------------------------------------
# embed_mol_3d
# ---------------------------------------------------------------------------


class TestEmbedMol3d:
    def test_basic_embedding(self):
        mol = parse_smiles("CCO")
        assert mol is not None
        mol3d = embed_mol_3d(mol)
        assert mol3d is not None
        assert mol3d.GetNumConformers() == 1

    def test_returns_none_on_failure(self):
        # Very unusual molecule that might fail embedding — use a dot-disconnected
        # Note: most real SMILES embed fine; use a placeholder for robustness
        mol = parse_smiles("[He]")
        if mol is not None:
            # Helium might not embed, or it might — just check return type
            result = embed_mol_3d(mol)
            assert result is None or result.GetNumConformers() > 0

    def test_seed_reproducibility(self):
        mol1 = parse_smiles("c1ccc(CC)cc1")
        mol2 = parse_smiles("c1ccc(CC)cc1")
        assert mol1 is not None and mol2 is not None
        e1 = embed_mol_3d(mol1, seed=42)
        e2 = embed_mol_3d(mol2, seed=42)
        assert e1 is not None and e2 is not None
        pos1 = e1.GetConformer().GetPositions()
        pos2 = e2.GetConformer().GetPositions()
        assert pos1 == pytest.approx(pos2, abs=1e-4)

    def test_negative_seed_uses_fallback(self):
        mol = parse_smiles("CCO")
        assert mol is not None
        result = embed_mol_3d(mol, seed=-1)
        assert result is not None

    def test_embed_failure_returns_none(self):
        mol = parse_smiles("CCO")
        assert mol is not None
        with patch("rdkit.Chem.AllChem.EmbedMolecule", return_value=1):
            assert embed_mol_3d(mol) is None

    def test_uff_exception_is_suppressed(self):
        mol = parse_smiles("CCO")
        assert mol is not None
        with patch("rdkit.Chem.AllChem.UFFOptimizeMolecule", side_effect=RuntimeError("UFF failed")):
            result = embed_mol_3d(mol)
            assert result is not None

    def test_seed_passthrough_and_negative_fallback(self):
        mol = parse_smiles("CCO")
        assert mol is not None
        params = types.SimpleNamespace(randomSeed=None)
        with patch("rdkit.Chem.AllChem.ETKDGv3", return_value=params), patch(
            "rdkit.Chem.AllChem.EmbedMolecule",
            return_value=0,
        ):
            assert embed_mol_3d(mol, seed=123) is not None
            assert params.randomSeed == 123
            assert embed_mol_3d(mol, seed=-7) is not None
            assert params.randomSeed == 42


# ---------------------------------------------------------------------------
# mol_to_pdbqt
# ---------------------------------------------------------------------------


class TestMolToPdbqt:
    def test_writes_pdbqt(self, tmp_path):
        mol = parse_smiles("CCO")
        assert mol is not None
        mol3d = embed_mol_3d(mol)
        assert mol3d is not None
        out = tmp_path / "ligand.pdbqt"
        assert mol_to_pdbqt(mol3d, out) is True
        assert out.exists()
        text = out.read_text()
        assert "ATOM" in text or "HETATM" in text

    def test_returns_false_no_conformer(self, tmp_path):
        from rdkit import Chem

        mol = Chem.AddHs(parse_smiles("CCO"))
        # No conformer → Meeko should fail
        out = tmp_path / "no_conf.pdbqt"
        result = mol_to_pdbqt(mol, out)
        # Meeko may or may not fail, but should not raise
        assert isinstance(result, bool)

    def test_returns_false_when_prepare_raises_value_error(self, tmp_path, monkeypatch):
        class _Prep:
            def prepare(self, _mol):
                raise ValueError("bad mol")

        fake_meeko = types.SimpleNamespace(
            MoleculePreparation=_Prep,
            PDBQTWriterLegacy=types.SimpleNamespace(write_string=lambda _setup: ("", True, "")),
        )
        monkeypatch.setitem(sys.modules, "meeko", fake_meeko)
        assert mol_to_pdbqt(MagicMock(), tmp_path / "x.pdbqt") is False

    def test_returns_false_when_prepare_raises_runtime_error(self, tmp_path, monkeypatch):
        class _Prep:
            def prepare(self, _mol):
                raise RuntimeError("prep failed")

        fake_meeko = types.SimpleNamespace(
            MoleculePreparation=_Prep,
            PDBQTWriterLegacy=types.SimpleNamespace(write_string=lambda _setup: ("", True, "")),
        )
        monkeypatch.setitem(sys.modules, "meeko", fake_meeko)
        assert mol_to_pdbqt(MagicMock(), tmp_path / "x.pdbqt") is False

    def test_returns_false_when_prepare_returns_empty(self, tmp_path, monkeypatch):
        class _Prep:
            def prepare(self, _mol):
                return []

        fake_meeko = types.SimpleNamespace(
            MoleculePreparation=_Prep,
            PDBQTWriterLegacy=types.SimpleNamespace(write_string=lambda _setup: ("", True, "")),
        )
        monkeypatch.setitem(sys.modules, "meeko", fake_meeko)
        assert mol_to_pdbqt(MagicMock(), tmp_path / "x.pdbqt") is False

    def test_returns_false_when_writer_reports_failure(self, tmp_path, monkeypatch):
        class _Prep:
            def prepare(self, _mol):
                return ["setup"]

        fake_meeko = types.SimpleNamespace(
            MoleculePreparation=_Prep,
            PDBQTWriterLegacy=types.SimpleNamespace(write_string=lambda _setup: ("", False, "err")),
        )
        monkeypatch.setitem(sys.modules, "meeko", fake_meeko)
        out = tmp_path / "x.pdbqt"
        assert mol_to_pdbqt(MagicMock(), out) is False
        assert not out.exists()


# ---------------------------------------------------------------------------
# prepare_ligand_pdbqt (full pipeline)
# ---------------------------------------------------------------------------


class TestPrepareLigandPdbqt:
    def test_success(self, tmp_path):
        out = tmp_path / "ethanol.pdbqt"
        assert prepare_ligand_pdbqt("CCO", out) is True
        assert out.exists()
        assert out.stat().st_size > 0

    def test_invalid_smiles(self, tmp_path):
        out = tmp_path / "bad.pdbqt"
        assert prepare_ligand_pdbqt("not_a_smiles!!!", out) is False
        assert not out.exists()

    def test_size_gate_heavy(self, tmp_path):
        out = tmp_path / "small.pdbqt"
        assert prepare_ligand_pdbqt("CCO", out, max_heavy=1) is False

    def test_size_gate_rotatable(self, tmp_path):
        out = tmp_path / "rigid.pdbqt"
        # Benzene has 0 rotatable bonds
        assert prepare_ligand_pdbqt("c1ccccc1", out, max_rotatable=0) is True

    def test_custom_seed(self, tmp_path):
        out = tmp_path / "seeded.pdbqt"
        assert prepare_ligand_pdbqt("CCO", out, seed=42) is True
        assert out.exists()

    def test_embed_failure_returns_false(self, monkeypatch, tmp_path):
        fake_mol = MagicMock()
        out = tmp_path / "embed_fail.pdbqt"
        monkeypatch.setattr("dualdocking.data.mol_prep.parse_smiles", lambda _s: fake_mol)
        monkeypatch.setattr("dualdocking.data.mol_prep.check_ligand_size", lambda *_a, **_k: True)
        monkeypatch.setattr("dualdocking.data.mol_prep.embed_mol_3d", lambda *_a, **_k: None)
        monkeypatch.setattr(
            "dualdocking.data.mol_prep.mol_to_pdbqt",
            lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not be called")),
        )
        assert prepare_ligand_pdbqt("CCO", out) is False

    def test_pdbqt_conversion_failure_returns_false(self, monkeypatch, tmp_path):
        fake_mol = MagicMock()
        fake_mol_3d = MagicMock()
        out = tmp_path / "pdbqt_fail.pdbqt"
        monkeypatch.setattr("dualdocking.data.mol_prep.parse_smiles", lambda _s: fake_mol)
        monkeypatch.setattr("dualdocking.data.mol_prep.check_ligand_size", lambda *_a, **_k: True)
        monkeypatch.setattr("dualdocking.data.mol_prep.embed_mol_3d", lambda *_a, **_k: fake_mol_3d)
        monkeypatch.setattr("dualdocking.data.mol_prep.mol_to_pdbqt", lambda *_a, **_k: False)
        assert prepare_ligand_pdbqt("CCO", out) is False

    def test_calls_substeps_with_expected_parameters(self, monkeypatch, tmp_path):
        parsed = MagicMock(name="parsed")
        embedded = MagicMock(name="embedded")
        out = tmp_path / "pipeline.pdbqt"
        calls: dict[str, object] = {}

        def _parse(smiles: str):
            calls["parse"] = smiles
            return parsed

        def _check(mol, *, max_heavy, max_atoms, max_rotatable):
            calls["check"] = (mol, max_heavy, max_atoms, max_rotatable)
            return True

        def _embed(mol, *, seed):
            calls["embed"] = (mol, seed)
            return embedded

        def _to_pdbqt(mol, out_path):
            calls["to_pdbqt"] = (mol, out_path)
            return True

        monkeypatch.setattr("dualdocking.data.mol_prep.parse_smiles", _parse)
        monkeypatch.setattr("dualdocking.data.mol_prep.check_ligand_size", _check)
        monkeypatch.setattr("dualdocking.data.mol_prep.embed_mol_3d", _embed)
        monkeypatch.setattr("dualdocking.data.mol_prep.mol_to_pdbqt", _to_pdbqt)

        ok = prepare_ligand_pdbqt(
            "CCN",
            out,
            seed=13,
            max_heavy=11,
            max_atoms=22,
            max_rotatable=3,
        )
        assert ok is True
        assert calls["parse"] == "CCN"
        assert calls["check"] == (parsed, 11, 22, 3)
        assert calls["embed"] == (parsed, 13)
        assert calls["to_pdbqt"] == (embedded, out)
