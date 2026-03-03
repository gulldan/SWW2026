"""Tests for GNINA CNN-based molecular docking scorer."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from dualdocking.scoring.gnina import (
    GninaResult,
    _parse_gnina_output,
    generate_conformer,
    score_batch,
)

# ---------------------------------------------------------------------------
# generate_conformer
# ---------------------------------------------------------------------------


class TestGenerateConformer:
    def test_valid_smiles_returns_mol_with_3d(self) -> None:
        mol = generate_conformer("CCO")
        assert mol is not None
        conf = mol.GetConformer()
        assert conf.Is3D()
        assert conf.GetNumAtoms() > 0

    def test_invalid_smiles_returns_none(self) -> None:
        assert generate_conformer("not_a_smiles") is None

    def test_empty_smiles_returns_none(self) -> None:
        assert generate_conformer("") is None

    def test_different_seeds_produce_different_coords(self) -> None:
        mol1 = generate_conformer("c1ccccc1", seed=1)
        mol2 = generate_conformer("c1ccccc1", seed=999)
        assert mol1 is not None and mol2 is not None
        pos1 = mol1.GetConformer().GetPositions()
        pos2 = mol2.GetConformer().GetPositions()
        # Positions should differ (different random seeds)
        assert not np.allclose(pos1, pos2, atol=1e-3)

    def test_complex_molecule(self) -> None:
        # Aspirin
        mol = generate_conformer("CC(=O)Oc1ccccc1C(=O)O")
        assert mol is not None
        assert mol.GetConformer().Is3D()


# ---------------------------------------------------------------------------
# _parse_gnina_output
# ---------------------------------------------------------------------------


class TestParseGninaOutput:
    def test_parse_sdf_with_all_properties(self, tmp_path: Path) -> None:
        """SDF with CNNscore, CNNaffinity, and minimizedAffinity should parse all 3."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        from rdkit.Chem import AllChem

        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("CNNscore", "0.85")
        mol.SetProp("CNNaffinity", "-6.3")
        mol.SetProp("minimizedAffinity", "-5.8")

        sdf_path = tmp_path / "output.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()

        results = _parse_gnina_output(sdf_path)
        assert len(results) == 1
        cnn_score, cnn_affinity, vina_score = results[0]
        assert abs(cnn_score - 0.85) < 1e-6
        assert abs(cnn_affinity - (-6.3)) < 1e-6
        assert abs(vina_score - (-5.8)) < 1e-6

    def test_parse_sdf_missing_vina(self, tmp_path: Path) -> None:
        """SDF without minimizedAffinity returns NaN for vina_score."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        from rdkit.Chem import AllChem

        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("CNNscore", "0.85")
        mol.SetProp("CNNaffinity", "-6.3")

        sdf_path = tmp_path / "output.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()

        results = _parse_gnina_output(sdf_path)
        assert len(results) == 1
        cnn_score, cnn_affinity, vina_score = results[0]
        assert abs(cnn_score - 0.85) < 1e-6
        assert abs(cnn_affinity - (-6.3)) < 1e-6
        assert np.isnan(vina_score)

    def test_parse_sdf_missing_all_properties(self, tmp_path: Path) -> None:
        """SDF without any scoring properties should return all NaN."""
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        from rdkit.Chem import AllChem

        AllChem.EmbedMolecule(mol, randomSeed=42)

        sdf_path = tmp_path / "output.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()

        results = _parse_gnina_output(sdf_path)
        assert len(results) == 1
        assert np.isnan(results[0][0])
        assert np.isnan(results[0][1])
        assert np.isnan(results[0][2])

    def test_parse_multiple_molecules(self, tmp_path: Path) -> None:
        """Multiple molecules in SDF should all be parsed."""
        sdf_path = tmp_path / "output.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        for i, smi in enumerate(["CCO", "CCC", "CCCC"]):
            mol = Chem.MolFromSmiles(smi)
            mol = Chem.AddHs(mol)
            from rdkit.Chem import AllChem

            AllChem.EmbedMolecule(mol, randomSeed=42)
            mol.SetProp("CNNscore", str(0.5 + i * 0.1))
            mol.SetProp("CNNaffinity", str(-5.0 - i))
            mol.SetProp("minimizedAffinity", str(-4.0 - i * 0.5))
            writer.write(mol)
        writer.close()

        results = _parse_gnina_output(sdf_path)
        assert len(results) == 3
        assert abs(results[0][0] - 0.5) < 1e-6
        assert abs(results[2][1] - (-7.0)) < 1e-6
        assert abs(results[1][2] - (-4.5)) < 1e-6


# ---------------------------------------------------------------------------
# score_batch (with mocked Docker)
# ---------------------------------------------------------------------------


class TestScoreBatch:
    def _make_receptor(self, tmp_path: Path) -> Path:
        """Create a minimal PDB file for testing."""
        pdb = tmp_path / "receptor.pdb"
        pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n")
        return pdb

    def _write_gnina_output(self, output_dir: Path, smiles_list: list[str]) -> None:
        """Write a fake GNINA output SDF with CNN and Vina properties."""
        sdf_path = output_dir / "output.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol = Chem.AddHs(mol)
            from rdkit.Chem import AllChem

            AllChem.EmbedMolecule(mol, randomSeed=42)
            mol.SetProp("CNNscore", str(0.9 - i * 0.1))
            mol.SetProp("CNNaffinity", str(-7.0 + i * 0.5))
            mol.SetProp("minimizedAffinity", str(-6.5 + i * 0.3))
            mol.SetProp("_Name", f"mol_{i}")
            writer.write(mol)
        writer.close()

    def test_score_batch_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """score_batch with successful Docker run returns valid results."""
        receptor = self._make_receptor(tmp_path)
        smiles = ["CCO", "CCC"]

        def fake_run(cmd, **kwargs):
            for i, arg in enumerate(cmd):
                if arg == "-v" and "/work" in cmd[i + 1]:
                    work_dir = Path(cmd[i + 1].split(":")[0])
                    self._write_gnina_output(work_dir, smiles)
                    break
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        results = score_batch(smiles, receptor, center=(0.0, 0.0, 0.0), box_size=25.0)

        assert len(results) == 2
        assert all(isinstance(r, GninaResult) for r in results)
        assert results[0].valid
        assert results[1].valid
        assert abs(results[0].cnn_score - 0.9) < 1e-6
        assert abs(results[0].cnn_affinity - (-7.0)) < 1e-6
        assert abs(results[0].vina_score - (-6.5)) < 1e-6

    def test_score_batch_docker_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """score_batch returns invalid results when Docker fails."""
        receptor = self._make_receptor(tmp_path)

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="error")

        monkeypatch.setattr(subprocess, "run", fake_run)

        results = score_batch(["CCO"], receptor, center=(0.0, 0.0, 0.0))
        assert len(results) == 1
        assert not results[0].valid
        assert np.isnan(results[0].cnn_score)
        assert np.isnan(results[0].vina_score)

    def test_score_batch_timeout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """score_batch returns invalid results on timeout."""
        receptor = self._make_receptor(tmp_path)

        def fake_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 300))

        monkeypatch.setattr(subprocess, "run", fake_run)

        results = score_batch(["CCO"], receptor, center=(0.0, 0.0, 0.0), timeout=1)
        assert len(results) == 1
        assert not results[0].valid

    def test_score_batch_all_invalid_smiles(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """score_batch with all invalid SMILES returns all-invalid results without calling Docker."""
        receptor = self._make_receptor(tmp_path)
        run_called = False

        def fake_run(cmd, **kwargs):
            nonlocal run_called
            run_called = True
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        results = score_batch(["not_valid", "also_bad"], receptor, center=(0.0, 0.0, 0.0))
        assert len(results) == 2
        assert not any(r.valid for r in results)
        assert not run_called  # Docker should not be called

    def test_score_batch_mixed_valid_invalid(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """score_batch handles mix of valid and invalid SMILES correctly."""
        receptor = self._make_receptor(tmp_path)
        valid_smiles = ["CCO"]  # Only first one is valid

        def fake_run(cmd, **kwargs):
            for i, arg in enumerate(cmd):
                if arg == "-v" and "/work" in cmd[i + 1]:
                    work_dir = Path(cmd[i + 1].split(":")[0])
                    self._write_gnina_output(work_dir, valid_smiles)
                    break
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        results = score_batch(["CCO", "not_valid"], receptor, center=(0.0, 0.0, 0.0))
        assert len(results) == 2
        assert results[0].valid
        assert not results[1].valid

    def test_score_batch_autobox_ligand(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """score_batch uses autobox_ligand mount when provided."""
        receptor = self._make_receptor(tmp_path)
        ligand = tmp_path / "ligand.pdb"
        ligand.write_text("HETATM    1  C   LIG A   1       1.0   2.0   3.0  1.00  0.00\nEND\n")

        captured_cmd: list[str] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            for i, arg in enumerate(cmd):
                if arg == "-v" and "/work" in cmd[i + 1]:
                    work_dir = Path(cmd[i + 1].split(":")[0])
                    self._write_gnina_output(work_dir, ["CCO"])
                    break
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        results = score_batch(["CCO"], receptor, autobox_ligand=ligand)
        assert len(results) == 1
        # Verify autobox mount and argument are in the command
        assert "/autobox:ro" in " ".join(captured_cmd)
        assert "--autobox_ligand" in captured_cmd

    def test_score_batch_no_binding_box_returns_invalid(self, tmp_path: Path) -> None:
        """score_batch without binding box info returns all-invalid results."""
        receptor = self._make_receptor(tmp_path)
        results = score_batch(["CCO"], receptor)
        assert len(results) == 1
        assert not results[0].valid

    def test_score_batch_no_score_only_flag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """score_batch should NOT pass --score_only (runs full docking)."""
        receptor = self._make_receptor(tmp_path)
        captured_cmd: list[str] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            for i, arg in enumerate(cmd):
                if arg == "-v" and "/work" in cmd[i + 1]:
                    work_dir = Path(cmd[i + 1].split(":")[0])
                    self._write_gnina_output(work_dir, ["CCO"])
                    break
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        score_batch(["CCO"], receptor, center=(0.0, 0.0, 0.0))
        assert "--score_only" not in captured_cmd

    def test_score_batch_exhaustiveness_in_cmd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """score_batch passes --exhaustiveness flag."""
        receptor = self._make_receptor(tmp_path)
        captured_cmd: list[str] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            for i, arg in enumerate(cmd):
                if arg == "-v" and "/work" in cmd[i + 1]:
                    work_dir = Path(cmd[i + 1].split(":")[0])
                    self._write_gnina_output(work_dir, ["CCO"])
                    break
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        score_batch(["CCO"], receptor, center=(0.0, 0.0, 0.0), exhaustiveness=16)
        assert "--exhaustiveness" in captured_cmd
        idx = captured_cmd.index("--exhaustiveness")
        assert captured_cmd[idx + 1] == "16"

    def test_score_batch_rectangular_box(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """score_batch accepts tuple box_size for rectangular box."""
        receptor = self._make_receptor(tmp_path)
        captured_cmd: list[str] = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            for i, arg in enumerate(cmd):
                if arg == "-v" and "/work" in cmd[i + 1]:
                    work_dir = Path(cmd[i + 1].split(":")[0])
                    self._write_gnina_output(work_dir, ["CCO"])
                    break
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        score_batch(["CCO"], receptor, center=(1.0, 2.0, 3.0), box_size=(20.0, 25.0, 30.0))
        assert "--size_x" in captured_cmd
        idx_x = captured_cmd.index("--size_x")
        idx_y = captured_cmd.index("--size_y")
        idx_z = captured_cmd.index("--size_z")
        assert captured_cmd[idx_x + 1] == "20.0"
        assert captured_cmd[idx_y + 1] == "25.0"
        assert captured_cmd[idx_z + 1] == "30.0"


# ---------------------------------------------------------------------------
# GninaResult dataclass
# ---------------------------------------------------------------------------


class TestGninaResult:
    def test_frozen(self) -> None:
        r = GninaResult("CCO", 0.9, -7.0, -6.5, True)
        with pytest.raises(AttributeError):
            r.cnn_score = 0.5  # type: ignore[misc]

    def test_fields(self) -> None:
        r = GninaResult("CCO", 0.85, -6.3, -5.8, True)
        assert r.smiles == "CCO"
        assert r.cnn_score == 0.85
        assert r.cnn_affinity == -6.3
        assert r.vina_score == -5.8
        assert r.valid is True

    def test_invalid_result_has_nan_vina(self) -> None:
        from dualdocking.scoring.gnina import _invalid_result

        r = _invalid_result("CCO")
        assert not r.valid
        assert np.isnan(r.cnn_score)
        assert np.isnan(r.cnn_affinity)
        assert np.isnan(r.vina_score)
