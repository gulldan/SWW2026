"""Tests for generator_freedpp module."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from dualdocking.generation.freedpp import (
    GenerationConfig,
    _build_freedpp_args,
    _cache_key,
    _parse_output_smiles,
    _resolve_pdb_path,
    _run_freedpp,
)

# --- _cache_key ---


def test_cache_key_deterministic() -> None:
    """Одинаковый config → одинаковый hash."""
    cfg1 = GenerationConfig(target_id="CDK2", seed=42, epochs=200)
    cfg2 = GenerationConfig(target_id="CDK2", seed=42, epochs=200)
    assert _cache_key(cfg1) == _cache_key(cfg2)


def test_cache_key_varies() -> None:
    """Разные targets → разные hash."""
    cfg1 = GenerationConfig(target_id="CDK2", seed=42)
    cfg2 = GenerationConfig(target_id="ERK2", seed=42)
    assert _cache_key(cfg1) != _cache_key(cfg2)


def test_cache_key_varies_by_mode() -> None:
    """Single_a vs double → разные hash."""
    cfg1 = GenerationConfig(target_id="CDK2", reward_mode="single_a")
    cfg2 = GenerationConfig(target_id="CDK2", reward_mode="double", second_target_id="ERK2")
    assert _cache_key(cfg1) != _cache_key(cfg2)


# --- _parse_output_smiles ---


def test_parse_output_smiles(tmp_path: Path) -> None:
    """Mock CSV в формате FREED++ → корректные SMILES."""
    mols_dir = tmp_path / "output" / "run" / "mols"
    mols_dir.mkdir(parents=True)

    csv_path = mols_dir / "sample_000200.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "score"])
        writer.writerow(["c1ccccc1", "-7.5"])
        writer.writerow(["CC(=O)O", "-6.2"])

    smiles = _parse_output_smiles(tmp_path)
    assert smiles == ["c1ccccc1", "CC(=O)O"]


def test_parse_output_smiles_empty(tmp_path: Path) -> None:
    """Нет CSV файлов → пустой список."""
    smiles = _parse_output_smiles(tmp_path)
    assert smiles == []


# --- _build_freedpp_args ---


def test_build_freedpp_args_single() -> None:
    """Single_a mode — корректные CLI args."""
    cfg = GenerationConfig(
        target_id="CDK2",
        n_molecules=100,
        seed=42,
        epochs=50,
        num_sub_proc=4,
    )
    pdbqt_a = Path("/workspace/receptor.pdbqt")
    args = _build_freedpp_args(cfg, pdbqt_a, "1.0,2.0,3.0", "20.0,20.0,20.0")

    assert "--commands" in args
    assert "train,sample" in args
    assert "--receptor" in args
    assert "/workspace/receptor.pdbqt" in args
    assert "--epochs" in args
    assert "50" in args
    assert "--num_mols" in args
    assert "100" in args
    assert "--seed" in args
    assert "42" in args


def test_build_freedpp_args_double() -> None:
    """Double mode args содержат те же базовые аргументы (dual обрабатывается через env vars)."""
    cfg = GenerationConfig(
        target_id="CDK2",
        reward_mode="double",
        second_target_id="ERK2",
    )
    pdbqt_a = Path("/workspace/receptor.pdbqt")
    args = _build_freedpp_args(cfg, pdbqt_a, "1.0,2.0,3.0", "20.0,20.0,20.0")
    assert "--receptor" in args


# --- _resolve_pdb_path ---


def test_resolve_pdb_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Находит PDB по target_id через meta.yaml."""
    pair_dir = tmp_path / "CDK2_ERK2" / "A"
    pair_dir.mkdir(parents=True)

    meta = {"id": "CDK2", "pdb_id": "1HCL", "uniprot": "P24941"}
    (pair_dir / "meta.yaml").write_text(yaml.dump(meta))
    pdb_path = pair_dir / "1hcl.pdb"
    pdb_path.write_text("ATOM ...")

    import dualdocking.generation.freedpp as gen_mod

    monkeypatch.setattr(gen_mod, "DATA_DIR", tmp_path)

    result = _resolve_pdb_path("CDK2")
    assert result == pdb_path


def test_resolve_pdb_path_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Target не найден → FileNotFoundError."""
    import dualdocking.generation.freedpp as gen_mod

    monkeypatch.setattr(gen_mod, "DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="NONEXISTENT"):
        _resolve_pdb_path("NONEXISTENT")


def test_parse_output_smiles_no_double_count(tmp_path: Path) -> None:
    """When both JSON and CSV exist with overlapping SMILES, no double-counting."""
    import json as json_mod

    mols_dir = tmp_path / "output" / "run" / "mols"
    mols_dir.mkdir(parents=True)

    # Create JSON with some SMILES
    json_path = mols_dir / "sample_000200.json"
    json_path.write_text(json_mod.dumps(["c1ccccc1", "CCO", "CC(=O)O"]))

    # Create CSV with overlapping SMILES
    csv_path = mols_dir / "sample_000200.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "score"])
        writer.writerow(["c1ccccc1", "-7.5"])
        writer.writerow(["CCCC", "-6.0"])

    smiles = _parse_output_smiles(tmp_path)
    # JSON is primary; CSV should NOT be read when JSON produced results
    assert smiles == ["c1ccccc1", "CCO", "CC(=O)O"]


# --- _run_freedpp branching ---


class _FakePopen:
    """Captures the docker command for inspection."""

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.returncode = 0
        self.stdout = iter([])

    def wait(self, timeout=None):
        pass


class TestRunFreedppBranching:
    """Test _run_freedpp receptor selection and env var logic."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.workspace = tmp_path / "workspace"
        self.workspace.mkdir()
        self.pdbqt_a = tmp_path / "receptor_a.pdbqt"
        self.pdbqt_a.write_text("ATOM ...")
        self.pdbqt_b = tmp_path / "receptor_b.pdbqt"
        self.pdbqt_b.write_text("ATOM ...")

    def _run(self, config, pdbqt_b=None, center_b=None, size_b=None):
        captured = {}
        real_popen = _FakePopen

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            return real_popen(cmd, **kwargs)

        with (
            patch("dualdocking.generation.freedpp.subprocess.Popen", side_effect=fake_popen),
            patch("dualdocking.generation.freedpp.tqdm", side_effect=lambda *a, **kw: None),
        ):
            _run_freedpp(
                config,
                self.workspace,
                self.pdbqt_a,
                "1.0,2.0,3.0",
                "20.0,20.0,20.0",
                pdbqt_b=pdbqt_b,
                center_b=center_b,
                size_b=size_b,
            )
        return captured["cmd"]

    def test_run_freedpp_single_a_uses_receptor_a(self):
        cfg = GenerationConfig(target_id="CDK2", reward_mode="single_a")
        cmd = self._run(cfg)
        cmd_str = " ".join(cmd)
        assert "receptor_a.pdbqt" in cmd_str
        assert "REWARD_MODE=single_a" in cmd_str

    def test_run_freedpp_single_b_uses_receptor_b(self):
        cfg = GenerationConfig(
            target_id="CDK2", reward_mode="single_b", second_target_id="ERK2"
        )
        cmd = self._run(cfg, pdbqt_b=self.pdbqt_b, center_b="4.0,5.0,6.0", size_b="25.0,25.0,25.0")
        cmd_str = " ".join(cmd)
        assert "receptor_b.pdbqt" in cmd_str
        assert "REWARD_MODE=single_b" in cmd_str
        # single_b should NOT have DUAL_RECEPTOR_B env var
        assert "DUAL_RECEPTOR_B" not in cmd_str

    def test_run_freedpp_double_has_dual_config(self):
        cfg = GenerationConfig(
            target_id="CDK2", reward_mode="double", second_target_id="ERK2"
        )
        cmd = self._run(cfg, pdbqt_b=self.pdbqt_b, center_b="4.0,5.0,6.0", size_b="25.0,25.0,25.0")
        cmd_str = " ".join(cmd)
        assert "DUAL_RECEPTOR_B" in cmd_str
        assert "DUAL_CONFIG" in cmd_str
        # double uses receptor A as primary
        assert "receptor_a.pdbqt" in cmd_str
        # dual_config.json should be written
        dual_cfg_path = self.workspace / "dual_config.json"
        assert dual_cfg_path.exists()
        data = json.loads(dual_cfg_path.read_text())
        assert "receptor_b" in data

    def test_run_freedpp_single_b_fails_without_receptor_b(self):
        cfg = GenerationConfig(
            target_id="CDK2", reward_mode="single_b", second_target_id="ERK2"
        )
        with pytest.raises(ValueError, match="single_b requires receptor B"):
            self._run(cfg, pdbqt_b=None)
