"""Tests for generator_freedpp module."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from dualdocking.generation.freedpp import (
    GenerationConfig,
    _build_freedpp_args,
    _cache_key,
    _ensure_docker_image,
    _parse_output_smiles,
    _resolve_pdb_path,
    _run_freedpp,
    parse_training_log,
)

# --- _cache_key ---


def test_cache_key_deterministic() -> None:
    """Same config produces the same hash."""
    cfg1 = GenerationConfig(target_id="CDK2", seed=42, epochs=200)
    cfg2 = GenerationConfig(target_id="CDK2", seed=42, epochs=200)
    assert _cache_key(cfg1) == _cache_key(cfg2)


def test_cache_key_varies() -> None:
    """Different targets produce different hashes."""
    cfg1 = GenerationConfig(target_id="CDK2", seed=42)
    cfg2 = GenerationConfig(target_id="ERK2", seed=42)
    assert _cache_key(cfg1) != _cache_key(cfg2)


def test_cache_key_varies_by_mode() -> None:
    """single_a vs double produce different hashes."""
    cfg1 = GenerationConfig(target_id="CDK2", reward_mode="single_a")
    cfg2 = GenerationConfig(target_id="CDK2", reward_mode="double", second_target_id="ERK2")
    assert _cache_key(cfg1) != _cache_key(cfg2)


# --- _parse_output_smiles ---


def test_parse_output_smiles(tmp_path: Path) -> None:
    """Mock CSV in FREED++ format yields correct SMILES."""
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
    """No CSV files yields an empty list."""
    smiles = _parse_output_smiles(tmp_path)
    assert smiles == []


# --- _build_freedpp_args ---


def test_build_freedpp_args_single() -> None:
    """single_a mode produces correct CLI args."""
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
    """Double mode includes the same base args (dual handled via env vars)."""
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
    """Resolves PDB by target_id via meta.yaml."""
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


def test_resolve_pdb_path_skips_fixed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When both 1hcl.pdb and 1hcl.fixed.pdb exist, prefer the non-fixed one."""
    pair_dir = tmp_path / "CDK2_ERK2" / "A"
    pair_dir.mkdir(parents=True)

    meta = {"id": "CDK2", "pdb_id": "1HCL", "uniprot": "P24941"}
    (pair_dir / "meta.yaml").write_text(yaml.dump(meta))
    (pair_dir / "1hcl.pdb").write_text("ATOM ...")
    (pair_dir / "1hcl.fixed.pdb").write_text("ATOM fixed ...")

    import dualdocking.generation.freedpp as gen_mod

    monkeypatch.setattr(gen_mod, "DATA_DIR", tmp_path)

    result = _resolve_pdb_path("CDK2")
    assert result.name == "1hcl.pdb"


def test_resolve_pdb_path_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown target raises FileNotFoundError."""
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
        cfg = GenerationConfig(target_id="CDK2", reward_mode="single_b", second_target_id="ERK2")
        cmd = self._run(cfg, pdbqt_b=self.pdbqt_b, center_b="4.0,5.0,6.0", size_b="25.0,25.0,25.0")
        cmd_str = " ".join(cmd)
        assert "receptor_b.pdbqt" in cmd_str
        assert "REWARD_MODE=single_b" in cmd_str
        # single_b should NOT have DUAL_RECEPTOR_B env var
        assert "DUAL_RECEPTOR_B" not in cmd_str

    def test_run_freedpp_double_has_dual_config(self):
        cfg = GenerationConfig(target_id="CDK2", reward_mode="double", second_target_id="ERK2")
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
        cfg = GenerationConfig(target_id="CDK2", reward_mode="single_b", second_target_id="ERK2")
        with pytest.raises(ValueError, match="single_b requires receptor B"):
            self._run(cfg, pdbqt_b=None)


def test_ensure_docker_image_uses_stage0_fetch_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing FREED data should trigger fetch via cli/s0_fetch/fetch_freed_fragments.py."""
    import dualdocking.generation.freedpp as mod

    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(mod, "DOCKER_BIN", "docker")
    monkeypatch.setattr(mod, "FREED_DATA_FILES", (tmp_path / "missing_a.json", tmp_path / "missing_b.csv"))

    dockerfile = tmp_path / "docker" / "freedpp" / "Dockerfile"
    dockerfile.parent.mkdir(parents=True, exist_ok=True)
    dockerfile.write_text("FROM scratch\n")

    docker_calls: list[list[str]] = []
    fetch_calls: list[list[str]] = []

    def _fake_subprocess_run(cmd, **kwargs):
        docker_calls.append(cmd)
        if cmd[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="No such image")
        if cmd[:2] == ["docker", "build"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def _fake_run_project_command(cmd, **kwargs):
        fetch_calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", _fake_subprocess_run)
    monkeypatch.setattr(mod, "run_project_command", _fake_run_project_command)

    _ensure_docker_image("dualdocking/freedpp:test")

    assert fetch_calls, "Expected fetch script call for missing FREED data files"
    assert fetch_calls[0] == ["python", "cli/s0_fetch/fetch_freed_fragments.py"]


# --- parse_training_log ---


def test_parse_training_log_basic(tmp_path: Path) -> None:
    """Extract epoch, reward, and loss from typical FREED++ log lines."""
    log = tmp_path / "freedpp_double_seed_42_adhoc_20260101T000000Z.log"
    log.write_text(
        "[2026-01-01T00:00:00Z] Starting training\n"
        "[2026-01-01T00:01:00Z] Epoch 1 loss=0.523 reward=-6.2\n"
        "[2026-01-01T00:02:00Z] Epoch 2 loss=0.412 reward=-7.1\n"
        "[2026-01-01T00:03:00Z] Epoch 3 loss=0.310 reward=-7.8 score=-8.0\n"
    )
    records = parse_training_log(log)
    assert len(records) == 3
    assert records[0]["epoch"] == 1
    assert records[0]["loss"] == pytest.approx(0.523)
    assert records[0]["reward"] == pytest.approx(-6.2)
    assert records[2]["score"] == pytest.approx(-8.0)


def test_parse_training_log_bracket_format(tmp_path: Path) -> None:
    """Parse [epoch/total] format used by some FREED++ versions."""
    log = tmp_path / "freedpp.log"
    log.write_text(
        "[2026-01-01T00:00:00Z] [1/200] loss=1.5 reward=-3.0\n"
        "[2026-01-01T00:01:00Z] [2/200] loss=1.2 reward=-4.5\n"
    )
    records = parse_training_log(log)
    assert len(records) == 2
    assert records[0]["epoch"] == 1
    assert records[1]["epoch"] == 2


def test_parse_training_log_empty(tmp_path: Path) -> None:
    """Empty log returns empty list."""
    log = tmp_path / "empty.log"
    log.write_text("")
    assert parse_training_log(log) == []


def test_parse_training_log_missing_file(tmp_path: Path) -> None:
    """Missing file returns empty list without error."""
    assert parse_training_log(tmp_path / "nonexistent.log") == []


def test_parse_training_log_no_metrics(tmp_path: Path) -> None:
    """Lines with epoch but no metrics are skipped."""
    log = tmp_path / "freedpp.log"
    log.write_text(
        "[2026-01-01T00:00:00Z] Epoch 1 — starting\n"
        "[2026-01-01T00:01:00Z] Epoch 2 — starting\n"
    )
    assert parse_training_log(log) == []


def test_parse_training_log_duplicate_epoch(tmp_path: Path) -> None:
    """Later occurrence of same epoch updates the record."""
    log = tmp_path / "freedpp.log"
    log.write_text(
        "[2026-01-01T00:00:00Z] Epoch 1 loss=0.5\n"
        "[2026-01-01T00:00:01Z] Epoch 1 reward=-7.0\n"
    )
    records = parse_training_log(log)
    assert len(records) == 1
    assert records[0]["epoch"] == 1
    assert records[0]["loss"] == pytest.approx(0.5)
    assert records[0]["reward"] == pytest.approx(-7.0)
