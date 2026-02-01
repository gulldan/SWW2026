"""Tests for generator_freedpp module."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from dualdocking.generator_freedpp import (
    GenerationConfig,
    _build_freedpp_args,
    _cache_key,
    _parse_output_smiles,
    _resolve_pdb_path,
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
    """Single vs double → разные hash."""
    cfg1 = GenerationConfig(target_id="CDK2", reward_mode="single")
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
    """Single mode — корректные CLI args."""
    cfg = GenerationConfig(
        target_id="CDK2",
        n_molecules=100,
        seed=42,
        epochs=50,
        num_sub_proc=4,
    )
    workspace = Path("/workspace")
    pdbqt_a = Path("/workspace/receptor.pdbqt")
    args = _build_freedpp_args(cfg, workspace, pdbqt_a, "1.0,2.0,3.0", "20.0,20.0,20.0")

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
    workspace = Path("/workspace")
    pdbqt_a = Path("/workspace/receptor.pdbqt")
    args = _build_freedpp_args(cfg, workspace, pdbqt_a, "1.0,2.0,3.0", "20.0,20.0,20.0")
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

    import dualdocking.generator_freedpp as gen_mod

    monkeypatch.setattr(gen_mod, "DATA_DIR", tmp_path)

    result = _resolve_pdb_path("CDK2")
    assert result == pdb_path


def test_resolve_pdb_path_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Target не найден → FileNotFoundError."""
    import dualdocking.generator_freedpp as gen_mod

    monkeypatch.setattr(gen_mod, "DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="NONEXISTENT"):
        _resolve_pdb_path("NONEXISTENT")
