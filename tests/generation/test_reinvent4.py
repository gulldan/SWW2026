"""Tests for REINVENT4 backend adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

import dualdocking.generation.reinvent4.core as r4_mod
from dualdocking.generation.reinvent4.core import (
    CACHE_KEY_VERSION,
    Reinvent4GenerationConfig,
    _cache_key,
    _load_normalization,
    _parse_output_smiles,
)


# --- _cache_key ---


class TestCacheKey:
    def test_deterministic(self) -> None:
        """Same config produces the same hash."""
        cfg1 = Reinvent4GenerationConfig(target_id="CDK2", seed=42, n_steps=500)
        cfg2 = Reinvent4GenerationConfig(target_id="CDK2", seed=42, n_steps=500)
        assert _cache_key(cfg1) == _cache_key(cfg2)

    def test_varies_by_target(self) -> None:
        """Different targets produce different hashes."""
        cfg1 = Reinvent4GenerationConfig(target_id="CDK2", seed=42)
        cfg2 = Reinvent4GenerationConfig(target_id="ERK2", seed=42)
        assert _cache_key(cfg1) != _cache_key(cfg2)

    def test_varies_by_mode(self) -> None:
        """Different reward modes produce different hashes."""
        cfg1 = Reinvent4GenerationConfig(target_id="CDK2", reward_mode="single_a")
        cfg2 = Reinvent4GenerationConfig(target_id="CDK2", reward_mode="double", second_target_id="ERK2")
        assert _cache_key(cfg1) != _cache_key(cfg2)

    def test_varies_by_seed(self) -> None:
        """Different seeds produce different hashes."""
        cfg1 = Reinvent4GenerationConfig(target_id="CDK2", seed=42)
        cfg2 = Reinvent4GenerationConfig(target_id="CDK2", seed=137)
        assert _cache_key(cfg1) != _cache_key(cfg2)

    def test_varies_by_algorithm(self) -> None:
        """Different algorithms produce different hashes."""
        cfg1 = Reinvent4GenerationConfig(target_id="CDK2", algorithm="dap")
        cfg2 = Reinvent4GenerationConfig(target_id="CDK2", algorithm="reinforce")
        assert _cache_key(cfg1) != _cache_key(cfg2)

    def test_key_length(self) -> None:
        """Cache key is 16 hex characters."""
        cfg = Reinvent4GenerationConfig(target_id="CDK2")
        key = _cache_key(cfg)
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)


# --- _load_normalization ---


class TestLoadNormalization:
    def test_loads_target(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Load normalization stats for a valid target."""
        norm_yaml = tmp_path / "normalization.yaml"
        norm_yaml.write_text(yaml.dump({
            "targets": {
                "CDK2": {"mean": -7.42, "std": 0.705},
            }
        }))
        monkeypatch.setattr(r4_mod, "PROJECT_ROOT", tmp_path)
        (tmp_path / "configs").mkdir()
        import shutil

        shutil.copy2(norm_yaml, tmp_path / "configs" / "normalization.yaml")

        mean, std = _load_normalization("CDK2")
        assert mean == pytest.approx(-7.42)
        assert std == pytest.approx(0.705)

    def test_missing_target_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """KeyError when target not in normalization.yaml."""
        norm_yaml = tmp_path / "configs" / "normalization.yaml"
        norm_yaml.parent.mkdir(parents=True)
        norm_yaml.write_text(yaml.dump({"targets": {"CDK2": {"mean": -7.0, "std": 1.0}}}))
        monkeypatch.setattr(r4_mod, "PROJECT_ROOT", tmp_path)

        with pytest.raises(KeyError, match="MISSING"):
            _load_normalization("MISSING")

    def test_missing_file_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """FileNotFoundError when config file doesn't exist."""
        monkeypatch.setattr(r4_mod, "PROJECT_ROOT", tmp_path)
        with pytest.raises(FileNotFoundError):
            _load_normalization("CDK2")


# --- _parse_output_smiles ---


class TestParseOutputSmiles:
    def test_scaffold_memory_csv(self, tmp_path: Path) -> None:
        """Parse SMILES from scaffold_memory.csv."""
        csv_path = tmp_path / "scaffold_memory.csv"
        pd.DataFrame({"SMILES": ["CCO", "CCC", "CCCC"]}).to_csv(csv_path, index=False)

        smiles = _parse_output_smiles(tmp_path, n_molecules=10)
        assert smiles == ["CCO", "CCC", "CCCC"]

    def test_deduplication(self, tmp_path: Path) -> None:
        """Duplicate SMILES are removed."""
        csv_path = tmp_path / "scaffold_memory.csv"
        pd.DataFrame({"SMILES": ["CCO", "CCO", "CCC"]}).to_csv(csv_path, index=False)

        smiles = _parse_output_smiles(tmp_path, n_molecules=10)
        assert smiles == ["CCO", "CCC"]

    def test_truncation(self, tmp_path: Path) -> None:
        """Output truncated to n_molecules."""
        csv_path = tmp_path / "scaffold_memory.csv"
        pd.DataFrame({"SMILES": [f"C{'C' * i}" for i in range(20)]}).to_csv(csv_path, index=False)

        smiles = _parse_output_smiles(tmp_path, n_molecules=5)
        assert len(smiles) == 5

    def test_fallback_csv(self, tmp_path: Path) -> None:
        """Falls back to other CSV files when scaffold_memory is absent."""
        csv_path = tmp_path / "results.csv"
        pd.DataFrame({"smiles": ["CCO"]}).to_csv(csv_path, index=False)

        smiles = _parse_output_smiles(tmp_path, n_molecules=10)
        assert smiles == ["CCO"]

    def test_empty_workspace(self, tmp_path: Path) -> None:
        """Empty workspace returns empty list."""
        smiles = _parse_output_smiles(tmp_path, n_molecules=10)
        assert smiles == []

    def test_nested_scaffold_memory(self, tmp_path: Path) -> None:
        """Finds scaffold_memory in subdirectory."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        csv_path = sub / "scaffold_memory.csv"
        pd.DataFrame({"SMILES": ["CCO"]}).to_csv(csv_path, index=False)

        smiles = _parse_output_smiles(tmp_path, n_molecules=10)
        assert smiles == ["CCO"]


# --- generate (integration-level, mocked) ---


class TestGenerate:
    def test_requires_second_target_for_double_mode(self) -> None:
        """double mode without second_target_id raises ValueError."""
        from dualdocking.generation.reinvent4.core import generate

        cfg = Reinvent4GenerationConfig(
            target_id="CDK2",
            reward_mode="double",
            second_target_id=None,
        )
        with pytest.raises(ValueError, match="requires second_target_id"):
            generate(cfg)
