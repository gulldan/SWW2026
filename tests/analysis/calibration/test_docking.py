"""Tests for dualdocking.analysis.calibration.docking."""

from __future__ import annotations

import pytest

from dualdocking.analysis.calibration.docking import (
    auto_prep_workers,
    load_calibration_smiles,
    sample_smiles,
)


class TestLoadCalibrationSmiles:
    def test_loads_smiles_from_file(self, tmp_path):
        smi_file = tmp_path / "test.smi"
        smi_file.write_text("CCO ethanol\nCC methane\n\nCCCO butanol\n")
        result = load_calibration_smiles(smi_file)
        assert result == ["CCO", "CC", "CCCO"]

    def test_empty_file(self, tmp_path):
        smi_file = tmp_path / "empty.smi"
        smi_file.write_text("")
        result = load_calibration_smiles(smi_file)
        assert result == []

    def test_whitespace_only(self, tmp_path):
        smi_file = tmp_path / "ws.smi"
        smi_file.write_text("   \n\n   \n")
        result = load_calibration_smiles(smi_file)
        assert result == []

    def test_single_column(self, tmp_path):
        smi_file = tmp_path / "single.smi"
        smi_file.write_text("CCO\nCC\n")
        result = load_calibration_smiles(smi_file)
        assert result == ["CCO", "CC"]


class TestSampleSmiles:
    def test_returns_all_when_n_ge_len(self):
        smiles = ["A", "B", "C"]
        result = sample_smiles(smiles, 5, seed=42)
        assert result == ["A", "B", "C"]

    def test_subsamples_correctly(self):
        smiles = [f"C{'C' * i}" for i in range(100)]
        result = sample_smiles(smiles, 10, seed=42)
        assert len(result) == 10
        assert all(s in smiles for s in result)

    def test_deterministic(self):
        smiles = [f"C{'C' * i}" for i in range(100)]
        r1 = sample_smiles(smiles, 10, seed=42)
        r2 = sample_smiles(smiles, 10, seed=42)
        assert r1 == r2

    def test_different_seeds_differ(self):
        smiles = [f"C{'C' * i}" for i in range(100)]
        r1 = sample_smiles(smiles, 10, seed=42)
        r2 = sample_smiles(smiles, 10, seed=99)
        assert r1 != r2


class TestAutoPrepWorkers:
    def test_returns_positive_int(self):
        w = auto_prep_workers()
        assert isinstance(w, int)
        assert 1 <= w <= 8
