"""Tests for dualdocking.data.fetch.smiles module."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from dualdocking.data.fetch.smiles import (
    fetch_calibration_smiles,
    reservoir_sample,
    sha256_of_file,
)


def test_reservoir_sample_basic() -> None:
    """reservoir_sample returns up to k items."""
    rng = random.Random(42)
    items = [f"smi{i}" for i in range(100)]
    result = reservoir_sample(items, 10, rng)
    assert len(result) == 10
    assert all(isinstance(s, str) for s in result)


def test_reservoir_sample_fewer_than_k() -> None:
    """reservoir_sample returns all items when fewer than k."""
    rng = random.Random(42)
    items = ["a", "b", "c"]
    result = reservoir_sample(items, 10, rng)
    assert len(result) == 3


def test_reservoir_sample_skips_empty() -> None:
    """reservoir_sample should skip empty items."""
    rng = random.Random(42)
    items = ["a", "", "b", "", "c"]
    result = reservoir_sample(items, 10, rng)
    assert "" not in result
    assert len(result) == 3


def test_sha256_of_file(tmp_path: Path) -> None:
    """sha256_of_file computes correct hash."""
    f = tmp_path / "test.txt"
    f.write_text("hello")
    sha = sha256_of_file(f)
    assert len(sha) == 64
    assert sha == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_fetch_calibration_smiles_url_source(tmp_path: Path) -> None:
    """fetch_calibration_smiles should handle URL sources with mock HTTP."""
    from unittest.mock import MagicMock, patch

    output_dir = tmp_path / "calibration"
    output_path = tmp_path / "selected.smi"

    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.iter_lines = MagicMock(return_value=["CCC", "CCCC", "CCCCC"])
    fake_resp.__enter__ = MagicMock(return_value=fake_resp)
    fake_resp.__exit__ = MagicMock(return_value=False)

    sources = {
        "test_source": {
            "type": "url",
            "url": "https://example.com/smiles.txt",
            "smiles_column": 0,
        }
    }

    # requests is imported inside load_smiles_from_url, patch at requests module level
    with patch("requests.get", return_value=fake_resp):
        manifest = fetch_calibration_smiles(
            sources,
            "test_source",
            output_dir=output_dir,
            output_path=output_path,
            sample_n=10,
            seed=42,
            download_all=True,
        )

    assert "test_source" in manifest
    assert output_path.exists()
    smiles = output_path.read_text().strip().splitlines()
    assert len(smiles) == 3


def test_fetch_calibration_smiles_raises_on_missing_selected(tmp_path: Path) -> None:
    """fetch_calibration_smiles should raise if selected not in sources."""
    with pytest.raises(KeyError, match="not_here"):
        fetch_calibration_smiles(
            {"source_a": {"type": "url", "url": "x"}},
            "not_here",
            output_dir=tmp_path,
            output_path=tmp_path / "out.smi",
        )


def test_fetch_calibration_smiles_skips_existing(tmp_path: Path) -> None:
    """fetch_calibration_smiles should skip existing files when force=False."""
    output_dir = tmp_path / "calibration"
    output_dir.mkdir()
    output_path = tmp_path / "selected.smi"

    # Pre-create the file
    existing_file = output_dir / "test_source_n10_seed42.smi"
    existing_file.write_text("CCC\n")

    sources = {
        "test_source": {
            "type": "url",
            "url": "https://example.com/smiles.txt",
            "smiles_column": 0,
        }
    }

    manifest = fetch_calibration_smiles(
        sources,
        "test_source",
        output_dir=output_dir,
        output_path=output_path,
        sample_n=10,
        seed=42,
        force=False,
    )

    assert "test_source" in manifest
    assert "sha256" in manifest["test_source"]
