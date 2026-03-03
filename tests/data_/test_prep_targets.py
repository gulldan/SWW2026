"""Tests for dualdocking.data.prep.targets module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dualdocking.data.prep.targets import (
    PrepCheckpoint,
    TargetPrepConfig,
    prepare_pair,
    prepare_target,
)


def test_checkpoint_load_empty(tmp_path: Path) -> None:
    """PrepCheckpoint.load should return empty checkpoint when file missing."""
    cp = PrepCheckpoint.load(tmp_path / "nonexistent.json")
    assert cp.data == {}


def test_checkpoint_load_existing(tmp_path: Path) -> None:
    """PrepCheckpoint.load should load existing checkpoint data."""
    cp_path = tmp_path / "checkpoint.json"
    cp_path.write_text(json.dumps({"P1/A/pdb": True, "P1/B/pdb": True}))

    cp = PrepCheckpoint.load(cp_path)
    assert cp.is_done("P1", "A", "pdb") is True
    assert cp.is_done("P1", "B", "pdb") is True
    assert cp.is_done("P1", "A", "fasta") is False


def test_checkpoint_save_and_reload(tmp_path: Path) -> None:
    """PrepCheckpoint should persist and reload correctly."""
    cp_path = tmp_path / "checkpoint.json"
    cp = PrepCheckpoint.load(cp_path)

    cp.mark("P1", "A", "pdb")
    cp.mark("P1", "B", "fasta")

    cp2 = PrepCheckpoint.load(cp_path)
    assert cp2.is_done("P1", "A", "pdb") is True
    assert cp2.is_done("P1", "B", "fasta") is True


def test_prepare_target_pdb_only(tmp_path: Path) -> None:
    """prepare_target with full=False should only download PDB."""
    data_dir = tmp_path / "targets"
    cp = PrepCheckpoint.load(tmp_path / "checkpoint.json")

    config = TargetPrepConfig(
        target_id="CDK2",
        pdb_id="1HCL",
        uniprot_id="P24941",
        pair_id="CDK2_ERK2",
        role="A",
        data_dir=data_dir,
        msa_cache_dir=tmp_path / "msa",
        project_root=tmp_path,
    )

    fake_resp = MagicMock()
    fake_resp.text = "ATOM 1 N ALA"

    with patch("dualdocking.data.fetch.pdb.get_with_retry", return_value=fake_resp):
        prepare_target(config, cp, full=False)

    # PDB should exist
    pdb_path = data_dir / "CDK2_ERK2" / "A" / "1hcl.pdb"
    assert pdb_path.exists()
    assert "ATOM" in pdb_path.read_text()

    # Checkpoint should be marked
    assert cp.is_done("CDK2_ERK2", "A", "pdb") is True
    # FASTA should NOT be marked
    assert cp.is_done("CDK2_ERK2", "A", "fasta") is False


def test_prepare_target_skips_completed(tmp_path: Path) -> None:
    """prepare_target should skip steps already in checkpoint."""
    data_dir = tmp_path / "targets"
    cp = PrepCheckpoint.load(tmp_path / "checkpoint.json")

    # Pre-mark PDB as done
    cp.mark("CDK2_ERK2", "A", "pdb")

    # Also create the output dir so it doesn't fail
    out_dir = data_dir / "CDK2_ERK2" / "A"
    out_dir.mkdir(parents=True)

    config = TargetPrepConfig(
        target_id="CDK2",
        pdb_id="1HCL",
        uniprot_id="P24941",
        pair_id="CDK2_ERK2",
        role="A",
        data_dir=data_dir,
        msa_cache_dir=tmp_path / "msa",
        project_root=tmp_path,
    )

    # Should not call download_pdb since step is already done
    with patch("dualdocking.data.fetch.pdb.get_with_retry") as mock_get:
        prepare_target(config, cp, full=False)
        mock_get.assert_not_called()


def test_prepare_pair_both_targets(tmp_path: Path) -> None:
    """prepare_pair should prepare both targets A and B."""
    data_dir = tmp_path / "targets"
    cp = PrepCheckpoint.load(tmp_path / "checkpoint.json")

    pair_cfg = {
        "target_a": {"id": "CDK2", "pdb_id": "1HCL", "uniprot": "P24941"},
        "target_b": {"id": "ERK2", "pdb_id": "2OJG", "uniprot": "P28482"},
    }

    fake_resp = MagicMock()
    fake_resp.text = "ATOM"

    with patch("dualdocking.data.fetch.pdb.get_with_retry", return_value=fake_resp):
        prepare_pair(
            "CDK2_ERK2",
            pair_cfg,
            data_dir=data_dir,
            msa_cache_dir=tmp_path / "msa",
            project_root=tmp_path,
            checkpoint=cp,
            full=False,
        )

    assert cp.is_done("CDK2_ERK2", "A", "pdb") is True
    assert cp.is_done("CDK2_ERK2", "B", "pdb") is True
    assert (data_dir / "CDK2_ERK2" / "A" / "1hcl.pdb").exists()
    assert (data_dir / "CDK2_ERK2" / "B" / "2ojg.pdb").exists()
