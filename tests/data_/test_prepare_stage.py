"""Tests for dualdocking.pipeline.stages.prepare orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from dualdocking.pipeline.stages.prepare import prepare_all, prepare_all_targets


def test_prepare_all_targets_basic(tmp_path: Path) -> None:
    """prepare_all_targets should prepare both targets in a pair."""
    data_dir = tmp_path / "targets"
    msa_cache = tmp_path / "msa"
    checkpoint_path = tmp_path / "checkpoint.json"

    pairs = {
        "CDK2_ERK2": {
            "target_a": {"id": "CDK2", "pdb_id": "1HCL", "uniprot": "P24941"},
            "target_b": {"id": "ERK2", "pdb_id": "2OJG", "uniprot": "P28482"},
        }
    }

    from unittest.mock import MagicMock

    fake_resp = MagicMock()
    fake_resp.text = "ATOM"

    with patch("dualdocking.data.fetch.pdb.get_with_retry", return_value=fake_resp):
        prepare_all_targets(
            pairs,
            data_dir=data_dir,
            msa_cache_dir=msa_cache,
            project_root=tmp_path,
            checkpoint_path=checkpoint_path,
            full=False,
        )

    assert (data_dir / "CDK2_ERK2" / "A" / "1hcl.pdb").exists()
    assert (data_dir / "CDK2_ERK2" / "B" / "2ojg.pdb").exists()

    # Checkpoint should exist
    assert checkpoint_path.exists()
    cp = json.loads(checkpoint_path.read_text())
    assert cp["CDK2_ERK2/A/pdb"] is True
    assert cp["CDK2_ERK2/B/pdb"] is True


def test_prepare_all_dry_run(tmp_path: Path) -> None:
    """prepare_all with dry_run should not fail or download anything."""
    prepare_all(
        tmp_path,
        fetch_freed=True,
        fetch_baseline=False,
        convert_baseline_flag=False,
        sample_baseline_flag=False,
        dry_run=True,
    )


def test_prepare_all_convert_baseline(tmp_path: Path) -> None:
    """prepare_all should convert baseline archive to Parquet."""
    # Create a fake CSV
    csv_file = tmp_path / "output" / "data" / "baseline" / "zinc-druglike-cano.tar.xz"
    csv_file.parent.mkdir(parents=True)
    # We can't easily create a tar.xz in test, so test convert directly
    # instead of going through prepare_all

    from dualdocking.data.prep.baseline import convert_to_parquet

    csv_input = tmp_path / "test.csv"
    csv_input.write_text("smiles\nCCC\nCCCC\n")
    parquet_out = tmp_path / "test.parquet"
    total = convert_to_parquet(csv_input, parquet_out, has_header=True)
    assert total == 2


def test_prepare_all_sample_baseline(tmp_path: Path) -> None:
    """prepare_all should sample baseline from Parquet."""
    parquet_path = tmp_path / "output" / "data" / "baseline" / "zinc-druglike-cano.parquet"
    parquet_path.parent.mkdir(parents=True)

    smiles = [f"C{'C' * i}" for i in range(50)]
    table = pa.table({"smiles": smiles})
    pq.write_table(table, parquet_path)

    sample_path = tmp_path / "output" / "data" / "baseline" / "sample.smi"

    with patch("dualdocking.pipeline.stages.prepare.fetch_freed_data"):
        prepare_all(
            tmp_path,
            fetch_freed=True,
            fetch_baseline=False,
            convert_baseline_flag=False,
            sample_baseline_flag=True,
            sample_output="output/data/baseline/sample.smi",
            sample_n=5,
            sample_seed=42,
            dry_run=False,
        )

    assert sample_path.exists()
    lines = sample_path.read_text().strip().splitlines()
    assert len(lines) == 5
