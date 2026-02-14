"""Tests for dualdocking.data.prep.baseline module."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dualdocking.data.prep.baseline import (
    convert_to_parquet,
    sample_baseline,
)


def _write_test_parquet(path: Path, smiles: list[str]) -> None:
    """Write a test Parquet file with a smiles column."""
    table = pa.table({"smiles": smiles})
    pq.write_table(table, path)


def test_sample_baseline_basic(tmp_path: Path) -> None:
    """sample_baseline should produce a .smi file with the correct count."""
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.smi"

    smiles = [f"CCC{'C' * i}" for i in range(100)]
    _write_test_parquet(input_path, smiles)

    result = sample_baseline(input_path, output_path, n=10, seed=42, column="smiles")

    assert result == output_path
    assert output_path.exists()
    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 10

    # SHA sidecar should exist
    sha_path = output_path.with_suffix(".smi.sha256")
    assert sha_path.exists()
    sha = sha_path.read_text().strip()
    assert len(sha) == 64


def test_sample_baseline_deterministic(tmp_path: Path) -> None:
    """sample_baseline should be deterministic given the same seed."""
    input_path = tmp_path / "input.parquet"
    smiles = [f"CCC{'C' * i}" for i in range(50)]
    _write_test_parquet(input_path, smiles)

    out1 = tmp_path / "out1.smi"
    out2 = tmp_path / "out2.smi"

    sample_baseline(input_path, out1, n=5, seed=42)
    sample_baseline(input_path, out2, n=5, seed=42)

    assert out1.read_text() == out2.read_text()


def test_sample_baseline_n_exceeds_total(tmp_path: Path) -> None:
    """sample_baseline should raise when n > total rows."""
    input_path = tmp_path / "input.parquet"
    _write_test_parquet(input_path, ["CCC", "CCCC"])

    with pytest.raises(RuntimeError, match="Requested n=10 > total_rows=2"):
        sample_baseline(input_path, tmp_path / "out.smi", n=10)


def test_sample_baseline_missing_column(tmp_path: Path) -> None:
    """sample_baseline should raise for missing column."""
    input_path = tmp_path / "input.parquet"
    table = pa.table({"other_col": ["a", "b"]})
    pq.write_table(table, input_path)

    with pytest.raises(RuntimeError, match="Column 'smiles' not found"):
        sample_baseline(input_path, tmp_path / "out.smi", n=1)


def test_convert_to_parquet_basic(tmp_path: Path) -> None:
    """convert_to_parquet should produce a valid Parquet file."""
    csv_file = tmp_path / "input.csv"
    csv_file.write_text("smiles\nCCC\nCCCC\nCCCCC\n")

    output = tmp_path / "output.parquet"
    total = convert_to_parquet(csv_file, output, has_header=True)

    assert total == 3
    assert output.exists()

    df = pq.read_table(output).to_pydict()
    assert df["smiles"] == ["CCC", "CCCC", "CCCCC"]


def test_convert_to_parquet_single_column(tmp_path: Path) -> None:
    """convert_to_parquet should extract a single column."""
    csv_file = tmp_path / "input.csv"
    csv_file.write_text("smiles,other\nCCC,1\nCCCC,2\n")

    output = tmp_path / "output.parquet"
    convert_to_parquet(csv_file, output, column="smiles", has_header=True)

    df = pq.read_table(output).to_pydict()
    assert list(df.keys()) == ["smiles"]


def test_convert_to_parquet_no_header(tmp_path: Path) -> None:
    """convert_to_parquet should handle files without header."""
    csv_file = tmp_path / "input.csv"
    csv_file.write_text("CCC\nCCCC\n")

    output = tmp_path / "output.parquet"
    convert_to_parquet(csv_file, output, has_header=False)

    df = pq.read_table(output).to_pydict()
    # Without header, should use col0
    assert "col0" in df
    assert len(df["col0"]) == 2


def test_convert_to_parquet_max_rows(tmp_path: Path) -> None:
    """convert_to_parquet should stop at max_rows."""
    csv_file = tmp_path / "input.csv"
    csv_file.write_text("smiles\n" + "\n".join(f"C{'C' * i}" for i in range(100)) + "\n")

    output = tmp_path / "output.parquet"
    total = convert_to_parquet(csv_file, output, has_header=True, max_rows=5)

    assert total == 5
