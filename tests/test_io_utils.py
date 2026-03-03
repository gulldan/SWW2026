"""Tests for dualdocking.data.io_utils — I/O utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dualdocking.data.io_utils import (
    check_inputs,
    read_df,
    read_molecules,
    stem_exists,
    write_df,
)

# ---------------------------------------------------------------------------
# read_df
# ---------------------------------------------------------------------------


def test_read_df_prefers_parquet(tmp_path: Path):
    """When both .parquet and .csv exist, parquet is preferred."""
    stem = tmp_path / "data"
    df_parquet = pd.DataFrame({"src": ["parquet"], "val": [1]})
    df_csv = pd.DataFrame({"src": ["csv"], "val": [2]})
    df_parquet.to_parquet(stem.with_suffix(".parquet"), index=False)
    df_csv.to_csv(stem.with_suffix(".csv"), index=False)

    result = read_df(stem)
    assert result["src"].iloc[0] == "parquet"


def test_read_df_falls_back_to_csv(tmp_path: Path):
    """When only .csv exists, it is used as fallback."""
    stem = tmp_path / "data"
    df = pd.DataFrame({"a": [10, 20]})
    df.to_csv(stem.with_suffix(".csv"), index=False)

    result = read_df(stem)
    assert list(result["a"]) == [10, 20]


def test_read_df_raises_when_neither_exists(tmp_path: Path):
    stem = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError, match="No data file found"):
        read_df(stem)


# ---------------------------------------------------------------------------
# write_df
# ---------------------------------------------------------------------------


def test_write_df_creates_parquet(tmp_path: Path):
    stem = tmp_path / "output"
    df = pd.DataFrame({"x": [1, 2, 3]})
    result_path = write_df(df, stem)

    assert result_path == stem.with_suffix(".parquet")
    assert result_path.exists()
    loaded = pd.read_parquet(result_path)
    assert list(loaded["x"]) == [1, 2, 3]


def test_write_df_csv_too(tmp_path: Path):
    """csv_too=True must also produce a .csv alongside the parquet."""
    stem = tmp_path / "output"
    df = pd.DataFrame({"x": [4, 5]})
    write_df(df, stem, csv_too=True)

    assert stem.with_suffix(".parquet").exists()
    assert stem.with_suffix(".csv").exists()
    loaded_csv = pd.read_csv(stem.with_suffix(".csv"))
    assert list(loaded_csv["x"]) == [4, 5]


def test_write_df_no_csv_by_default(tmp_path: Path):
    """By default, no .csv is created."""
    stem = tmp_path / "output"
    df = pd.DataFrame({"x": [1]})
    write_df(df, stem)

    assert stem.with_suffix(".parquet").exists()
    assert not stem.with_suffix(".csv").exists()


def test_write_df_creates_parent_dirs(tmp_path: Path):
    """write_df must create intermediate parent directories."""
    stem = tmp_path / "a" / "b" / "c" / "output"
    df = pd.DataFrame({"v": [99]})
    result_path = write_df(df, stem)

    assert result_path.exists()
    loaded = pd.read_parquet(result_path)
    assert list(loaded["v"]) == [99]


# ---------------------------------------------------------------------------
# read_molecules
# ---------------------------------------------------------------------------


def test_read_molecules_parquet(tmp_path: Path):
    path = tmp_path / "mols.parquet"
    df = pd.DataFrame({"smiles": ["CCO", "c1ccccc1", "CC(=O)O"]})
    df.to_parquet(path, index=False)

    result = read_molecules(path)
    assert result == ["CCO", "c1ccccc1", "CC(=O)O"]


def test_read_molecules_csv(tmp_path: Path):
    path = tmp_path / "mols.csv"
    df = pd.DataFrame({"smiles": ["CCO", "c1ccccc1"]})
    df.to_csv(path, index=False)

    result = read_molecules(path)
    assert result == ["CCO", "c1ccccc1"]


def test_read_molecules_smi(tmp_path: Path):
    path = tmp_path / "mols.smi"
    path.write_text("CCO ethanol\nc1ccccc1 benzene\nCC(=O)O acetic_acid\n")

    result = read_molecules(path)
    assert result == ["CCO", "c1ccccc1", "CC(=O)O"]


def test_read_molecules_smi_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "mols.smi"
    path.write_text("CCO\n\n\nc1ccccc1\n")

    result = read_molecules(path)
    assert result == ["CCO", "c1ccccc1"]


def test_read_molecules_missing_file(tmp_path: Path):
    path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="Molecule file not found"):
        read_molecules(path)


def test_read_molecules_unsupported_format(tmp_path: Path):
    path = tmp_path / "mols.xyz"
    path.write_text("some content")
    with pytest.raises(ValueError, match="Unsupported molecule format"):
        read_molecules(path)


def test_read_molecules_parquet_missing_smiles_column(tmp_path: Path):
    path = tmp_path / "mols.parquet"
    df = pd.DataFrame({"not_smiles": ["CCO"]})
    df.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="Column 'smiles' not found"):
        read_molecules(path)


def test_read_molecules_csv_missing_smiles_column(tmp_path: Path):
    path = tmp_path / "mols.csv"
    df = pd.DataFrame({"not_smiles": ["CCO"]})
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Column 'smiles' not found"):
        read_molecules(path)


# ---------------------------------------------------------------------------
# stem_exists
# ---------------------------------------------------------------------------


def test_stem_exists_parquet(tmp_path: Path):
    stem = tmp_path / "data"
    stem.with_suffix(".parquet").write_bytes(b"fake")
    assert stem_exists(stem) is True


def test_stem_exists_csv(tmp_path: Path):
    stem = tmp_path / "data"
    stem.with_suffix(".csv").write_text("a,b\n1,2\n")
    assert stem_exists(stem) is True


def test_stem_exists_neither(tmp_path: Path):
    stem = tmp_path / "data"
    assert stem_exists(stem) is False


# ---------------------------------------------------------------------------
# check_inputs
# ---------------------------------------------------------------------------


def test_check_inputs_passes_when_all_exist(tmp_path: Path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("content")
    f2.write_text("content")

    # Should return without error
    check_inputs([f1, f2], "test_stage")


def test_check_inputs_exits_on_missing_file(tmp_path: Path):
    existing = tmp_path / "exists.txt"
    existing.write_text("content")
    missing = tmp_path / "missing.txt"

    with pytest.raises(SystemExit) as exc_info:
        check_inputs([existing, missing], "test_stage")
    assert exc_info.value.code == 1


def test_check_inputs_exits_on_empty_file(tmp_path: Path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")

    with pytest.raises(SystemExit) as exc_info:
        check_inputs([empty], "test_stage")
    assert exc_info.value.code == 1


def test_check_inputs_with_hint(tmp_path: Path):
    """check_inputs must still exit when hint is provided."""
    missing = tmp_path / "missing.txt"

    with pytest.raises(SystemExit) as exc_info:
        check_inputs([missing], "test_stage", hint="run preprocessing first")
    assert exc_info.value.code == 1
