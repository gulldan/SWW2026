"""I/O utilities for the DualDocking pipeline.

Provides unified read/write for tabular data (Parquet with CSV fallback),
molecule loading from multiple formats, and preflight input validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from loguru import logger


def read_df(stem: Path) -> pd.DataFrame:
    """Read a DataFrame from ``{stem}.parquet`` or ``{stem}.csv``.

    Tries Parquet first, then falls back to CSV for backward compatibility.

    Args:
        stem: Path without extension (e.g. ``results/scored_joined``).

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If neither ``.parquet`` nor ``.csv`` exists.
    """
    parquet_path = stem.with_suffix(".parquet")
    csv_path = stem.with_suffix(".csv")

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(f"No data file found: tried {parquet_path} and {csv_path}")


def write_df(df: pd.DataFrame, stem: Path, *, csv_too: bool = False) -> Path:
    """Write a DataFrame to ``{stem}.parquet``, optionally also to CSV.

    Args:
        df: DataFrame to write.
        stem: Path without extension.
        csv_too: If ``True``, also write ``{stem}.csv`` alongside Parquet.

    Returns:
        Path to the written Parquet file.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = stem.with_suffix(".parquet")
    df.to_parquet(parquet_path, index=False)

    if csv_too:
        csv_path = stem.with_suffix(".csv")
        df.to_csv(csv_path, index=False)

    return parquet_path


def read_molecules(path: Path) -> list[str]:
    """Read SMILES strings from a file.

    Supported formats:

    - ``.parquet`` — expects a ``smiles`` column.
    - ``.csv`` — expects a ``smiles`` column.
    - ``.smi`` — one SMILES per line (first whitespace-delimited token).

    Args:
        path: Path to the molecule file.

    Returns:
        List of SMILES strings.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the format is unsupported or the ``smiles`` column
            is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Molecule file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(path)
        if "smiles" not in df.columns:
            raise ValueError(f"Column 'smiles' not found in {path}")
        return df["smiles"].tolist()

    if suffix == ".csv":
        df = pd.read_csv(path)
        if "smiles" not in df.columns:
            raise ValueError(f"Column 'smiles' not found in {path}")
        return df["smiles"].tolist()

    if suffix == ".smi":
        lines = path.read_text().strip().splitlines()
        smiles: list[str] = [raw.strip().split()[0] for raw in lines if raw.strip()]
        return smiles

    raise ValueError(f"Unsupported molecule format: {suffix} (expected .parquet, .csv, or .smi)")


def stem_exists(stem: Path) -> bool:
    """Check whether ``{stem}.parquet`` or ``{stem}.csv`` exists.

    Args:
        stem: Path without extension.

    Returns:
        ``True`` if at least one format exists.
    """
    return stem.with_suffix(".parquet").exists() or stem.with_suffix(".csv").exists()


def check_inputs(
    required: list[Path],
    label: str,
    *,
    hint: str | None = None,
) -> None:
    """Verify that all required input files exist and are non-empty.

    Exits with code 1 and a clear error message if any file is missing.

    Args:
        required: Paths to check.
        label: Human-readable stage name for the error message.
        hint: Optional hint about which command to run to produce the
            missing inputs.
    """
    missing: list[Path] = [p for p in required if not p.exists() or p.stat().st_size == 0]

    if not missing:
        return

    for p in missing:
        logger.error("{} requires {} (missing or empty)", label, p)
    if hint:
        logger.error("Hint: {}", hint)
    sys.exit(1)
