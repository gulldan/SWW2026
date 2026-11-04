"""Baseline SMILES sampling from Parquet and archive conversion.

Extracted from ``cli/s1_prepare/sample_baseline.py`` and
``cli/s1_prepare/convert_baseline_parquet.py``.
"""

from __future__ import annotations

import bisect
import csv
import io
import random
import tarfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from dualdocking.data.http_utils import sha256_file


def sample_baseline(
    input_path: Path,
    output_path: Path,
    *,
    n: int = 1000,
    seed: int = 42,
    column: str = "smiles",
) -> Path:
    """Sample N SMILES from a Parquet dataset using index-based sampling.

    Args:
        input_path: Path to the input Parquet file.
        output_path: Path for the output .smi file.
        n: Number of SMILES to sample.
        seed: Random seed for reproducibility.
        column: Name of the Parquet column containing SMILES.

    Returns:
        Path to the written .smi file.

    Raises:
        RuntimeError: If the column is missing, *n* exceeds total rows,
            or too few SMILES are collected.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(input_path)
    if column not in pf.schema.names:
        raise RuntimeError(f"Column '{column}' not found in {input_path}. Columns: {pf.schema.names}")

    rng = random.Random(seed)  # noqa: S311

    total_rows = pf.metadata.num_rows
    if n > total_rows:
        raise RuntimeError(f"Requested n={n} > total_rows={total_rows}")

    indices = rng.sample(range(total_rows), n)
    indices_sorted = sorted(indices)

    # Build cumulative row-group boundaries.
    ends: list[int] = []
    running = 0
    for rg in range(pf.metadata.num_row_groups):
        running += pf.metadata.row_group(rg).num_rows
        ends.append(running)

    # Map indices to row groups.
    group_map: dict[int, list[int]] = {}
    for idx in indices_sorted:
        rg = bisect.bisect_left(ends, idx + 1)
        start = 0 if rg == 0 else ends[rg - 1]
        local = idx - start
        group_map.setdefault(rg, []).append(local)

    # Fetch required rows.
    values: dict[int, str] = {}
    for rg, locals_idx in group_map.items():
        batch = pf.read_row_group(rg, columns=[column])[column].to_pylist()
        for local in locals_idx:
            smi = batch[local]
            if not smi:
                continue
            values[ends[rg - 1] + local if rg > 0 else local] = str(smi)

    if len(values) < n:
        raise RuntimeError(f"Only collected {len(values)} SMILES from {input_path}")

    sample = [values[idx] for idx in indices if idx in values]
    if len(sample) < n:
        raise RuntimeError(f"Only collected {len(sample)} SMILES after filtering empties")

    output_path.write_text("\n".join(sample) + "\n")
    sha = sha256_file(output_path)
    sha_path = output_path.with_suffix(output_path.suffix + ".sha256")
    sha_path.write_text(sha + "\n")

    logger.info("Saved {} SMILES -> {}", len(sample), output_path)
    logger.info("SHA256 -> {} ({})", sha_path, sha)
    return output_path


@contextmanager
def _open_input(path: Path, member: str | None = None) -> Iterator[io.BufferedReader]:
    """Open a plain file or a member inside a tar archive for binary reading.

    Args:
        path: Path to a plain file or tar archive.
        member: Optional member name inside the tar archive.

    Yields:
        A binary file-like object for the selected content.

    Raises:
        RuntimeError: If the tar archive contains no files or extraction fails.
    """
    if path.suffixes[-2:] in ([".tar", ".xz"], [".tar", ".gz"], [".tar", ".bz2"]) or path.suffix == ".tar":
        mode = "r:*"
        with tarfile.open(path, mode=mode) as tf:
            if member is None:
                selected = None
                for item in tf:
                    if item.isfile():
                        selected = item
                        break
                if selected is None:
                    raise RuntimeError(f"No files found in tar archive: {path}")
            else:
                selected = tf.getmember(member)
            fileobj = tf.extractfile(selected)
            if fileobj is None:
                raise RuntimeError(f"Failed to extract {selected.name} from {path}")
            yield fileobj
        return

    with path.open("rb") as f:
        yield f


def _detect_header(path: Path, member: str | None, delimiter: str) -> bool:
    """Heuristically detect whether the input has a CSV header row.

    Args:
        path: Path to the input file (or tar archive).
        member: Optional tar member name.
        delimiter: CSV column delimiter.

    Returns:
        True if the first row appears to be a header.
    """
    with _open_input(path, member) as f:
        first = f.readline().decode("utf-8", errors="replace").strip()
    if not first:
        return False
    lowered = first.lower()
    if "smiles" in lowered.split(delimiter):
        return True
    return lowered == "smiles"


def convert_to_parquet(
    input_path: Path,
    output_path: Path,
    *,
    member: str | None = None,
    column: str | None = None,
    delimiter: str = ",",
    has_header: bool | None = None,
    row_group_size: int = 100_000,
    max_rows: int | None = None,
) -> int:
    """Stream a CSV (or tarred CSV) into a compressed Parquet file.

    Args:
        input_path: Path to the input CSV or tar archive.
        output_path: Path for the output Parquet file.
        member: Member name inside the tar archive (auto-detected if omitted).
        column: Column name or zero-based index to retain (all if omitted).
        delimiter: CSV field delimiter character.
        has_header: Force header presence (auto-detected if omitted).
        row_group_size: Number of rows per Parquet row group.
        max_rows: Stop after writing this many rows (for debugging).

    Returns:
        Total number of rows written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if has_header is None:
        has_header = _detect_header(input_path, member, delimiter)

    logger.info("Input: {}", input_path)
    logger.info("Header: {}", has_header)
    logger.info("Output: {}", output_path)

    total_rows = 0
    writer: pq.ParquetWriter | None = None
    try:
        with _open_input(input_path, member) as f:
            text = io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline="")
            reader = csv.reader(text, delimiter=delimiter)

            header = next(reader, None)
            if header is None:
                raise RuntimeError("Empty input file")
            if not has_header:
                first_row = header
                header = [f"col{i}" for i in range(len(first_row))]
            else:
                first_row = None

            if column is not None:
                if column.isdigit():
                    col_idx = int(column)
                    if col_idx < 0 or col_idx >= len(header):
                        raise RuntimeError(f"Column index {col_idx} out of range (0..{len(header) - 1})")
                    keep_indices = [col_idx]
                    keep_names = [header[col_idx]]
                else:
                    if column not in header:
                        raise RuntimeError(f"Column '{column}' not found. Available: {header}")
                    keep_indices = [header.index(column)]
                    keep_names = [column]
            else:
                keep_indices = list(range(len(header)))
                keep_names = header

            def _emit_batch(rows: list[list[str]]) -> None:
                nonlocal writer, total_rows
                columns_data: dict[str, list[str]] = {name: [] for name in keep_names}
                for row in rows:
                    for name, idx in zip(keep_names, keep_indices):
                        columns_data[name].append(row[idx] if idx < len(row) else "")
                table = pa.table(columns_data)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table, row_group_size=row_group_size)
                total_rows += len(rows)

            batch: list[list[str]] = []
            if first_row is not None:
                batch.append(first_row)

            for row in reader:
                if not row:
                    continue
                batch.append(row)
                if len(batch) >= row_group_size:
                    _emit_batch(batch)
                    batch = []
                if max_rows is not None and total_rows + len(batch) >= max_rows:
                    _emit_batch(batch)
                    logger.warning("Stopping early at max_rows={}", max_rows)
                    batch = []
                    break

            if batch:
                _emit_batch(batch)
    finally:
        if writer is not None:
            writer.close()

    logger.info("Wrote {} rows -> {}", total_rows, output_path)
    return total_rows
