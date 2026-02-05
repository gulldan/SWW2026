"""Convert baseline dataset to Parquet in a streaming, low-memory way."""

from __future__ import annotations

import csv
import io
import tarfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import click
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger


@contextmanager
def _open_input(path: Path, member: str | None = None) -> Iterator[io.BufferedReader]:
    if path.suffixes[-2:] in ([".tar", ".xz"], [".tar", ".gz"], [".tar", ".bz2"]) or path.suffix == ".tar":
        mode = "r:*"
        tf = tarfile.open(path, mode=mode)
        try:
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
            yield fileobj  # type: ignore[misc]
        finally:
            tf.close()
        return

    with path.open("rb") as f:
        yield f


def _detect_header(path: Path, member: str | None, delimiter: str) -> bool:
    with _open_input(path, member) as f:
        first = f.readline().decode("utf-8", errors="replace").strip()
    if not first:
        return False
    lowered = first.lower()
    if "smiles" in lowered.split(delimiter):
        return True
    if lowered == "smiles":
        return True
    return False


@click.command()
@click.option("--input", "input_path", default="data/baseline/zinc-druglike-cano.tar.xz")
@click.option("--output", "output_path", default="data/baseline/zinc-druglike-cano.parquet")
@click.option("--member", default=None, help="Member file inside tar archive (optional)")
@click.option("--column", default=None, help="Only keep this column (name or index)")
@click.option("--delimiter", default=",", help="CSV delimiter")
@click.option("--has-header/--no-header", default=None, help="Force header detection")
@click.option("--row-group-size", default=100_000, help="Parquet row group size")
@click.option("--max-rows", default=None, type=int, help="Stop after N rows (debug)")
def main(
    input_path: str,
    output_path: str,
    member: str | None,
    column: str | None,
    delimiter: str,
    has_header: bool | None,
    row_group_size: int,
    max_rows: int | None,
) -> None:
    """Stream CSV (or tarred CSV) into Parquet."""
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if has_header is None:
        has_header = _detect_header(in_path, member, delimiter)

    logger.info("Input: {}", in_path)
    if member:
        logger.info("Member: {}", member)
    logger.info("Header: {}", has_header)
    logger.info("Output: {}", out_path)

    total_rows = 0
    writer: pq.ParquetWriter | None = None
    try:
        with _open_input(in_path, member) as f:
            text = io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline="")
            reader = csv.reader(text, delimiter=delimiter)

            header = next(reader, None)
            if header is None:
                raise RuntimeError("Empty input file")
            if not has_header:
                # Treat the first row as data.
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
                columns: dict[str, list[str]] = {name: [] for name in keep_names}
                for row in rows:
                    for name, idx in zip(keep_names, keep_indices):
                        columns[name].append(row[idx] if idx < len(row) else "")
                table = pa.table(columns)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
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

    logger.info("Wrote {} rows -> {}", total_rows, out_path)


if __name__ == "__main__":
    main()
