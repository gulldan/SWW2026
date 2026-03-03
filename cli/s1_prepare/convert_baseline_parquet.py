"""Convert baseline dataset to Parquet in a streaming, low-memory way.

Thin CLI wrapper -- core logic lives in :mod:`dualdocking.data.prep.baseline`.
"""

from __future__ import annotations

from pathlib import Path

import click

from dualdocking.data.prep.baseline import convert_to_parquet


@click.command()
@click.option("--input", "input_path", default="output/data/baseline/zinc-druglike-cano.tar.xz")
@click.option("--output", "output_path", default="output/data/baseline/zinc-druglike-cano.parquet")
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
    """Stream a CSV (or tarred CSV) into a compressed Parquet file."""
    convert_to_parquet(
        Path(input_path),
        Path(output_path),
        member=member,
        column=column,
        delimiter=delimiter,
        has_header=has_header,
        row_group_size=row_group_size,
        max_rows=max_rows,
    )


if __name__ == "__main__":
    main()
