"""Sample a baseline SMILES set from a large Parquet file (streaming).

Thin CLI wrapper -- core logic lives in :mod:`dualdocking.data.prep.baseline`.
"""

from __future__ import annotations

from pathlib import Path

import click

from dualdocking.data.prep.baseline import sample_baseline


@click.command()
@click.option("--input", "input_path", default="output/data/baseline/zinc-druglike-cano.parquet")
@click.option("--output", "output_path", default="output/data/baseline/zinc_druglike_baseline_1000_seed42.smi")
@click.option("--n", default=1000, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--column", default="smiles")
def main(input_path: str, output_path: str, n: int, seed: int, column: str) -> None:
    """Reservoir sample N SMILES from a Parquet dataset."""
    sample_baseline(
        Path(input_path),
        Path(output_path),
        n=n,
        seed=seed,
        column=column,
    )


if __name__ == "__main__":
    main()
