"""Sample a baseline SMILES set from a large Parquet file (streaming).

Uses random index sampling (not reservoir) for O(1) memory and writes
the result as a ``.smi`` file with an accompanying SHA-256 sidecar.
"""

from __future__ import annotations

import bisect
import hashlib
import random
from pathlib import Path

import click
import pyarrow.parquet as pq
from loguru import logger


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@click.command()
@click.option("--input", "input_path", default="output/data/baseline/zinc-druglike-cano.parquet")
@click.option("--output", "output_path", default="output/data/baseline/zinc_druglike_baseline_1000_seed42.smi")
@click.option("--n", default=1000, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--column", default="smiles")
def main(input_path: str, output_path: str, n: int, seed: int, column: str) -> None:
    """Reservoir sample N SMILES from a Parquet dataset.

    Selects *n* random SMILES strings from the Parquet file using
    index-based sampling, writes them as a newline-delimited ``.smi``
    file, and stores a SHA-256 sidecar for integrity verification.

    Args:
        input_path: Path to the input Parquet file.
        output_path: Path for the output .smi file.
        n: Number of SMILES to sample.
        seed: Random seed for reproducibility.
        column: Name of the Parquet column containing SMILES.

    Raises:
        RuntimeError: If the column is missing, *n* exceeds total rows,
            or too few SMILES are collected.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(in_path)
    if column not in pf.schema.names:
        raise RuntimeError(f"Column '{column}' not found in {in_path}. Columns: {pf.schema.names}")

    rng = random.Random(seed)  # noqa: S311

    total_rows = pf.metadata.num_rows
    if n > total_rows:
        raise RuntimeError(f"Requested n={n} > total_rows={total_rows}")

    # Pick random global row indices (uniform) without scanning entire dataset.
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
        raise RuntimeError(f"Only collected {len(values)} SMILES from {in_path}")

    # Preserve deterministic order of the RNG sample.
    sample = [values[idx] for idx in indices if idx in values]
    if len(sample) < n:
        raise RuntimeError(f"Only collected {len(sample)} SMILES after filtering empties")

    out_path.write_text("\n".join(sample) + "\n")
    sha = _sha256(out_path)
    sha_path = out_path.with_suffix(out_path.suffix + ".sha256")
    sha_path.write_text(sha + "\n")

    logger.info("Saved {} SMILES -> {}", len(sample), out_path)
    logger.info("SHA256 -> {} ({})", sha_path, sha)


if __name__ == "__main__":
    main()
