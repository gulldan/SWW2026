"""One-shot preparation of local data dependencies for experiments.

Orchestrates fetching FREED docker data, downloading the baseline
archive, converting it to Parquet, and optionally sampling a baseline
SMILES file.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
from loguru import logger

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def _run(cmd: list[str], dry_run: bool) -> None:
    """Run a subprocess command, raising on failure.

    Args:
        cmd: Command and arguments to execute.
        dry_run: If True, only log the command without executing.

    Raises:
        click.ClickException: If the command exits with a non-zero code.
    """
    logger.info("RUN {}", " ".join(cmd))
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode != 0:
        tail = ((result.stdout or "") + (result.stderr or ""))[-4000:]
        raise click.ClickException(f"Command failed (rc={result.returncode}): {' '.join(cmd)}\n{tail}")


@click.command()
@click.option("--dry-run", is_flag=True, default=False, help="Print commands without executing")
@click.option(
    "--fetch-freed-data/--skip-fetch-freed-data",
    default=True,
    show_default=True,
    help="Ensure docker/freedpp/data/{zinc_crem.json,alert_collections.csv}.",
)
@click.option("--freed-force", is_flag=True, default=False, help="Overwrite existing FREED data files.")
@click.option(
    "--fetch-baseline/--skip-fetch-baseline",
    default=False,
    show_default=True,
    help="Download baseline archive data/baseline/zinc-druglike-cano.tar.xz.",
)
@click.option("--baseline-url", default=None, help="URL of baseline archive (or ZINC_BASELINE_URL env).")
@click.option("--baseline-output", default="output/data/baseline/zinc-druglike-cano.tar.xz", show_default=True)
@click.option("--baseline-overwrite", is_flag=True, default=False)
@click.option(
    "--convert-baseline/--skip-convert-baseline",
    default=False,
    show_default=True,
    help="Convert baseline tar.xz/CSV to Parquet.",
)
@click.option("--baseline-member", default=None, help="Member file inside tar archive (optional).")
@click.option("--baseline-column", default="smiles", show_default=True, help="Column name or index for conversion.")
@click.option("--parquet-output", default="output/data/baseline/zinc-druglike-cano.parquet", show_default=True)
@click.option("--parquet-overwrite", is_flag=True, default=False)
@click.option(
    "--sample-baseline/--skip-sample-baseline",
    default=False,
    show_default=True,
    help="Sample baseline .smi from baseline Parquet.",
)
@click.option("--sample-output", default="output/data/baseline/zinc_druglike_baseline_1000_seed42.smi", show_default=True)
@click.option("--sample-n", default=1000, type=int, show_default=True)
@click.option("--sample-seed", default=42, type=int, show_default=True)
@click.option("--sample-overwrite", is_flag=True, default=False)
def main(  # noqa: PLR0913
    dry_run: bool,
    fetch_freed_data: bool,
    freed_force: bool,
    fetch_baseline: bool,
    baseline_url: str | None,
    baseline_output: str,
    baseline_overwrite: bool,
    convert_baseline: bool,
    baseline_member: str | None,
    baseline_column: str,
    parquet_output: str,
    parquet_overwrite: bool,
    sample_baseline: bool,
    sample_output: str,
    sample_n: int,
    sample_seed: int,
    sample_overwrite: bool,
) -> None:
    """Prepare FREED docker data, baseline archive, Parquet and optional sample.

    Runs a sequence of sub-scripts to ensure all local data dependencies
    are in place before experiment execution.

    Args:
        dry_run: Print commands without executing them.
        fetch_freed_data: Download FREED fragment library and alerts.
        freed_force: Overwrite existing FREED data files.
        fetch_baseline: Download the baseline ZINC archive.
        baseline_url: Override URL for baseline download.
        baseline_output: Local path for the baseline archive.
        baseline_overwrite: Overwrite existing baseline archive.
        convert_baseline: Convert the baseline archive to Parquet.
        baseline_member: Tar member to extract (auto-detect if omitted).
        baseline_column: Column to retain during conversion.
        parquet_output: Destination path for the Parquet file.
        parquet_overwrite: Overwrite existing Parquet file.
        sample_baseline: Sample a .smi file from the Parquet dataset.
        sample_output: Destination path for the sampled .smi file.
        sample_n: Number of SMILES to sample.
        sample_seed: Random seed for reproducibility.
        sample_overwrite: Overwrite existing sample file.
    """
    py = sys.executable

    if fetch_freed_data:
        cmd = [py, str(PROJECT_ROOT / "scripts" / "s0_fetch" / "fetch_freed_fragments.py")]
        if freed_force:
            cmd.append("--force")
        _run(cmd, dry_run)

    baseline_archive = PROJECT_ROOT / baseline_output
    if fetch_baseline:
        cmd = [
            py,
            str(PROJECT_ROOT / "scripts" / "s0_fetch" / "fetch_zinc_baseline.py"),
            "--output",
            baseline_output,
        ]
        if baseline_url:
            cmd.extend(["--url", baseline_url])
        if baseline_overwrite:
            cmd.append("--overwrite")
        _run(cmd, dry_run)

    parquet_path = PROJECT_ROOT / parquet_output
    if convert_baseline:
        if not dry_run and not baseline_archive.exists():
            raise click.ClickException(
                f"Baseline archive not found: {baseline_archive}. Use --fetch-baseline (and --baseline-url if needed)."
            )
        if parquet_path.exists() and not parquet_overwrite:
            logger.info("Parquet already exists, skip conversion: {}", parquet_path)
        else:
            cmd = [
                py,
                str(PROJECT_ROOT / "scripts" / "s1_prepare" / "convert_baseline_parquet.py"),
                "--input",
                baseline_output,
                "--output",
                parquet_output,
                "--column",
                baseline_column,
            ]
            if baseline_member:
                cmd.extend(["--member", baseline_member])
            _run(cmd, dry_run)

    sample_path = PROJECT_ROOT / sample_output
    if sample_baseline:
        if not dry_run and not parquet_path.exists():
            raise click.ClickException(f"Parquet file not found: {parquet_path}. Use --convert-baseline first.")
        if sample_path.exists() and not sample_overwrite:
            logger.info("Sample baseline exists, skip: {}", sample_path)
        else:
            cmd = [
                py,
                str(PROJECT_ROOT / "scripts" / "s1_prepare" / "sample_baseline.py"),
                "--input",
                parquet_output,
                "--output",
                sample_output,
                "--n",
                str(sample_n),
                "--seed",
                str(sample_seed),
                "--column",
                baseline_column,
            ]
            _run(cmd, dry_run)

    logger.success("Data dependency preparation complete.")


if __name__ == "__main__":
    main()
