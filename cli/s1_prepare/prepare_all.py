"""One-shot preparation of local data dependencies for experiments.

Thin CLI wrapper -- core logic lives in :mod:`dualdocking.pipeline.stages.prepare`.

Orchestrates fetching FREED docker data, downloading the baseline
archive, converting it to Parquet, and optionally sampling a baseline
SMILES file.
"""

from __future__ import annotations

from pathlib import Path

import click
from loguru import logger

from dualdocking.command_runner import run_project_command
from dualdocking.pipeline.stages.prepare import prepare_all

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def _run(cmd: list[str], dry_run: bool) -> None:
    """Run an internal project command with in-process optimization.

    Args:
        cmd: Command and arguments to execute.
        dry_run: If True, only log the command without executing.

    Raises:
        click.ClickException: If the command exits with a non-zero code.
    """
    logger.info("RUN {}", " ".join(cmd))
    if dry_run:
        return
    try:
        result = run_project_command(
            cmd,
            project_root=PROJECT_ROOT,
            capture_output=True,
            allow_inprocess=True,
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

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
    """Prepare FREED docker data, baseline archive, Parquet and optional sample."""
    prepare_all(
        PROJECT_ROOT,
        fetch_freed=fetch_freed_data,
        freed_force=freed_force,
        fetch_baseline=fetch_baseline,
        baseline_url=baseline_url,
        baseline_output=baseline_output,
        baseline_overwrite=baseline_overwrite,
        convert_baseline_flag=convert_baseline,
        baseline_member=baseline_member,
        baseline_column=baseline_column,
        parquet_output=parquet_output,
        parquet_overwrite=parquet_overwrite,
        sample_baseline_flag=sample_baseline,
        sample_output=sample_output,
        sample_n=sample_n,
        sample_seed=sample_seed,
        sample_overwrite=sample_overwrite,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()
