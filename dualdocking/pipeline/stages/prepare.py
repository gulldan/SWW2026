"""Orchestrator for the prepare stage (Stage 1).

Coordinates target preparation, baseline conversion, and sampling.

Extracted from ``cli/s1_prepare/prepare_all.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from dualdocking.data.fetch.docker import fetch_freed_data
from dualdocking.data.prep.baseline import convert_to_parquet, sample_baseline
from dualdocking.data.prep.targets import PrepCheckpoint, prepare_pair


def prepare_all_targets(
    pairs: dict[str, dict],
    *,
    data_dir: Path,
    msa_cache_dir: Path,
    project_root: Path,
    checkpoint_path: Path,
    full: bool = False,
) -> None:
    """Prepare all target pairs from config.

    Args:
        pairs: Dict of pair_id to pair config from pairs.yaml.
        data_dir: Base directory for target output files.
        msa_cache_dir: Directory for cached MSA files.
        project_root: Project root directory.
        checkpoint_path: Path to the checkpoint JSON file.
        full: If True, run all preparation steps (PDB + FASTA + MSA + PDBQT).
    """
    checkpoint = PrepCheckpoint.load(checkpoint_path)

    for pair_id, pair_cfg in pairs.items():
        logger.info("Preparing pair: {}", pair_id)
        prepare_pair(
            pair_id,
            pair_cfg,
            data_dir=data_dir,
            msa_cache_dir=msa_cache_dir,
            project_root=project_root,
            checkpoint=checkpoint,
            full=full,
        )

    if full:
        logger.info("Full preparation complete. Scoring pipeline ready for OFFLINE operation.")
    logger.info("Done. Targets saved to {}", data_dir)


def prepare_all(  # noqa: PLR0913
    project_root: Path,
    *,
    fetch_freed: bool = True,
    freed_force: bool = False,
    fetch_baseline: bool = False,
    baseline_url: str | None = None,
    baseline_output: str = "output/data/baseline/zinc-druglike-cano.tar.xz",
    baseline_overwrite: bool = False,
    convert_baseline_flag: bool = False,
    baseline_member: str | None = None,
    baseline_column: str = "smiles",
    parquet_output: str = "output/data/baseline/zinc-druglike-cano.parquet",
    parquet_overwrite: bool = False,
    sample_baseline_flag: bool = False,
    sample_output: str = "output/data/baseline/zinc_druglike_baseline_1000_seed42.smi",
    sample_n: int = 1000,
    sample_seed: int = 42,
    sample_overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """Run all preparation steps.

    Args:
        project_root: Project root directory.
        fetch_freed: Download FREED fragment library and alerts.
        freed_force: Overwrite existing FREED data files.
        fetch_baseline: Download the baseline ZINC archive.
        baseline_url: Override URL for baseline download.
        baseline_output: Local path for the baseline archive.
        baseline_overwrite: Overwrite existing baseline archive.
        convert_baseline_flag: Convert the baseline archive to Parquet.
        baseline_member: Tar member to extract.
        baseline_column: Column to retain during conversion.
        parquet_output: Destination path for the Parquet file.
        parquet_overwrite: Overwrite existing Parquet file.
        sample_baseline_flag: Sample a .smi file from the Parquet dataset.
        sample_output: Destination path for the sampled .smi file.
        sample_n: Number of SMILES to sample.
        sample_seed: Random seed for reproducibility.
        sample_overwrite: Overwrite existing sample file.
        dry_run: Print commands without executing them.
    """
    if fetch_freed:
        if dry_run:
            logger.info("[DRY RUN] fetch_freed_data")
        else:
            out_dir = project_root / "docker" / "freedpp" / "data"
            fetch_freed_data(out_dir, force=freed_force)

    baseline_archive = project_root / baseline_output
    if fetch_baseline:
        if dry_run:
            logger.info("[DRY RUN] fetch_zinc_baseline")
        else:
            cmd = [sys.executable, str(project_root / "cli" / "s0_fetch" / "fetch_zinc_baseline.py")]
            if baseline_url:
                cmd.extend(["--url", baseline_url])
            if baseline_overwrite:
                cmd.append("--overwrite")

            import subprocess

            subprocess.run(cmd, cwd=str(project_root), check=True, timeout=1800)  # noqa: S603

    parquet_path = project_root / parquet_output
    if convert_baseline_flag:
        if not dry_run and not baseline_archive.exists():
            raise RuntimeError(f"Baseline archive not found: {baseline_archive}. Use fetch_baseline first.")
        if parquet_path.exists() and not parquet_overwrite:
            logger.info("Parquet already exists, skip conversion: {}", parquet_path)
        elif not dry_run:
            convert_to_parquet(
                baseline_archive,
                parquet_path,
                member=baseline_member,
                column=baseline_column,
            )

    sample_path = project_root / sample_output
    if sample_baseline_flag:
        if not dry_run and not parquet_path.exists():
            raise RuntimeError(f"Parquet file not found: {parquet_path}. Use convert_baseline first.")
        if sample_path.exists() and not sample_overwrite:
            logger.info("Sample baseline exists, skip: {}", sample_path)
        elif not dry_run:
            sample_baseline(
                parquet_path,
                sample_path,
                n=sample_n,
                seed=sample_seed,
                column=baseline_column,
            )

    logger.info("Data dependency preparation complete.")
