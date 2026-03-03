"""Dock calibration SMILES set against all targets with Uni-Dock GPU.

Prepares ligand PDBQT files from SMILES (via 3D embedding + Meeko),
docks each ligand against every target's receptor, and writes per-target
score CSV files for downstream normalization.
"""

from __future__ import annotations

from pathlib import Path

import click

from dualdocking.analysis.calibration.docking import (
    load_calibration_smiles,
    run_calibration_docking,
    sample_smiles,
)
from dualdocking.scoring.docking import (
    DockerRunner,
    DockingParams,
    container_path,
    dock_batch_unidock,
    dock_ligand,
    docker_image_digest,
    docking_version,
    ensure_ligand_pdbqt,
    ensure_receptor_pdbqt,
    parse_score,
    parse_unidock_scores_sdf,
    parse_vec,
    resolve_program,
    safe_unlink,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


# Public aliases used by enrichment_benchmark.py cross-import
_load_calibration_smiles = load_calibration_smiles
_sample_smiles = sample_smiles


# ---------------------------------------------------------------------------
# Backward-compatible aliases for functions moved to dualdocking.scoring.docking.
# These thin wrappers bind PROJECT_ROOT so call-sites inside this file
# (and any remaining cross-script imports) keep working unchanged.
# ---------------------------------------------------------------------------

_resolve_program = resolve_program
_parse_score = parse_score
_docker_image_digest = docker_image_digest
_parse_unidock_scores_sdf = parse_unidock_scores_sdf


def _container_path(path: Path) -> str:
    """Shorthand: convert host path to container path using PROJECT_ROOT."""
    return container_path(path, PROJECT_ROOT)


def _safe_unlink(path: Path, runner: DockerRunner | None) -> None:
    """Shorthand: safe_unlink bound to PROJECT_ROOT."""
    safe_unlink(path, runner, PROJECT_ROOT)


def _ensure_receptor_pdbqt(
    runner: DockerRunner | None,
    pdb_path: Path,
    pdbqt_path: Path,
    timeout: int,
    overwrite: bool = False,
) -> None:
    """Ensure a receptor PDBQT file exists, preparing it from PDB if needed.

    Args:
        runner: Docker runner (or None for local execution).
        pdb_path: Input PDB file.
        pdbqt_path: Output PDBQT file.
        timeout: Subprocess timeout in seconds.
        overwrite: If True, regenerate even if the file exists.

    Raises:
        RuntimeError: If hydrogen addition or receptor preparation fails.
    """
    ensure_receptor_pdbqt(runner, pdb_path, pdbqt_path, timeout, project_root=PROJECT_ROOT, overwrite=overwrite)


def _ensure_ligand_pdbqt(
    runner: DockerRunner | None,
    smiles: str,
    sdf_path: Path,
    out_path: Path,
    timeout: int,
    seed: int,
    overwrite: bool = False,
) -> tuple[bool, str]:
    """Ensure a ligand PDBQT file is prepared from SMILES.

    Args:
        runner: Docker runner (or None for local execution).
        smiles: Input SMILES string.
        sdf_path: Intermediate SDF file path.
        out_path: Output PDBQT file path.
        timeout: Subprocess timeout in seconds.
        seed: Random seed for 3D embedding.
        overwrite: If True, regenerate even if the file exists.

    Returns:
        Tuple of (success, error_message).
    """
    return ensure_ligand_pdbqt(
        runner,
        smiles,
        sdf_path,
        out_path,
        timeout,
        seed,
        project_root=PROJECT_ROOT,
        overwrite=overwrite,
    )


def _dock_ligand(
    runner: DockerRunner | None,
    binary: str,
    receptor: Path,
    ligand: Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    params: DockingParams,
    log_path: Path,
    out_path: Path,
) -> float:
    """Dock a single ligand -- thin wrapper delegating to shared module."""
    return dock_ligand(
        runner,
        binary,
        receptor,
        ligand,
        center,
        size,
        params,
        log_path,
        out_path,
        project_root=PROJECT_ROOT,
    )


def _dock_batch_unidock(  # noqa: PLR0913
    runner: DockerRunner | None,
    binary: str,
    receptor: Path,
    ligands: list[Path],
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    params: DockingParams,
    output_dir: Path,
    batch_id: int,
    seed: int,
    timeout: int,
) -> dict[Path, float | None]:
    """Dock a batch of ligands -- thin wrapper delegating to shared module."""
    return dock_batch_unidock(
        runner,
        binary,
        receptor,
        ligands,
        center,
        size,
        params,
        output_dir,
        batch_id=batch_id,
        seed=seed,
        timeout=timeout,
        project_root=PROJECT_ROOT,
    )


def _prep_one_ligand(
    runner: DockerRunner | None,
    smi: str,
    sdf_path: Path,
    pdbqt_path: Path,
    timeout: int,
    seed: int,
    overwrite: bool,
) -> tuple[str, Path, bool, str]:
    """Prepare one ligand PDBQT file (thread-safe)."""
    ok, err = _ensure_ligand_pdbqt(
        runner,
        smi,
        sdf_path,
        pdbqt_path,
        timeout=timeout,
        seed=seed,
        overwrite=overwrite,
    )
    return smi, pdbqt_path, ok, err


def _auto_prep_workers() -> int:
    """Pick number of parallel ligand prep workers based on CPU count."""
    from dualdocking.analysis.calibration.docking import auto_prep_workers

    return auto_prep_workers()


_parse_vec = parse_vec

_docking_version = docking_version


@click.command()
@click.option("--calibration-config", default="configs/calibration.yaml")
@click.option("--calibration-set", default=None)
@click.option("--scores-dir", default="output/data/calibration/scores")
@click.option("--n", default=None, type=int, help="Number of ligands to dock")
@click.option("--seed", default=None, type=int, help="Sampling seed")
@click.option("--target-id", multiple=True, help="Target ID to dock (repeatable)")
@click.option("--docker-image", default="dualdocking/freedpp:latest")
@click.option("--use-docker/--no-docker", default=True)
@click.option("--gpus", default="all", help="Docker --gpus value")
@click.option("--docking-program", default="unidock_gpu")
@click.option("--docking-binary", default=None, help="Override docking binary")
@click.option("--overwrite/--skip-existing", default=False)
@click.option("--dry-run", is_flag=True)
@click.option(
    "--prep-workers",
    default=0,
    type=int,
    show_default=True,
    help="Parallel ligand prep workers. 0=auto (min(cpu_count, 8)), 1=sequential.",
)
def main(  # noqa: PLR0913
    calibration_config: str,
    calibration_set: str | None,
    scores_dir: str,
    n: int | None,
    seed: int | None,
    target_id: tuple[str, ...],
    docker_image: str,
    use_docker: bool,
    gpus: str,
    docking_program: str,
    docking_binary: str | None,
    overwrite: bool,
    dry_run: bool,
    prep_workers: int,
) -> None:
    """Dock calibration SMILES set against targets to build score distributions."""
    run_calibration_docking(
        project_root=PROJECT_ROOT,
        calibration_config=calibration_config,
        calibration_set=calibration_set,
        scores_dir=scores_dir,
        n=n,
        seed=seed,
        target_id=target_id,
        docker_image=docker_image,
        use_docker=use_docker,
        gpus=gpus,
        docking_program=docking_program,
        docking_binary=docking_binary,
        overwrite=overwrite,
        dry_run=dry_run,
        prep_workers=prep_workers,
    )


if __name__ == "__main__":
    main()
