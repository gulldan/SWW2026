"""Dock calibration SMILES set against all targets with Uni-Dock GPU.

Prepares ligand PDBQT files from SMILES (via 3D embedding + Meeko),
docks each ligand against every target's receptor, and writes per-target
score CSV files for downstream normalization.
"""

from __future__ import annotations

import json
import random
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

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


def _load_calibration_smiles(path: Path) -> list[str]:
    """Load SMILES strings from a whitespace-delimited .smi file.

    Args:
        path: Path to the SMILES file (one SMILES per line, first token).

    Returns:
        List of SMILES strings.
    """
    smiles: list[str] = []
    with path.open() as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            smiles.append(stripped.split()[0])
    return smiles


def _sample_smiles(smiles: list[str], n: int, seed: int) -> list[str]:
    """Subsample SMILES if needed.

    Args:
        smiles: Full list of SMILES strings.
        n: Desired sample size.
        seed: Random seed.

    Returns:
        At most *n* sampled SMILES.
    """
    if n >= len(smiles):
        return smiles
    rng = random.Random(seed)  # noqa: S311
    return rng.sample(smiles, n)


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
    """Dock a single ligand — thin wrapper delegating to shared module."""
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
    """Dock a batch of ligands — thin wrapper delegating to shared module."""
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
    """Prepare one ligand PDBQT file (thread-safe).

    Args:
        runner: Docker runner (or None for local execution).
        smi: SMILES string.
        sdf_path: Intermediate SDF file path.
        pdbqt_path: Output PDBQT file path.
        timeout: Subprocess timeout in seconds.
        seed: Random seed for 3D embedding.
        overwrite: If True, regenerate even if the file exists.

    Returns:
        Tuple of (smiles, pdbqt_path, success, error_message).
    """
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
    """Pick number of parallel ligand prep workers based on CPU count.

    Returns:
        Worker count (minimum of CPU count and 8).
    """
    import os as _os

    return min(_os.cpu_count() or 1, 8)


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
def main(  # noqa: PLR0913, PLR0915
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
    """Dock calibration SMILES set against targets to build score distributions.

    Reads the calibration SMILES file, prepares ligand PDBQT files,
    docks each against every target receptor, and writes per-target
    score CSVs and a manifest JSON.

    Args:
        calibration_config: Path to the calibration YAML config.
        calibration_set: Override path to the calibration SMILES file.
        scores_dir: Output directory for per-target score CSVs.
        n: Number of ligands to dock (from calibration config if omitted).
        seed: Sampling seed (from calibration config if omitted).
        target_id: Restrict to these target IDs (all if empty).
        docker_image: Docker image for docking tools.
        use_docker: If True, run docking inside Docker.
        gpus: Docker ``--gpus`` value.
        docking_program: Logical docking program name.
        docking_binary: Override docking binary path.
        overwrite: If True, overwrite existing score files.
        dry_run: If True, skip actual docking.
        prep_workers: Number of parallel ligand preparation workers.
    """
    import yaml

    from dualdocking.config import load_pairs
    from dualdocking.generation.freedpp import _ensure_docker_image, _resolve_pdb_path
    from dualdocking.scoring.binding_box import extract_binding_box

    cfg = yaml.safe_load((PROJECT_ROOT / calibration_config).read_text())
    calibration_path = PROJECT_ROOT / (calibration_set or cfg.get("output_path"))
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration set not found: {calibration_path}")

    sample_cfg = cfg.get("sample", {})
    n = n or int(sample_cfg.get("n", 1000))
    seed = seed or int(sample_cfg.get("seed", 137))

    docking_cfg = cfg.get("docking", {})
    params = DockingParams(
        exhaustiveness=int(docking_cfg.get("exhaustiveness", 8)),
        num_modes=int(docking_cfg.get("num_modes", 9)),
        energy_range=float(docking_cfg.get("energy_range", 3.0)),
        timeout_seconds=int(docking_cfg.get("timeout_seconds", 120)),
        box_padding=float(docking_cfg.get("box_padding", 10.0)),
    )
    batch_size = int(docking_cfg.get("batch_size", 64))
    batch_mode = bool(docking_cfg.get("batch_mode", False))

    scores_root = PROJECT_ROOT / scores_dir
    scores_root.mkdir(parents=True, exist_ok=True)
    receptors_dir = PROJECT_ROOT / "output" / "data" / "calibration" / "receptors"
    ligands_dir = PROJECT_ROOT / "output" / "data" / "calibration" / "ligands"
    work_dir = PROJECT_ROOT / "output" / "data" / "calibration" / "work"
    receptors_dir.mkdir(parents=True, exist_ok=True)
    ligands_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    smiles_all = _load_calibration_smiles(calibration_path)
    logger.info("Calibration set: {} ({} total, target n={})", calibration_path, len(smiles_all), n)

    ligands_manifest = ligands_dir / "ligands.smi"
    use_existing_manifest = ligands_manifest.exists() and not overwrite
    if use_existing_manifest:
        existing = [line.strip().split()[0] for line in ligands_manifest.read_text().splitlines() if line.strip()]
        if n is not None and len(existing) != n:
            logger.warning(
                "Ligands manifest has {} entries, requested n={}; reusing manifest",
                len(existing),
                n,
            )
        n = len(existing)
        candidates = existing
    else:
        rng = random.Random(seed)  # noqa: S311
        candidates = smiles_all.copy()
        rng.shuffle(candidates)
    smiles = []

    pairs = load_pairs(PROJECT_ROOT / "configs")["pairs"]
    target_ids: list[str] = []
    for pair in pairs.values():
        target_ids.append(pair["target_a"]["id"])
        target_ids.append(pair["target_b"]["id"])
    target_ids = sorted(set(target_ids))

    if target_id:
        target_ids = [t for t in target_ids if t in set(target_id)]
        if not target_ids:
            raise ValueError("No target IDs matched the provided --target-id values")

    binary = _resolve_program(docking_program, docking_binary)
    use_unidock_batch = batch_mode and "unidock" in Path(binary).name.lower()
    if "unidock" in Path(binary).name.lower() and not use_unidock_batch:
        logger.info("Uni-Dock batch mode disabled; using single-ligand docking")

    if dry_run:
        logger.info("Dry run enabled; no docking will be executed")
    elif use_docker:
        _ensure_docker_image(docker_image)

    runner: DockerRunner | None
    runner_ctx = DockerRunner(docker_image, PROJECT_ROOT, gpus=gpus) if use_docker else nullcontext(None)
    with runner_ctx as runner:
        if runner:
            logger.info("Using docker image: {}", docker_image)

        # Prepare ligands (parallel when workers > 1)
        ligand_paths: list[Path] = []
        failed_smiles: list[str] = []

        actual_workers = 1 if dry_run else (prep_workers if prep_workers > 0 else _auto_prep_workers())

        if actual_workers > 1 and not dry_run:
            from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

            # Pre-assign indices; submit extra candidates to cover failures
            max_tasks = min(len(candidates), int(n * 1.2) + 50)
            # ProcessPoolExecutor for local RDKit (CPU-bound, GIL-releasing C++);
            # ThreadPoolExecutor for Docker mode (subprocess I/O, runner not picklable).
            pool_class = ProcessPoolExecutor if runner is None else ThreadPoolExecutor
            logger.info(
                "Parallel ligand prep: {} candidates on {} {} workers (need {})",
                max_tasks,
                actual_workers,
                "process" if runner is None else "thread",
                n,
            )

            future_to_idx: dict = {}
            with pool_class(max_workers=actual_workers) as pool:
                for task_idx in range(max_tasks):
                    smi = candidates[task_idx]
                    sdf_p = ligands_dir / f"ligand_{task_idx:05d}.sdf"
                    pdbqt_p = ligands_dir / f"ligand_{task_idx:05d}.pdbqt"
                    f = pool.submit(
                        _prep_one_ligand,
                        runner,
                        smi,
                        sdf_p,
                        pdbqt_p,
                        params.timeout_seconds,
                        seed + task_idx,
                        overwrite,
                    )
                    future_to_idx[f] = task_idx

                # Collect results with progress bar
                results: list[tuple[int, str, Path, bool, str]] = []
                for f in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Ligands", unit="ligand"):
                    idx = future_to_idx[f]
                    smi_r, path_r, ok_r, err_r = f.result()
                    results.append((idx, smi_r, path_r, ok_r, err_r))

            # Sort by original candidate order for reproducibility
            results.sort(key=lambda r: r[0])
            for _idx, smi_r, path_r, ok_r, err_r in results:
                if len(smiles) >= n:
                    break
                if ok_r:
                    smiles.append(smi_r)
                    ligand_paths.append(path_r)
                else:
                    failed_smiles.append(smi_r)
                    logger.warning("Ligand prep failed: {} ({})", smi_r, err_r or "no error text")
                    if use_existing_manifest:
                        raise RuntimeError("Ligand prep failed for existing manifest; rerun with --overwrite")
        else:
            # Sequential (original path -- also used for dry_run)
            for smi in tqdm(candidates, desc="Ligands", unit="ligand"):
                if len(smiles) >= n:
                    break
                ligand_path = ligands_dir / f"ligand_{len(smiles):05d}.pdbqt"
                sdf_path = ligands_dir / f"ligand_{len(smiles):05d}.sdf"
                if dry_run:
                    smiles.append(smi)
                    ligand_paths.append(ligand_path)
                    continue
                ok, err = _ensure_ligand_pdbqt(
                    runner,
                    smi,
                    sdf_path,
                    ligand_path,
                    timeout=params.timeout_seconds,
                    seed=seed + len(smiles),
                    overwrite=overwrite,
                )
                if not ok:
                    failed_smiles.append(smi)
                    logger.warning("Ligand prep failed: {} ({})", smi, err or "no error text")
                    if use_existing_manifest:
                        raise RuntimeError("Ligand prep failed for existing manifest; rerun with --overwrite")
                    continue
                smiles.append(smi)
                ligand_paths.append(ligand_path)

        if len(smiles) < n:
            raise RuntimeError(f"Only {len(smiles)} ligands prepared, expected {n}")

        if failed_smiles and not use_existing_manifest:
            (ligands_dir / "ligands_failed.smi").write_text("\n".join(failed_smiles) + "\n")

        ligands_manifest.write_text("\n".join(smiles) + "\n")

        docking_version = _docking_version(runner, binary) if not dry_run else "unknown"
        manifest = {
            "docking_program": docking_program,
            "docking_binary": binary,
            "docking_version": docking_version,
            "docker_image": docker_image,
            "docker_image_digest": _docker_image_digest(docker_image),
            "calibration_set": str(calibration_set or cfg.get("output_path")),
            "n": len(smiles),
            "seed": seed,
            "params": {
                "exhaustiveness": params.exhaustiveness,
                "num_modes": params.num_modes,
                "energy_range": params.energy_range,
                "timeout_seconds": params.timeout_seconds,
                "box_padding": params.box_padding,
                "batch_mode": use_unidock_batch,
                "batch_size": batch_size if use_unidock_batch else 1,
            },
            "targets": target_ids,
            "date": datetime.now(UTC).date().isoformat(),
        }
        (scores_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

        for target in target_ids:
            scores_path = scores_root / f"{target}.csv"
            if scores_path.exists() and not overwrite:
                logger.info("Skip existing scores: {}", scores_path)
                continue

            pdb_path = _resolve_pdb_path(target)
            receptor_path = receptors_dir / f"{target}.pdbqt"
            if not dry_run:
                _ensure_receptor_pdbqt(
                    runner,
                    pdb_path,
                    receptor_path,
                    timeout=params.timeout_seconds,
                    overwrite=overwrite,
                )

            center_str, size_str = extract_binding_box(pdb_path, padding=params.box_padding)
            center = _parse_vec(center_str)
            size = _parse_vec(size_str)

            scores: list[float] = []
            failures = 0
            target_work = work_dir / target
            target_work.mkdir(parents=True, exist_ok=True)

            if use_unidock_batch and not dry_run:
                batch_iter = range(0, len(ligand_paths), batch_size)
                for batch_id, start in enumerate(tqdm(batch_iter, desc=f"Dock {target}", unit="batch")):
                    batch = ligand_paths[start : start + batch_size]
                    try:
                        batch_scores = _dock_batch_unidock(
                            runner,
                            binary,
                            receptor_path,
                            batch,
                            center,
                            size,
                            params,
                            target_work,
                            batch_id=batch_id,
                            seed=seed + start,
                            timeout=params.timeout_seconds,
                        )
                    except Exception as exc:
                        failures += len(batch)
                        logger.error("Batch docking failed for {} batch {}: {}", target, batch_id, exc)
                        continue
                    for lig in batch:
                        score = batch_scores.get(lig)
                        if score is None:
                            failures += 1
                            logger.error("Docking failed for {} ligand {}", target, lig.name)
                            continue
                        scores.append(score)
            else:
                for idx, ligand_path in enumerate(tqdm(ligand_paths, desc=f"Dock {target}", unit="ligand")):
                    if dry_run:
                        scores.append(0.0)
                        continue
                    log_path = target_work / f"{target}_{idx:05d}.log"
                    out_path = target_work / f"{target}_{idx:05d}.pdbqt"
                    try:
                        score = _dock_ligand(
                            runner,
                            binary,
                            receptor_path,
                            ligand_path,
                            center,
                            size,
                            params,
                            log_path,
                            out_path,
                        )
                        scores.append(score)
                    except Exception as exc:
                        failures += 1
                        logger.error("Docking failed for {} #{}: {}", target, idx, exc)
                        continue
                    finally:
                        if log_path.exists():
                            log_path.unlink(missing_ok=True)
                        if out_path.exists():
                            out_path.unlink(missing_ok=True)

            if failures:
                logger.warning("{} docking failures for target {}", failures, target)

            import pandas as pd

            from dualdocking.data.io_utils import write_df

            scores_path.parent.mkdir(parents=True, exist_ok=True)
            _scores_df = pd.DataFrame({"score": [round(s, 6) for s in scores]})
            write_df(_scores_df, scores_path.with_suffix(""), csv_too=True)

            logger.info("Scores -> {} (n={})", scores_path, len(scores))


if __name__ == "__main__":
    main()
