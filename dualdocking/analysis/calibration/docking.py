"""Calibration set docking: load SMILES, prepare ligands, dock against targets.

Provides the core logic for docking a calibration SMILES set against all
targets to build score distributions used for z-score normalization.
"""

from __future__ import annotations

import json
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from tqdm import tqdm

from dualdocking.config import load_pairs
from dualdocking.scoring.docking import (
    DockerRunner,
    DockingParams,
    dock_batch_unidock,
    dock_ligand,
    docker_image_digest,
    docking_version,
    ensure_ligand_pdbqt,
    ensure_receptor_pdbqt,
    parse_vec,
    resolve_program,
    safe_unlink,
)


def load_calibration_smiles(path: Path) -> list[str]:
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


def sample_smiles(smiles: list[str], n: int, seed: int) -> list[str]:
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


def auto_prep_workers() -> int:
    """Pick number of parallel ligand prep workers based on CPU count.

    Returns:
        Worker count (minimum of CPU count and 8).
    """
    import os as _os

    return min(_os.cpu_count() or 1, 8)


def prep_one_ligand(
    runner: DockerRunner | None,
    smi: str,
    sdf_path: Path,
    pdbqt_path: Path,
    timeout: int,
    seed: int,
    overwrite: bool,
    project_root: Path,
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
        project_root: Project root for container path mapping.

    Returns:
        Tuple of (smiles, pdbqt_path, success, error_message).
    """
    ok, err = ensure_ligand_pdbqt(
        runner,
        smi,
        sdf_path,
        pdbqt_path,
        timeout=timeout,
        seed=seed,
        project_root=project_root,
        overwrite=overwrite,
    )
    return smi, pdbqt_path, ok, err


def run_calibration_docking(  # noqa: PLR0913
    project_root: Path,
    calibration_config: str = "configs/calibration.yaml",
    calibration_set: str | None = None,
    scores_dir: str = "output/data/calibration/scores",
    n: int | None = None,
    seed: int | None = None,
    target_id: tuple[str, ...] | list[str] = (),
    docker_image: str = "dualdocking/freedpp:latest",
    use_docker: bool = True,
    gpus: str = "all",
    docking_program: str = "unidock_gpu",
    docking_binary: str | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    prep_workers: int = 0,
) -> dict[str, Any]:
    """Dock calibration SMILES set against targets to build score distributions.

    Reads the calibration SMILES file, prepares ligand PDBQT files,
    docks each against every target receptor, and writes per-target
    score CSVs and a manifest JSON.

    Args:
        project_root: Project root directory.
        calibration_config: Path to the calibration YAML config (relative to project_root).
        calibration_set: Override path to the calibration SMILES file.
        scores_dir: Output directory for per-target score CSVs (relative to project_root).
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

    Returns:
        Manifest dict with docking metadata and target list.
    """
    from dualdocking.generation.freedpp import _ensure_docker_image

    cfg = yaml.safe_load((project_root / calibration_config).read_text())
    calibration_path = project_root / (calibration_set or cfg.get("output_path"))
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

    scores_root = project_root / scores_dir
    scores_root.mkdir(parents=True, exist_ok=True)
    receptors_dir = project_root / "output" / "data" / "calibration" / "receptors"
    ligands_dir = project_root / "output" / "data" / "calibration" / "ligands"
    work_dir = project_root / "output" / "data" / "calibration" / "work"
    receptors_dir.mkdir(parents=True, exist_ok=True)
    ligands_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    smiles_all = load_calibration_smiles(calibration_path)
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
    smiles: list[str] = []

    pairs = load_pairs(project_root / "configs")["pairs"]
    target_ids: list[str] = []
    for pair in pairs.values():
        target_ids.append(pair["target_a"]["id"])
        target_ids.append(pair["target_b"]["id"])
    target_ids = sorted(set(target_ids))

    if target_id:
        target_ids = [t for t in target_ids if t in set(target_id)]
        if not target_ids:
            raise ValueError("No target IDs matched the provided --target-id values")

    binary = resolve_program(docking_program, docking_binary)
    use_unidock_batch = batch_mode and "unidock" in Path(binary).name.lower()
    if "unidock" in Path(binary).name.lower() and not use_unidock_batch:
        logger.info("Uni-Dock batch mode disabled; using single-ligand docking")

    if dry_run:
        logger.info("Dry run enabled; no docking will be executed")
    elif use_docker:
        _ensure_docker_image(docker_image)

    runner: DockerRunner | None
    runner_ctx = DockerRunner(docker_image, project_root, gpus=gpus) if use_docker else nullcontext(None)
    with runner_ctx as runner:
        if runner:
            logger.info("Using docker image: {}", docker_image)

        # Prepare ligands (parallel when workers > 1)
        ligand_paths: list[Path] = []
        failed_smiles: list[str] = []

        actual_workers = 1 if dry_run else (prep_workers if prep_workers > 0 else auto_prep_workers())

        if actual_workers > 1 and not dry_run:
            _prepare_ligands_parallel(
                runner, candidates, n, ligands_dir, params, seed, overwrite,
                actual_workers, use_existing_manifest, project_root,
                smiles, ligand_paths, failed_smiles,
            )
        else:
            _prepare_ligands_sequential(
                runner, candidates, n, ligands_dir, params, seed, overwrite,
                dry_run, use_existing_manifest, project_root,
                smiles, ligand_paths, failed_smiles,
            )

        if len(smiles) < n:
            raise RuntimeError(f"Only {len(smiles)} ligands prepared, expected {n}")

        if failed_smiles and not use_existing_manifest:
            (ligands_dir / "ligands_failed.smi").write_text("\n".join(failed_smiles) + "\n")

        ligands_manifest.write_text("\n".join(smiles) + "\n")

        dock_ver = docking_version(runner, binary) if not dry_run else "unknown"
        manifest = {
            "docking_program": docking_program,
            "docking_binary": binary,
            "docking_version": dock_ver,
            "docker_image": docker_image,
            "docker_image_digest": docker_image_digest(docker_image),
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

        n_gpus = _detect_gpu_count() if use_docker and gpus == "all" else 1
        if n_gpus > 1 and len(target_ids) > 1 and not dry_run:
            logger.info("Parallel calibration docking: {} targets across {} GPUs", len(target_ids), n_gpus)
            _dock_targets_parallel(
                target_ids, n_gpus, docker_image, project_root,
                scores_root, receptors_dir, work_dir, binary,
                params, ligand_paths, use_unidock_batch, batch_size, seed,
                overwrite, dry_run,
            )
        else:
            for target in target_ids:
                _dock_target(
                    target, scores_root, receptors_dir, work_dir, runner, binary,
                    params, ligand_paths, use_unidock_batch, batch_size, seed,
                    overwrite, dry_run, project_root,
                )

    return manifest


def _detect_gpu_count() -> int:
    """Detect number of available NVIDIA GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],  # noqa: S603, S607
            capture_output=True, text=True, check=True,
        )
        return max(1, len(result.stdout.strip().splitlines()))
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 1


def _dock_targets_parallel(  # noqa: PLR0913
    target_ids: list[str],
    n_gpus: int,
    docker_image: str,
    project_root: Path,
    scores_root: Path,
    receptors_dir: Path,
    work_dir: Path,
    binary: str,
    params: DockingParams,
    ligand_paths: list[Path],
    use_unidock_batch: bool,
    batch_size: int,
    seed: int,
    overwrite: bool,
    dry_run: bool,
) -> None:
    """Dock targets in parallel across GPUs, one container per GPU."""
    # Round-robin distribute targets across GPUs
    gpu_targets: list[list[str]] = [[] for _ in range(n_gpus)]
    for i, target in enumerate(target_ids):
        gpu_targets[i % n_gpus].append(target)

    def dock_on_gpu(gpu_idx: int) -> None:
        targets = gpu_targets[gpu_idx]
        if not targets:
            return
        with DockerRunner(docker_image, project_root, gpus=f'"device={gpu_idx}"') as gpu_runner:
            for target in targets:
                _dock_target(
                    target, scores_root, receptors_dir, work_dir,
                    gpu_runner, binary, params, ligand_paths,
                    use_unidock_batch, batch_size, seed,
                    overwrite, dry_run, project_root,
                )

    with ThreadPoolExecutor(max_workers=n_gpus) as pool:
        futures = [pool.submit(dock_on_gpu, i) for i in range(n_gpus)]
        for f in futures:
            f.result()


def _prepare_ligands_parallel(  # noqa: PLR0913
    runner: DockerRunner | None,
    candidates: list[str],
    n: int,
    ligands_dir: Path,
    params: DockingParams,
    seed: int,
    overwrite: bool,
    actual_workers: int,
    use_existing_manifest: bool,
    project_root: Path,
    smiles: list[str],
    ligand_paths: list[Path],
    failed_smiles: list[str],
) -> None:
    """Prepare ligands in parallel using thread/process pool."""
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

    max_tasks = min(len(candidates), int(n * 1.2) + 50)
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
                prep_one_ligand,
                runner,
                smi,
                sdf_p,
                pdbqt_p,
                params.timeout_seconds,
                seed + task_idx,
                overwrite,
                project_root,
            )
            future_to_idx[f] = task_idx

        results: list[tuple[int, str, Path, bool, str]] = []
        for f in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Ligands", unit="ligand"):
            idx = future_to_idx[f]
            smi_r, path_r, ok_r, err_r = f.result()
            results.append((idx, smi_r, path_r, ok_r, err_r))

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


def _prepare_ligands_sequential(  # noqa: PLR0913
    runner: DockerRunner | None,
    candidates: list[str],
    n: int,
    ligands_dir: Path,
    params: DockingParams,
    seed: int,
    overwrite: bool,
    dry_run: bool,
    use_existing_manifest: bool,
    project_root: Path,
    smiles: list[str],
    ligand_paths: list[Path],
    failed_smiles: list[str],
) -> None:
    """Prepare ligands sequentially (or in dry-run mode)."""
    for smi in tqdm(candidates, desc="Ligands", unit="ligand"):
        if len(smiles) >= n:
            break
        ligand_path = ligands_dir / f"ligand_{len(smiles):05d}.pdbqt"
        sdf_path = ligands_dir / f"ligand_{len(smiles):05d}.sdf"
        if dry_run:
            smiles.append(smi)
            ligand_paths.append(ligand_path)
            continue
        ok, err = ensure_ligand_pdbqt(
            runner,
            smi,
            sdf_path,
            ligand_path,
            timeout=params.timeout_seconds,
            seed=seed + len(smiles),
            project_root=project_root,
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


def _dock_target(  # noqa: PLR0913
    target: str,
    scores_root: Path,
    receptors_dir: Path,
    work_dir: Path,
    runner: DockerRunner | None,
    binary: str,
    params: DockingParams,
    ligand_paths: list[Path],
    use_unidock_batch: bool,
    batch_size: int,
    seed: int,
    overwrite: bool,
    dry_run: bool,
    project_root: Path,
) -> None:
    """Dock all ligands against a single target and write scores."""
    from dualdocking.generation.freedpp import _resolve_pdb_path
    from dualdocking.scoring.binding_box import extract_binding_box

    scores_path = scores_root / f"{target}.csv"
    if scores_path.exists() and not overwrite:
        logger.info("Skip existing scores: {}", scores_path)
        return

    pdb_path = _resolve_pdb_path(target)
    receptor_path = receptors_dir / f"{target}.pdbqt"
    if not dry_run:
        ensure_receptor_pdbqt(
            runner,
            pdb_path,
            receptor_path,
            timeout=params.timeout_seconds,
            project_root=project_root,
            overwrite=overwrite,
        )

    center_str, size_str = extract_binding_box(pdb_path, padding=params.box_padding)
    center = parse_vec(center_str)
    size = parse_vec(size_str)

    scores: list[float] = []
    failures = 0
    target_work = work_dir / target
    target_work.mkdir(parents=True, exist_ok=True)

    if use_unidock_batch and not dry_run:
        scores, failures = _dock_batch(
            runner, binary, receptor_path, ligand_paths, center, size,
            params, target_work, target, batch_size, seed, project_root,
        )
    else:
        scores, failures = _dock_sequential(
            runner, binary, receptor_path, ligand_paths, center, size,
            params, target_work, target, dry_run, project_root,
        )

    if failures:
        logger.warning("{} docking failures for target {}", failures, target)

    import pandas as pd

    from dualdocking.data.io_utils import write_df

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    _scores_df = pd.DataFrame({"score": [round(s, 6) for s in scores]})
    write_df(_scores_df, scores_path.with_suffix(""), csv_too=True)

    logger.info("Scores -> {} (n={})", scores_path, len(scores))


def dock_batch_ligands(  # noqa: PLR0913
    runner: DockerRunner | None,
    binary: str,
    receptor_path: Path,
    ligand_paths: list[Path],
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    params: DockingParams,
    target_work: Path,
    target: str,
    batch_size: int,
    seed: int,
    project_root: Path,
    *,
    cleanup_ligands: bool = False,
) -> tuple[list[float], int]:
    """Dock ligands in batches using UniDock GPU batch mode.

    Args:
        runner: Docker runner (or ``None`` for local execution).
        binary: Docking binary path/name.
        receptor_path: Receptor PDBQT path.
        ligand_paths: Ligand PDBQT paths.
        center: Docking box center.
        size: Docking box size.
        params: Docking parameters.
        target_work: Output work directory.
        target: Target label for logging.
        batch_size: Number of ligands per UniDock batch.
        seed: Base random seed.
        project_root: Project root path.
        cleanup_ligands: Remove ligand files after each batch item is processed.

    Returns:
        Tuple of ``(scores, failures)``.
    """
    scores: list[float] = []
    failures = 0
    batch_iter = range(0, len(ligand_paths), batch_size)
    for batch_id, start in enumerate(tqdm(batch_iter, desc=f"Dock {target}", unit="batch")):
        batch = ligand_paths[start : start + batch_size]
        try:
            batch_scores = dock_batch_unidock(
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
                project_root=project_root,
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
            else:
                scores.append(score)
            if cleanup_ligands:
                safe_unlink(lig, runner, project_root)
    return scores, failures


def dock_sequential_ligands(  # noqa: PLR0913
    runner: DockerRunner | None,
    binary: str,
    receptor_path: Path,
    ligand_paths: list[Path],
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    params: DockingParams,
    target_work: Path,
    target: str,
    dry_run: bool,
    project_root: Path,
    *,
    cleanup_ligands: bool = False,
) -> tuple[list[float], int]:
    """Dock ligands one at a time.

    Args:
        runner: Docker runner (or ``None`` for local execution).
        binary: Docking binary path/name.
        receptor_path: Receptor PDBQT path.
        ligand_paths: Ligand PDBQT paths.
        center: Docking box center.
        size: Docking box size.
        params: Docking parameters.
        target_work: Output work directory.
        target: Target label for logging.
        dry_run: If True, skip docking and emit zero scores.
        project_root: Project root path.
        cleanup_ligands: Remove ligand files after each item is processed.

    Returns:
        Tuple of ``(scores, failures)``.
    """
    scores: list[float] = []
    failures = 0
    for idx, ligand_path in enumerate(tqdm(ligand_paths, desc=f"Dock {target}", unit="ligand")):
        if dry_run:
            scores.append(0.0)
            continue
        log_path = target_work / f"{target}_{idx:05d}.log"
        out_path = target_work / f"{target}_{idx:05d}.pdbqt"
        try:
            score = dock_ligand(
                runner,
                binary,
                receptor_path,
                ligand_path,
                center,
                size,
                params,
                log_path,
                out_path,
                project_root=project_root,
            )
            scores.append(score)
        except Exception as exc:
            failures += 1
            logger.error("Docking failed for {} #{}: {}", target, idx, exc)
            continue
        finally:
            safe_unlink(log_path, runner, project_root)
            safe_unlink(out_path, runner, project_root)
            if cleanup_ligands:
                safe_unlink(ligand_path, runner, project_root)
    return scores, failures


# Backward-compatible aliases for legacy call sites.
_dock_batch = dock_batch_ligands
_dock_sequential = dock_sequential_ligands
