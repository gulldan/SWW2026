"""GPU-batch docking helpers for the FREED++ dual reward patch."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from subprocess import TimeoutExpired, run
from tempfile import TemporaryDirectory
from typing import Any

from dual_reward_patch_deps import prepare_ligand_pdbqt
from dual_reward_patch_state import (
    _MAX_HEAVY_ATOMS,
    _MAX_ROTATABLE_BONDS,
    _coords,
    _parse_result_from_pdbqt,
    _receptor_label,
    logger,
)

# ---------------------------------------------------------------------------
# GPU batch docking (Phase 2 optimisation)
# ---------------------------------------------------------------------------
# When enabled, CacheAndPool for DockingScore pre-computes all docking scores
# via a single ``unidock --gpu_batch`` call instead of N×num_sub_proc CPU calls.
_GPU_BATCH_ENABLED = os.environ.get("GPU_BATCH_DOCK", "1") != "0"
_gpu_dock_cache: dict[tuple[str, str], float] = {}  # (receptor_path, smiles) → score
_gpu_batch_log_count = 0


def _read_positive_int_env(name: str, default: int) -> int:
    """Read positive integer environment variable with safe fallback."""
    raw = os.environ.get(name, str(default)).strip()
    try:
        parsed = int(raw)
    except ValueError:
        logger.warning("gpu_batch_dock: invalid %s=%r, using default=%d", name, raw, default)
        return default
    if parsed < 1:
        logger.warning("gpu_batch_dock: %s must be >=1, got %d; using default=%d", name, parsed, default)
        return default
    return parsed


_MP_START_METHOD = os.environ.get("FREEDPP_PREP_MP_START_METHOD", "spawn").strip().lower() or "spawn"
_VALID_MP_START_METHODS = {"spawn", "forkserver", "fork"}
_PREP_MAX_WORKERS = _read_positive_int_env("FREEDPP_PREP_MAX_WORKERS", 8)
_GPU_BATCH_CHUNK_SIZE = _read_positive_int_env("GPU_BATCH_CHUNK_SIZE", 16)
_GPU_BATCH_TIMEOUT_SEC = _read_positive_int_env("GPU_BATCH_TIMEOUT_SEC", 600)


def _cleanup_gpu_cache_entries(primary_receptor: str, dual_receptor_b: str | None, smiles: list[str]) -> None:
    """Remove precomputed entries consumed by current CacheAndPool call."""
    for smi in smiles:
        _gpu_dock_cache.pop((primary_receptor, smi), None)
        if dual_receptor_b is not None:
            _gpu_dock_cache.pop((dual_receptor_b, smi), None)


def _prepare_pdbqt_worker(args: tuple[str, int, str]) -> bool:
    """Worker for parallel ligand preparation via ``multiprocessing.Pool.map``."""
    smiles, seed, outpath = args
    return prepare_ligand_pdbqt(
        smiles,
        outpath,
        seed=seed,
        max_heavy=_MAX_HEAVY_ATOMS,
        max_rotatable=_MAX_ROTATABLE_BONDS,
    )


def _prepare_ligands_parallel(tasks: list[tuple[str, int, str]], n_workers: int) -> list[bool]:
    """Prepare ligands in parallel using a configurable multiprocessing context."""
    if n_workers <= 1:
        return [_prepare_pdbqt_worker(task) for task in tasks]

    start_method = _MP_START_METHOD if _MP_START_METHOD in _VALID_MP_START_METHODS else "spawn"
    if start_method != _MP_START_METHOD:
        logger.warning(
            "gpu_batch_dock: invalid FREEDPP_PREP_MP_START_METHOD=%s, falling back to '%s'",
            _MP_START_METHOD,
            start_method,
        )

    try:
        ctx = mp.get_context(start_method)
    except ValueError:
        logger.warning("gpu_batch_dock: unsupported start method '%s', falling back to 'spawn'", start_method)
        ctx = mp.get_context("spawn")

    with ctx.Pool(processes=n_workers) as pool:
        return pool.map(_prepare_pdbqt_worker, tasks)


def _iter_chunks(items: list[tuple[int, tuple[str, int, str]]], chunk_size: int) -> list[list[tuple[int, tuple[str, int, str]]]]:
    """Split *items* into contiguous chunks of size at most *chunk_size*."""
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _run_gpu_batch(
    vina_program: str,
    receptor: str,
    center: list[float],
    size: list[float],
    pdbqt_paths: list[str],
    output_dir: str,
    exhaustiveness: int,
    num_modes: int,
    seed: int,
    timeout: int = _GPU_BATCH_TIMEOUT_SEC,
) -> bool:
    """Run UniDock ``--gpu_batch`` for a list of prepared PDBQT files.

    Returns:
        ``True`` on success, ``False`` on failure.
    """
    receptor_name = _receptor_label(receptor)
    logger.info(
        "gpu_batch_dock: stage=run start receptor=%s ligands=%d exhaustiveness=%d num_modes=%d seed=%d timeout=%ds",
        receptor_name,
        len(pdbqt_paths),
        exhaustiveness,
        num_modes,
        seed,
        timeout,
    )
    logger.debug(
        "gpu_batch_dock: stage=run box receptor=%s center=(%.3f, %.3f, %.3f) size=(%.3f, %.3f, %.3f)",
        receptor_name,
        center[0],
        center[1],
        center[2],
        size[0],
        size[1],
        size[2],
    )

    cmd = [
        vina_program,
        "--receptor",
        receptor,
        "--gpu_batch",
        *pdbqt_paths,
        "--dir",
        output_dir,
        "--center_x",
        str(center[0]),
        "--center_y",
        str(center[1]),
        "--center_z",
        str(center[2]),
        "--size_x",
        str(size[0]),
        "--size_y",
        str(size[1]),
        "--size_z",
        str(size[2]),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        str(num_modes),
        "--seed",
        str(seed),
    ]
    try:
        result = run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("gpu_batch_dock: unidock rc=%s: %s", result.returncode, (result.stderr or "")[-500:])
            return False
        logger.info("gpu_batch_dock: stage=run done receptor=%s", receptor_name)
        return True
    except TimeoutExpired:
        logger.warning("gpu_batch_dock: stage=run timeout receptor=%s after %ss", receptor_name, timeout)
        return False


def _precompute_gpu_batch_for_receptor(
    unseen_smiles: list[str],
    config: dict[str, Any],
    receptor: str,
    center: Any,
    size: Any,
) -> None:
    """Pre-compute docking scores for *unseen_smiles* against one receptor.

    Results are stored in ``_gpu_dock_cache[(receptor, smi)]``.
    """
    global _gpu_batch_log_count

    error_val = config["error_val"]
    seed = config["seed"]
    receptor_name = _receptor_label(receptor)
    unseen_count = len(unseen_smiles)
    logger.info(
        "gpu_batch_dock: stage=precompute start receptor=%s unseen=%d",
        receptor_name,
        unseen_count,
    )

    if not unseen_smiles:
        logger.debug("gpu_batch_dock: stage=precompute skipped receptor=%s (no unseen ligands)", receptor_name)
        return

    center_f = _coords(center) if isinstance(center, str) else [float(x) for x in center]
    size_f = _coords(size) if isinstance(size, str) else [float(x) for x in size]

    with TemporaryDirectory(prefix="gpu_batch_") as tmpdir:
        lig_dir = Path(tmpdir) / "ligands"
        out_dir = Path(tmpdir) / "output"
        lig_dir.mkdir()
        out_dir.mkdir()

        tasks: list[tuple[str, int, str]] = []
        for i, smi in enumerate(unseen_smiles):
            fname = f"lig_{i:06d}.pdbqt"
            tasks.append((smi, seed, str(lig_dir / fname)))

        cpu_count = os.cpu_count() or 1
        n_workers = min(len(tasks), cpu_count, _PREP_MAX_WORKERS)
        logger.info(
            "gpu_batch_dock: stage=prepare receptor=%s workers=%d tasks=%d cpu_count=%d workers_cap=%d",
            receptor_name,
            n_workers,
            len(tasks),
            cpu_count,
            _PREP_MAX_WORKERS,
        )
        prep_ok = _prepare_ligands_parallel(tasks, n_workers)

        valid_items = [(i, tasks[i]) for i in range(len(tasks)) if prep_ok[i]]
        valid_paths = [t[2] for _, t in valid_items]
        prep_failed = len(tasks) - len(valid_items)
        logger.info(
            "gpu_batch_dock: stage=prepare done receptor=%s prepared=%d failed=%d",
            receptor_name,
            len(valid_items),
            prep_failed,
        )

        for i in range(len(tasks)):
            if not prep_ok[i]:
                _gpu_dock_cache[(receptor, tasks[i][0])] = error_val

        if not valid_paths:
            logger.warning(
                "gpu_batch_dock: stage=precompute no_valid_ligands receptor=%s all=%d",
                receptor_name,
                len(tasks),
            )
            return

        chunk_size = min(len(valid_items), _GPU_BATCH_CHUNK_SIZE)
        chunks = _iter_chunks(valid_items, chunk_size)
        logger.info(
            "gpu_batch_dock: stage=run_chunked receptor=%s chunks=%d chunk_size=%d timeout=%ds",
            receptor_name,
            len(chunks),
            chunk_size,
            _GPU_BATCH_TIMEOUT_SEC,
        )

        n_cached = 0
        n_chunk_fail_ligands = 0
        n_chunk_failures = 0

        for chunk_idx, chunk in enumerate(chunks, start=1):
            chunk_paths = [task[2] for _, task in chunk]
            chunk_ok = _run_gpu_batch(
                config["vina_program"],
                receptor,
                center_f,
                size_f,
                chunk_paths,
                str(out_dir),
                config["exhaustiveness"],
                config["num_modes"],
                seed,
                timeout=_GPU_BATCH_TIMEOUT_SEC,
            )

            if not chunk_ok:
                n_chunk_failures += 1
                n_chunk_fail_ligands += len(chunk)
                logger.warning(
                    "gpu_batch_dock: chunk %d/%d failed for %s (%d ligands) -> CPU fallback",
                    chunk_idx,
                    len(chunks),
                    receptor_name,
                    len(chunk),
                )
                continue

            for _, (smi, _, fpath) in chunk:
                stem = Path(fpath).stem
                out_path = out_dir / f"{stem}_out.pdbqt"
                parsed = _parse_result_from_pdbqt(out_path)
                if parsed is not None:
                    _gpu_dock_cache[(receptor, smi)] = parsed
                    n_cached += 1

        n_gpu_missing = len(valid_items) - n_cached
        if _gpu_batch_log_count < 5:
            _gpu_batch_log_count += 1
            logger.info(
                "gpu_batch_dock: %d/%d ligands scored (GPU), %d unresolved -> CPU fallback, receptor %s",
                n_cached,
                len(tasks),
                n_gpu_missing,
                receptor_name,
            )

        if n_chunk_failures > 0:
            logger.warning(
                "gpu_batch_dock: %d/%d chunks failed (%d ligands) for %s",
                n_chunk_failures,
                len(chunks),
                n_chunk_fail_ligands,
                receptor_name,
            )
        elif n_gpu_missing > 0:
            logger.warning(
                "gpu_batch_dock: %d/%d unresolved scores for %s",
                n_gpu_missing,
                len(valid_items),
                receptor_name,
            )


__all__ = [
    "_GPU_BATCH_ENABLED",
    "_cleanup_gpu_cache_entries",
    "_gpu_dock_cache",
    "_precompute_gpu_batch_for_receptor",
]
