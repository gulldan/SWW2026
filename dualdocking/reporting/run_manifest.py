"""Run manifest -- lifecycle tracking for experiment runs.

Records the full provenance chain of an experiment execution:
environment, config hashes, per-task status, and timing.

File format: JSON at ``results/run_manifest_{run_id}.json``.

Writes use atomic temp-file + ``os.replace`` (same pattern as :mod:`cache_io`)
with ``fcntl.flock`` for concurrent safety.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts() -> str:
    """Return an ISO 8601 timestamp in UTC with Z suffix.

    Returns:
        Timestamp string, e.g. ``"2026-02-10T12:34:56.789Z"``.
    """
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    """Return hex SHA-256 of *path*, or empty string if the file is missing.

    Args:
        path: File to hash.

    Returns:
        Hex digest, or ``""`` if *path* does not exist.
    """
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_info() -> dict[str, Any]:
    """Collect current git commit hash and dirty status.

    Returns:
        Dict with ``"commit"`` (short hash or ``"unknown"``) and
        ``"dirty"`` (bool).
    """
    info: dict[str, Any] = {"commit": "unknown", "dirty": False}
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            info["commit"] = out.stdout.strip().splitlines()[0]
    except FileNotFoundError:
        pass
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0:
            info["dirty"] = bool(out.stdout.strip())
    except FileNotFoundError:
        pass
    return info


# ---------------------------------------------------------------------------
# Atomic JSON I/O (mirrors cache_io.py pattern)
# ---------------------------------------------------------------------------

_LOCK_TIMEOUT: float = 10.0
_RETRY_INITIAL: float = 0.05
_RETRY_CAP: float = 1.0


def _acquire_flock(fd: int, timeout: float) -> bool:
    """Try to acquire an exclusive flock with exponential backoff.

    Args:
        fd: Open file descriptor for the lock file.
        timeout: Maximum seconds to wait for the lock.

    Returns:
        ``True`` if the lock was acquired, ``False`` on timeout.
    """
    deadline = time.monotonic() + timeout
    delay = _RETRY_INITIAL
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (OSError, BlockingIOError):
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(delay, deadline - time.monotonic()))
            delay = min(delay * 2, _RETRY_CAP)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write *data* as JSON atomically (temp + fsync + replace under flock).

    Args:
        path: Destination file path.
        data: JSON-serializable dictionary to persist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    lock_path.touch(exist_ok=True)
    lock_fd: int | None = None
    tmp_path: str | None = None
    try:
        lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        if not _acquire_flock(lock_fd, _LOCK_TIMEOUT):
            logger.warning("Lock timeout writing manifest: {}", path)
            return
        payload = json.dumps(data, indent=2, ensure_ascii=False).encode()
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{path.stem}.",
            suffix=".tmp",
        )
        os.write(fd, payload)
        os.fsync(fd)
        os.close(fd)
        Path(tmp_path).replace(path)
        tmp_path = None
    except OSError as exc:
        logger.error("Failed to write manifest {}: {}", path, exc)
    finally:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
        if lock_fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)


def _read_json(path: Path) -> dict[str, Any]:
    """Read JSON from *path* under shared flock.

    Args:
        path: JSON file to read.

    Returns:
        Parsed dict, or ``{}`` on any error (missing file, corrupt JSON, lock timeout).
    """
    if not path.exists():
        return {}
    lock_path = path.with_suffix(".lock")
    lock_path.touch(exist_ok=True)
    lock_fd: int | None = None
    try:
        lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        if not _acquire_flock(lock_fd, _LOCK_TIMEOUT):
            logger.warning("Lock timeout reading manifest: {}", path)
            return {}
        raw = path.read_text()
        if not raw.strip():
            return {}
        return json.loads(raw)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Corrupted manifest {}: {}", path, exc)
        return {}
    finally:
        if lock_fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest_path(results_dir: Path, run_id: str) -> Path:
    """Canonical path for a run manifest.

    Args:
        results_dir: Base results directory.
        run_id: Unique run identifier.

    Returns:
        Path of the form ``{results_dir}/run_manifest_{run_id}.json``.
    """
    return results_dir / f"run_manifest_{run_id}.json"


def init_manifest(
    path: Path,
    *,
    run_id: str,
    config_dir: Path,
    tasks: list[tuple[str, str, int]],
    parallel: int = 1,
    resume: bool = False,
) -> dict[str, Any]:
    """Create a new manifest and persist it.

    Args:
        path: Output manifest file path.
        run_id: Unique run identifier (e.g. ``EXPERIMENT_RUN_ID``).
        config_dir: Directory containing ``pairs.yaml``, ``experiment.yaml``, etc.
        tasks: List of ``(pair, mode, seed)`` tuples.
        parallel: Number of parallel workers.
        resume: Whether resume mode is active.

    Returns:
        The manifest document (also written to *path*).
    """
    git = _git_info()
    config_hashes = {
        name: _sha256_file(config_dir / name) for name in ("pairs.yaml", "experiment.yaml", "metrics.yaml", "normalization.yaml")
    }

    task_entries: list[dict[str, Any]] = [
        {
            "task_id": f"{pair}/{mode}/seed_{seed}",
            "pair": pair,
            "mode": mode,
            "seed": seed,
            "status": "pending",
            "stages": {},
        }
        for pair, mode, seed in tasks
    ]

    doc: dict[str, Any] = {
        "manifest_version": "1.0.0",
        "run_id": run_id,
        "started_at": _ts(),
        "completed_at": None,
        "status": "running",
        "environment": {
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
        },
        "config_hashes": config_hashes,
        "execution": {
            "parallel": parallel,
            "resume": resume,
        },
        "tasks": task_entries,
        "summary": {
            "total": len(task_entries),
            "completed": 0,
            "failed": 0,
            "skipped": 0,
        },
    }
    _atomic_write_json(path, doc)
    logger.info("Manifest initialized: {} ({} tasks)", path, len(task_entries))
    return doc


def update_task(
    path: Path,
    *,
    task_id: str,
    stage: str,
    status: str,
    duration_seconds: float | None = None,
    error: str | None = None,
) -> None:
    """Update a single task's stage status in the manifest.

    Args:
        path: Manifest file path.
        task_id: Task identifier, e.g. ``"CDK2_ERK2/single_a/seed_42"``.
        stage: Lifecycle stage: ``"generate"`` or ``"score"``.
        status: ``"running"``, ``"completed"``, ``"failed"``, or ``"skipped"``.
        duration_seconds: Wall-clock time for this stage (set on completion).
        error: Error message (set on failure, truncated to 2000 chars).
    """
    doc = _read_json(path)
    if not doc:
        logger.warning("Cannot update task — manifest not found: {}", path)
        return

    for task in doc.get("tasks", []):
        if task["task_id"] == task_id:
            task["stages"][stage] = {
                "status": status,
                "timestamp": _ts(),
            }
            if duration_seconds is not None:
                task["stages"][stage]["duration_seconds"] = round(duration_seconds, 1)
            if error is not None:
                task["stages"][stage]["error"] = error[:2000]

            # Derive task-level status from stages
            statuses = [s["status"] for s in task["stages"].values()]
            if any(s == "failed" for s in statuses):
                task["status"] = "failed"
            elif any(s == "running" for s in statuses):
                task["status"] = "running"
            elif all(s in ("completed", "skipped") for s in statuses) and len(statuses) >= 2:
                task["status"] = "completed"
            elif status == "skipped" and stage == "generate" and len(statuses) == 1:
                # generate skipped in resume mode -- task still in progress until score completes
                task["status"] = "running"
            else:
                task["status"] = "running"
            break

    # Recompute summary
    tasks_list = doc.get("tasks", [])
    doc["summary"] = {
        "total": len(tasks_list),
        "completed": sum(1 for t in tasks_list if t["status"] == "completed"),
        "failed": sum(1 for t in tasks_list if t["status"] == "failed"),
        "skipped": sum(1 for t in tasks_list if t["status"] == "skipped"),
    }

    _atomic_write_json(path, doc)


def skip_task(path: Path, *, task_id: str) -> None:
    """Mark an entire task as skipped (already completed in a previous run).

    Args:
        path: Manifest file path.
        task_id: Task identifier to mark as skipped.
    """
    doc = _read_json(path)
    if not doc:
        return
    for task in doc.get("tasks", []):
        if task["task_id"] == task_id:
            task["status"] = "skipped"
            task["stages"]["generate"] = {"status": "skipped", "timestamp": _ts()}
            task["stages"]["score"] = {"status": "skipped", "timestamp": _ts()}
            break

    tasks_list = doc.get("tasks", [])
    doc["summary"] = {
        "total": len(tasks_list),
        "completed": sum(1 for t in tasks_list if t["status"] == "completed"),
        "failed": sum(1 for t in tasks_list if t["status"] == "failed"),
        "skipped": sum(1 for t in tasks_list if t["status"] == "skipped"),
    }
    _atomic_write_json(path, doc)


def finalize_manifest(path: Path, *, status: str = "completed") -> None:
    """Mark the entire run as finished.

    Args:
        path: Manifest file path.
        status: Final run status: ``"completed"``, ``"failed"``, or ``"partial"``.
    """
    doc = _read_json(path)
    if not doc:
        logger.warning("Cannot finalize — manifest not found: {}", path)
        return

    doc["completed_at"] = _ts()
    doc["status"] = status

    # Compute total duration
    started = doc.get("started_at")
    if started:
        try:
            t0 = datetime.fromisoformat(started)
            t1 = datetime.now(UTC)
            doc["total_duration_seconds"] = round((t1 - t0).total_seconds(), 1)
        except (ValueError, TypeError):
            pass

    _atomic_write_json(path, doc)
    logger.info("Manifest finalized: {} (status={})", path, status)


def load_manifest(path: Path) -> dict[str, Any]:
    """Load an existing manifest from disk.

    Args:
        path: Manifest file path.

    Returns:
        Manifest dict, or ``{}`` if the file doesn't exist or is corrupted.
    """
    return _read_json(path)
