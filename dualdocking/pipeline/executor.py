"""Parallel task execution with GPU scheduling.

This module provides parallel execution of pipeline tasks with automatic
GPU assignment and thread pooling.
"""

from __future__ import annotations

import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from loguru import logger
from tqdm import tqdm

if TYPE_CHECKING:
    from dualdocking.pipeline.config import StageTask


class TaskExecutor:
    """Parallel task executor with GPU pool management.

    Attributes:
        n_gpus: Number of available GPUs.
        max_workers: Maximum number of concurrent tasks.
        gpu_pool: Queue of available GPU IDs.
    """

    def __init__(self, n_gpus: int, max_workers: int | None = None) -> None:
        """Initialize the task executor.

        Args:
            n_gpus: Number of available GPUs.
            max_workers: Maximum concurrent tasks (defaults to *n_gpus*).
        """
        self.n_gpus = n_gpus
        self.max_workers = min(max_workers or n_gpus, n_gpus)

        # GPU pool: each task acquires a GPU ID before running
        self.gpu_pool: queue.Queue[int] = queue.Queue()
        for g in range(n_gpus):
            self.gpu_pool.put(g)

        if self.max_workers < (max_workers or n_gpus):
            logger.warning(
                "Capping max_workers {} to {} (available GPUs). Cannot run more concurrent tasks than GPUs.",
                max_workers,
                self.max_workers,
            )

    def run_tasks_sequential(
        self,
        tasks: list[StageTask],
        task_fn: Callable[[StageTask, int | None], None],
    ) -> tuple[int, int]:
        """Execute tasks sequentially (single-threaded).

        Args:
            tasks: List of pipeline tasks to execute.
            task_fn: Callable that executes a task, signature: ``(task, gpu_id)``.

        Returns:
            Tuple of ``(n_completed, n_failed)``.
        """
        n_completed = 0
        n_failed = 0

        pbar = tqdm(tasks, desc="Tasks", unit="task", dynamic_ncols=True)
        for task in pbar:
            pbar.set_postfix_str(str(task), refresh=True)
            try:
                task_fn(task, 0)  # Always use GPU 0 in sequential mode
                n_completed += 1
            except RuntimeError:
                n_failed += 1
        pbar.set_postfix_str(f"done={n_completed} failed={n_failed}")
        pbar.close()

        return n_completed, n_failed

    def run_tasks_parallel(
        self,
        tasks: list[StageTask],
        task_fn: Callable[[StageTask, int | None], None],
    ) -> tuple[int, int]:
        """Execute tasks in parallel with GPU pool scheduling.

        Args:
            tasks: List of pipeline tasks to execute.
            task_fn: Callable that executes a task, signature: ``(task, gpu_id)``.

        Returns:
            Tuple of ``(n_completed, n_failed)``.
        """
        n_completed = 0
        n_failed = 0

        def _run_with_gpu_pool(task: StageTask) -> None:
            """Wrapper that acquires GPU from pool, runs task, releases GPU."""
            gid = self.gpu_pool.get()
            try:
                task_fn(task, gid)
            finally:
                self.gpu_pool.put(gid)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_run_with_gpu_pool, task): str(task) for task in tasks}
            with tqdm(total=len(tasks), desc="Tasks", unit="task", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()
                        n_completed += 1
                    except RuntimeError:
                        n_failed += 1
                    pbar.update(1)
                    pbar.set_postfix_str(f"done={n_completed} failed={n_failed}")

        return n_completed, n_failed

    def execute(
        self,
        tasks: list[StageTask],
        task_fn: Callable[[StageTask, int | None], None],
    ) -> tuple[int, int]:
        """Execute tasks using appropriate strategy (sequential or parallel).

        Args:
            tasks: List of pipeline tasks to execute.
            task_fn: Callable that executes a task, signature: ``(task, gpu_id)``.

        Returns:
            Tuple of ``(n_completed, n_failed)``.
        """
        if self.max_workers <= 1:
            return self.run_tasks_sequential(tasks, task_fn)
        return self.run_tasks_parallel(tasks, task_fn)


def _read_generator_backend(run_dir: Path) -> str | None:
    """Read ``generator_backend`` from *run_dir*/metadata.json.

    Returns:
        Backend name string (e.g. ``"freedpp"``, ``"reinvent4"``) or
        ``None`` if the file is missing or unreadable.
    """
    import json

    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text())
        return data.get("generator_backend")
    except Exception:
        return None


def _current_generator_backend(project_root: Path) -> str:
    """Read the configured generator backend from experiment.yaml.

    Returns:
        Canonical backend name (e.g. ``"freedpp"``, ``"reinvent4"``).
    """
    import yaml

    from dualdocking.utils.naming import canonicalize_name

    exp_path = project_root / "configs" / "experiment.yaml"
    if not exp_path.exists():
        return "freedpp"
    try:
        data = yaml.safe_load(exp_path.read_text())
        gen = data.get("generator", {})
        raw = gen.get("backend", gen.get("name", "freedpp"))
        return canonicalize_name(
            str(raw).strip(),
            {"freed": "freedpp", "freedpp": "freedpp", "reinvent": "reinvent4", "reinvent4": "reinvent4"},
        )
    except Exception:
        return "freedpp"


def run_single_task(
    task: StageTask,
    project_root: Path,
    log_path: Path,
    resume: bool,
    manifest_path: Path | None,
    gpu_id: int | None,
    generate_cmd_fn: Callable[[StageTask], list[str]],
    score_cmd_fn: Callable[[StageTask], list[str]],
    run_cmd_fn: Callable[[list[str], Path, int | None], None],
) -> None:
    """Execute generate + score pipeline for a single task.

    Skips entirely if ``scored_joined`` exists **and** was generated by the
    current backend. In resume mode, also skips generation if ``generated``
    exists and was produced by the same backend.

    Args:
        task: Pipeline task to execute.
        project_root: Project root directory.
        log_path: Path to experiment log file.
        resume: Whether to skip generation if ``generated`` exists.
        manifest_path: Path to run manifest file, or ``None`` to skip tracking.
        gpu_id: CUDA device index, or ``None`` for default.
        generate_cmd_fn: Callable that builds the generate command.
        score_cmd_fn: Callable that builds the score command.
        run_cmd_fn: Callable that executes a command with signature:
            ``(cmd, log_path, gpu_id)``.

    Raises:
        RuntimeError: If either generate or score subprocess fails.
    """
    from dualdocking.data.io_utils import stem_exists
    from dualdocking.reporting.run_manifest import skip_task, update_task

    run_dir = task.run_dir(project_root)
    scored = run_dir / "scored_joined"
    generated = run_dir / "generated"
    tid = str(task)

    # Check whether existing results were produced by the current backend.
    # If generated data exists but metadata is missing / unreadable, treat
    # it as a mismatch to avoid silently reusing stale data from a different
    # generator.
    current_backend = _current_generator_backend(project_root)
    existing_backend = _read_generator_backend(run_dir)
    has_generated = stem_exists(generated)
    if existing_backend is not None:
        backend_match = existing_backend == current_backend
    elif has_generated:
        # generated data exists but no metadata → cannot verify backend → mismatch
        backend_match = False
    else:
        # no generated data at all → nothing to mismatch
        backend_match = True

    # Skip if already scored — but only if same backend produced the data.
    if stem_exists(scored) and backend_match:
        logger.info("[SKIP] {} (scored_joined exists)", task)
        if manifest_path:
            skip_task(manifest_path, task_id=tid)
        return

    if not backend_match:
        logger.info("[BACKEND MISMATCH] {} (existing={}, current={}) — regenerating", task, existing_backend, current_backend)

    gpu_tag = f" [GPU {gpu_id}]" if gpu_id is not None else ""

    # --- Generate stage ---
    if stem_exists(generated) and resume and backend_match:
        logger.info("[RESUME] {} (generated exists)", task)
        if manifest_path:
            update_task(manifest_path, task_id=tid, stage="generate", status="skipped")
    else:
        logger.info("[GENERATE]{} {} — started", gpu_tag, task)
        if manifest_path:
            update_task(manifest_path, task_id=tid, stage="generate", status="running")
        t0 = time.monotonic()
        try:
            run_cmd_fn(generate_cmd_fn(task), log_path, gpu_id)
        except RuntimeError as exc:
            elapsed = time.monotonic() - t0
            logger.error("[GENERATE]{} {} — FAILED ({:.0f}s)", gpu_tag, task, elapsed)
            if manifest_path:
                update_task(manifest_path, task_id=tid, stage="generate", status="failed", error=str(exc))
            raise
        elapsed = time.monotonic() - t0
        logger.info("[GENERATE]{} {} — done ({:.0f}s)", gpu_tag, task, elapsed)
        if manifest_path:
            update_task(manifest_path, task_id=tid, stage="generate", status="completed", duration_seconds=elapsed)

    # --- Score stage ---
    logger.info("[SCORE]{} {} — started", gpu_tag, task)
    if manifest_path:
        update_task(manifest_path, task_id=tid, stage="score", status="running")
    t0 = time.monotonic()
    try:
        run_cmd_fn(score_cmd_fn(task), log_path, gpu_id)
    except RuntimeError as exc:
        elapsed = time.monotonic() - t0
        logger.error("[SCORE]{} {} — FAILED ({:.0f}s)", gpu_tag, task, elapsed)
        if manifest_path:
            update_task(manifest_path, task_id=tid, stage="score", status="failed", error=str(exc))
        raise
    elapsed = time.monotonic() - t0
    logger.info("[SCORE]{} {} — done ({:.0f}s)", gpu_tag, task, elapsed)
    if manifest_path:
        update_task(manifest_path, task_id=tid, stage="score", status="completed", duration_seconds=elapsed)
