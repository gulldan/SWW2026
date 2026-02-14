"""Preflight checks for pipeline stage dependencies.

This module validates that required inputs exist before executing stages,
providing clear error messages about missing dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dualdocking.pipeline.config import StageTask


def _missing_paths(paths: list[Path]) -> list[str]:
    """Return string paths for entries that do not exist."""
    return [str(path) for path in paths if not path.exists()]


def _check_task_stem_output(task: StageTask, project_root: Path, stem_name: str) -> list[str]:
    """Check that a task run directory contains a specific stem output."""
    from dualdocking.data.io_utils import stem_exists

    run_dir = task.run_dir(project_root)
    if stem_exists(run_dir / stem_name):
        return []
    return [f"{run_dir}/{stem_name}.parquet"]


def _collect_missing_for_tasks(
    tasks: list[StageTask],
    project_root: Path,
    checker: Callable[[StageTask, Path], list[str]],
) -> list[str]:
    """Collect missing outputs for all tasks using a checker callback."""
    missing: list[str] = []
    for task in tasks:
        missing.extend(checker(task, project_root))
    return missing


def check_fetch_outputs(project_root: Path) -> list[str]:
    """Check that fetch stage outputs exist.

    Args:
        project_root: Project root directory.

    Returns:
        List of missing filesystem paths.
    """
    return _missing_paths(
        [
            project_root / "output" / "data" / "targets",
            project_root / "output" / "data" / "calibration",
        ]
    )


def check_calibration_outputs(project_root: Path) -> list[str]:
    """Check that calibration outputs exist.

    Args:
        project_root: Project root directory.

    Returns:
        List of missing filesystem paths.
    """
    return _missing_paths(
        [
            project_root / "configs" / "normalization.yaml",
            project_root / "output" / "data" / "calibration" / "scores",
        ]
    )


def check_generation_outputs(task: StageTask, project_root: Path) -> list[str]:
    """Check that generation outputs exist for a specific task.

    Args:
        task: Pipeline task to check.
        project_root: Project root directory.

    Returns:
        List of missing filesystem paths.
    """
    return _check_task_stem_output(task, project_root, "generated")


def check_score_outputs(task: StageTask, project_root: Path) -> list[str]:
    """Check that scoring outputs exist for a specific task.

    Args:
        task: Pipeline task to check.
        project_root: Project root directory.

    Returns:
        List of missing filesystem paths.
    """
    return _check_task_stem_output(task, project_root, "scored_joined")


def check_analyze_outputs(project_root: Path) -> list[str]:
    """Check that analysis outputs exist.

    Args:
        project_root: Project root directory.

    Returns:
        List of missing filesystem paths.
    """
    missing: list[str] = []
    report_dir = project_root / "report"

    data_file = report_dir / "data" / "all_results.parquet"
    if not data_file.exists():
        missing.append(str(data_file))

    for name in ("metrics.json", "stats.json"):
        p = report_dir / name
        if not p.exists():
            missing.append(str(p))

    return missing


def check_report_outputs(project_root: Path) -> list[str]:
    """Check that report outputs exist.

    Args:
        project_root: Project root directory.

    Returns:
        List of missing filesystem paths.
    """
    return _missing_paths([project_root / "report" / "figures"])


def check_score_prerequisites(tasks: list[StageTask], project_root: Path) -> list[str]:
    """Check that generated files exist for all scoring tasks.

    Args:
        tasks: List of pipeline tasks to validate.
        project_root: Project root directory.

    Returns:
        List of missing generation output paths.
    """
    return _collect_missing_for_tasks(tasks, project_root, check_generation_outputs)


def check_analyze_prerequisites(tasks: list[StageTask], project_root: Path) -> list[str]:
    """Check that scored files exist for all analysis tasks.

    Args:
        tasks: List of pipeline tasks to validate.
        project_root: Project root directory.

    Returns:
        List of missing scored output paths.
    """
    return _collect_missing_for_tasks(tasks, project_root, check_score_outputs)


def preflight_check(
    stage: str,
    project_root: Path,
    tasks: list[StageTask] | None = None,
) -> list[str]:
    """Run preflight checks for a given stage.

    Args:
        stage: Stage name (e.g., ``"fetch"``, ``"calibrate"``, ``"generate"``).
        project_root: Project root directory.
        tasks: Optional list of tasks for task-specific stages.

    Returns:
        List of missing prerequisite paths. Empty if all checks pass.
    """
    if stage == "fetch":
        return []  # No prerequisites
    if stage == "calibrate":
        return check_fetch_outputs(project_root)
    if stage == "generate":
        return check_calibration_outputs(project_root)
    if stage == "score":
        if tasks is None:
            return []
        return check_score_prerequisites(tasks, project_root)
    if stage == "analyze":
        if tasks is None:
            return []
        return check_analyze_prerequisites(tasks, project_root)
    if stage == "report":
        return check_analyze_outputs(project_root)
    if stage == "audit":
        return check_analyze_outputs(project_root)
    return []
