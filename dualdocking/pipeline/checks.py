"""Preflight checks for pipeline stage dependencies.

This module validates that required inputs exist before executing stages,
providing clear error messages about missing dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dualdocking.pipeline.config import StageTask


def check_fetch_outputs(project_root: Path) -> list[str]:
    """Check that fetch stage outputs exist.

    Args:
        project_root: Project root directory.

    Returns:
        List of missing filesystem paths.
    """
    missing: list[str] = []
    targets_dir = project_root / "output" / "data" / "targets"
    if not targets_dir.exists():
        missing.append(str(targets_dir))
    cal_dir = project_root / "output" / "data" / "calibration"
    if not cal_dir.exists():
        missing.append(str(cal_dir))
    return missing


def check_calibration_outputs(project_root: Path) -> list[str]:
    """Check that calibration outputs exist.

    Args:
        project_root: Project root directory.

    Returns:
        List of missing filesystem paths.
    """
    missing: list[str] = []
    norm = project_root / "configs" / "normalization.yaml"
    if not norm.exists():
        missing.append(str(norm))
    scores_dir = project_root / "output" / "data" / "calibration" / "scores"
    if not scores_dir.exists():
        missing.append(str(scores_dir))
    return missing


def check_generation_outputs(task: StageTask, project_root: Path) -> list[str]:
    """Check that generation outputs exist for a specific task.

    Args:
        task: Pipeline task to check.
        project_root: Project root directory.

    Returns:
        List of missing filesystem paths.
    """
    from dualdocking.data.io_utils import stem_exists

    missing: list[str] = []
    run_dir = task.run_dir(project_root)
    if not stem_exists(run_dir / "generated"):
        missing.append(f"{run_dir}/generated.parquet")
    return missing


def check_score_outputs(task: StageTask, project_root: Path) -> list[str]:
    """Check that scoring outputs exist for a specific task.

    Args:
        task: Pipeline task to check.
        project_root: Project root directory.

    Returns:
        List of missing filesystem paths.
    """
    from dualdocking.data.io_utils import stem_exists

    missing: list[str] = []
    run_dir = task.run_dir(project_root)
    if not stem_exists(run_dir / "scored_joined"):
        missing.append(f"{run_dir}/scored_joined.parquet")
    return missing


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
    missing: list[str] = []
    figs = project_root / "report" / "figures"
    if not figs.exists():
        missing.append(str(figs))
    return missing


def check_score_prerequisites(tasks: list[StageTask], project_root: Path) -> list[str]:
    """Check that generated files exist for all scoring tasks.

    Args:
        tasks: List of pipeline tasks to validate.
        project_root: Project root directory.

    Returns:
        List of missing generation output paths.
    """
    missing: list[str] = []
    for task in tasks:
        missing.extend(check_generation_outputs(task, project_root))
    return missing


def check_analyze_prerequisites(tasks: list[StageTask], project_root: Path) -> list[str]:
    """Check that scored files exist for all analysis tasks.

    Args:
        tasks: List of pipeline tasks to validate.
        project_root: Project root directory.

    Returns:
        List of missing scored output paths.
    """
    missing: list[str] = []
    for task in tasks:
        missing.extend(check_score_outputs(task, project_root))
    return missing


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
