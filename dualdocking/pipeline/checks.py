"""Preflight checks for pipeline stage dependencies.

This module validates that required inputs exist before executing stages,
providing clear error messages about missing dependencies.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dualdocking.pipeline.config import StageTask


def _missing_paths(paths: list[Path]) -> list[str]:
    """Return string paths for entries that do not exist."""
    return [str(path) for path in paths if not path.exists()]


def _dedupe(items: list[str]) -> list[str]:
    """Return ``items`` without duplicates while preserving order."""
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


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


def _check_pair_target_pdbs(tasks: list[StageTask], project_root: Path) -> list[str]:
    """Check that pair-specific receptor PDB files exist for all task pairs."""
    from dualdocking.config import load_pairs

    pairs_path = project_root / "configs" / "pairs.yaml"
    if not pairs_path.exists():
        return [str(pairs_path)]

    try:
        pairs_cfg = load_pairs(pairs_path)["pairs"]
    except Exception:
        return [str(pairs_path)]

    missing: list[str] = []
    for pair_id in sorted({task.pair for task in tasks}):
        pair_cfg = pairs_cfg.get(pair_id)
        if not isinstance(pair_cfg, dict):
            missing.append(f"{pairs_path}:{pair_id}")
            continue

        for role, label in (("target_a", "A"), ("target_b", "B")):
            target_cfg = pair_cfg.get(role, {})
            pdb_id = target_cfg.get("pdb_id", target_cfg.get("pdb")) if isinstance(target_cfg, dict) else None
            if not pdb_id:
                missing.append(f"{pairs_path}:{pair_id}.{role}.pdb_id")
                continue

            pdb_name = str(pdb_id)
            candidates = [pdb_name]
            if pdb_name.lower() != pdb_name:
                candidates.append(pdb_name.lower())

            paths = [project_root / "output" / "data" / "targets" / pair_id / label / f"{name}.pdb" for name in candidates]
            if not any(path.exists() for path in paths):
                missing.append(str(paths[0]))

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
            project_root / "output" / "data" / "calibration",
        ]
    )


def check_prepare_outputs(project_root: Path) -> list[str]:
    """Check that prepare stage outputs exist.

    Args:
        project_root: Project root directory.

    Returns:
        List of missing filesystem paths.
    """
    return _missing_paths(
        [
            project_root / "output" / "data" / "targets",
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


def check_gnina_prerequisites(tasks: list[StageTask], project_root: Path) -> list[str]:
    """Check that GNINA stage prerequisites are available.

    Requires scored task outputs, pair receptor PDBs, and a GNINA binary
    either at ``output/bin/gnina`` or on ``PATH``.
    """
    missing = check_analyze_prerequisites(tasks, project_root)
    missing.extend(_check_pair_target_pdbs(tasks, project_root))

    gnina_local = project_root / "output" / "bin" / "gnina"
    gnina_local_ok = gnina_local.exists() and bool(gnina_local.stat().st_mode & 0o111)
    if not gnina_local_ok and not shutil.which("gnina"):
        missing.append(str(gnina_local))

    return _dedupe(missing)


def check_md_prerequisites(tasks: list[StageTask], project_root: Path) -> list[str]:
    """Check that MD stage prerequisites are available.

    Requires scored task outputs and pair receptor PDB files.
    """
    missing = check_analyze_prerequisites(tasks, project_root)
    missing.extend(_check_pair_target_pdbs(tasks, project_root))
    return _dedupe(missing)


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
    stage_checkers: dict[str, Callable[[Path], list[str]]] = {
        "prepare": check_fetch_outputs,
        "calibrate": check_prepare_outputs,
        "generate": check_calibration_outputs,
        "report": check_analyze_outputs,
        "audit": check_analyze_outputs,
    }
    task_checkers: dict[str, Callable[[list[StageTask], Path], list[str]]] = {
        "score": check_score_prerequisites,
        "analyze": check_analyze_prerequisites,
        "gnina": check_gnina_prerequisites,
        "md": check_md_prerequisites,
    }

    if stage == "fetch":
        return []

    task_checker = task_checkers.get(stage)
    if task_checker is not None:
        return [] if tasks is None else task_checker(tasks, project_root)

    checker = stage_checkers.get(stage)
    return [] if checker is None else checker(project_root)
