"""Pipeline stage definitions and execution logic.

This module defines all pipeline stages (fetch, calibrate, generate, score, etc.)
and provides functions to execute them.

It also serves as the package for sub-stage orchestrators (``fetch``, ``prepare``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dualdocking.pipeline.config import LatexModeType, StageTask


@dataclass(frozen=True)
class StageDefinition:
    """Metadata for a pipeline stage.

    Attributes:
        name: Stage identifier (e.g., ``"fetch"``).
        description: Human-readable description.
        dependencies: List of stage names that must complete first.
    """

    name: str
    description: str
    dependencies: list[str]


# All available pipeline stages in execution order
STAGE_DEFINITIONS: list[StageDefinition] = [
    StageDefinition("fetch", "Download PDB, FASTA, SMILES, Docker image", []),
    StageDefinition("calibrate", "Dock calibration set, compute normalization", ["fetch"]),
    StageDefinition("generate", "Run FREED++ molecule generation", ["calibrate"]),
    StageDefinition("score", "Score molecules with Boltz-2", ["generate"]),
    StageDefinition("analyze", "Aggregate results, compute metrics, sensitivity, correlation", ["score"]),
    StageDefinition("report", "Generate plots, render molecules, LaTeX report", ["analyze"]),
    StageDefinition("audit", "Run claim traceability audit", ["analyze"]),
    StageDefinition("gnina", "Cross-validate with GNINA CNN scoring", ["score"]),
    StageDefinition("md", "MD validation (GROMACS 50 ns) for top candidates", ["score"]),
]

STAGE_ORDER: list[str] = [s.name for s in STAGE_DEFINITIONS]
STAGE_DEPS: dict[str, list[str]] = {s.name: s.dependencies for s in STAGE_DEFINITIONS}
STAGE_DESCRIPTIONS: dict[str, str] = {s.name: s.description for s in STAGE_DEFINITIONS}


def _build_task_command(script_path: str, task: StageTask) -> list[str]:
    """Build a standard ``uv run python`` command for a (pair, mode, seed) task."""
    return [
        "uv",
        "run",
        "python",
        script_path,
        "--pair",
        task.pair,
        "--mode",
        task.mode,
        "--seed",
        str(task.seed),
    ]


def build_fetch_command() -> list[str]:
    """Build command to run the fetch stage.

    Returns:
        Command tokens for subprocess execution.
    """
    return ["uv", "run", "python", "cli/s0_fetch/fetch_all.py"]


def build_calibrate_commands() -> list[list[str]]:
    """Build commands to run the calibration stage.

    Returns:
        List of command token lists.
    """
    return [
        ["uv", "run", "python", "cli/s2_calibrate/dock_calibration.py"],
        ["uv", "run", "python", "cli/s2_calibrate/compute_normalization.py"],
    ]


def build_generate_command(task: StageTask) -> list[str]:
    """Build command to run molecule generation for a task.

    Args:
        task: Pipeline task specifying pair, mode, and seed.

    Returns:
        Command tokens for subprocess execution.
    """
    return _build_task_command("cli/s3_generate/generate.py", task)


def build_score_command(task: StageTask, molecules: str | None = None) -> list[str]:
    """Build command to run scoring for a task.

    Args:
        task: Pipeline task specifying pair, mode, and seed.
        molecules: Optional path to external molecule file.

    Returns:
        Command tokens for subprocess execution.
    """
    cmd = _build_task_command("cli/s4_score/score.py", task)
    if molecules:
        cmd.extend(["--molecules", molecules])
    return cmd


def _pair_args(pairs: list[str] | None) -> list[str]:
    """Build ``--pair X --pair Y`` CLI fragment from pair filter list."""
    if not pairs:
        return []
    args: list[str] = []
    for p in pairs:
        args.extend(["--pair", p])
    return args


def build_analyze_commands(
    pairs: list[str] | None = None,
    seeds: list[int] | None = None,
) -> list[list[str]]:
    """Build commands to run the analysis stage.

    Args:
        pairs: Optional pair filter list. When set, only these pairs are analyzed.
        seeds: Optional seed filter list. When set, only these seeds are analyzed.

    Returns:
        List of command token lists for aggregate, metrics, sensitivity,
        correlation, and summarize cli.
    """
    pa = _pair_args(pairs)
    sa = ["--seeds", ",".join(str(s) for s in seeds)] if seeds else []
    return [
        ["uv", "run", "python", "cli/s5_analyze/aggregate.py", *pa, *sa],
        ["uv", "run", "python", "cli/s5_analyze/compute_metrics.py", *pa, *sa],
        ["uv", "run", "python", "cli/s5_analyze/threshold_sensitivity.py", *pa, *sa],
        ["uv", "run", "python", "cli/s5_analyze/proxy_correlation.py", *pa, *sa],
        ["uv", "run", "python", "cli/s5_analyze/summarize.py", *pa],
    ]


def build_report_commands(latex_mode: LatexModeType = "off") -> list[list[str]]:
    """Build commands to run the report stage.

    Args:
        latex_mode: LaTeX compilation mode (``"off"``, ``"draft"``, or ``"full"``).

    Returns:
        List of command token lists for plots, rendering, and optional LaTeX.
    """
    cmds = [
        ["uv", "run", "python", "cli/s6_report/plot_results.py"],
        ["uv", "run", "python", "cli/s6_report/render_molecules.py"],
    ]
    if latex_mode != "off":
        cmds.append(
            [
                "uv",
                "run",
                "python",
                "cli/s6_report/generate_latex_report.py",
                "--mode",
                latex_mode,
            ]
        )
    return cmds


def build_audit_command() -> list[str]:
    """Build command to run the audit stage.

    Returns:
        Command tokens for subprocess execution.
    """
    return ["uv", "run", "python", "cli/s7_audit/claim_audit.py"]


def build_gnina_command(task: StageTask) -> list[str]:
    """Build command to run GNINA cross-validation for a task.

    Args:
        task: Pipeline task specifying pair, mode, and seed.

    Returns:
        Command tokens for subprocess execution.
    """
    return _build_task_command("cli/s6_report/score_gnina.py", task)


def build_md_command(pair: str) -> list[str]:
    """Build command to run MD validation for a pair.

    Args:
        pair: Target pair identifier.

    Returns:
        Command tokens for subprocess execution.
    """
    return ["uv", "run", "python", "cli/s6_md/run_md.py", "--pair", pair]


def build_prepare_commands(
    prepare_freed_data: bool = True,
    prepare_baseline: bool = False,
    baseline_url: str | None = None,
) -> list[list[str]]:
    """Build data preparation commands.

    Args:
        prepare_freed_data: Whether to fetch FREED++ data dependencies.
        prepare_baseline: Whether to fetch the ZINC baseline archive.
        baseline_url: Optional custom URL for baseline archive.

    Returns:
        List containing zero or one command token lists.
    """
    if not prepare_freed_data and not prepare_baseline:
        return []

    cmd = ["uv", "run", "python", "cli/s1_prepare/prepare_all.py"]
    if not prepare_freed_data:
        cmd.append("--skip-fetch-freed-data")
    if prepare_baseline:
        cmd.append("--fetch-baseline")
        if baseline_url:
            cmd.extend(["--baseline-url", baseline_url])
    return [cmd]


def ensure_calibration(
    pair_id: str,
    pair_cfg: dict,
    project_root: Path,
    log_path: Path,
    run_cmd_fn: Callable[[list[str], Path], None],
) -> None:
    """Ensure both targets in a pair have calibrated normalization statistics.

    If any target is uncalibrated, runs the full calibration pipeline.

    Args:
        pair_id: Target pair identifier (e.g., ``"CDK2_ERK2"``).
        pair_cfg: Pair configuration dict with ``target_a`` and ``target_b``.
        project_root: Project root directory.
        log_path: Path to experiment log file.
        run_cmd_fn: Callable that executes commands (signature: ``(cmd, log_path)``).

    Raises:
        RuntimeError: If calibration fails or ``normalization.yaml`` is missing.
    """
    import yaml

    from dualdocking.pipeline.core import log_to_file, utc_timestamp

    norm_path = project_root / "configs" / "normalization.yaml"
    if not norm_path.exists():
        raise RuntimeError(f"normalization.yaml not found: {norm_path}")

    norm_data = yaml.safe_load(norm_path.read_text())
    norm_targets = norm_data.get("targets", {})

    uncalibrated: list[str] = []
    for role in ("target_a", "target_b"):
        tid = pair_cfg[role]["id"]
        entry = norm_targets.get(tid, {})
        if not entry or entry.get("n_samples", 0) == 0 or not entry.get("stable_mean") or not entry.get("stable_std"):
            uncalibrated.append(tid)

    if not uncalibrated:
        return

    log_to_file(log_path, f"[{utc_timestamp()}] CALIBRATE {pair_id}: uncalibrated targets = {uncalibrated}")

    # Step 1: prepare target data
    run_cmd_fn(
        ["uv", "run", "python", "cli/s1_prepare/prepare_targets.py", "--pair", pair_id, "--prepare-all"],
        log_path,
    )

    # Step 2: dock calibration for uncalibrated targets only
    dock_cmd = ["uv", "run", "python", "cli/s2_calibrate/dock_calibration.py"]
    for tid in uncalibrated:
        dock_cmd.extend(["--target-id", tid])
    run_cmd_fn(dock_cmd, log_path)

    # Step 3: regenerate normalization.yaml
    run_cmd_fn(["uv", "run", "python", "cli/s2_calibrate/compute_normalization.py"], log_path)

    # Verify calibration succeeded
    norm_data = yaml.safe_load(norm_path.read_text())
    norm_targets = norm_data.get("targets", {})
    still_bad: list[str] = []
    for tid in uncalibrated:
        entry = norm_targets.get(tid, {})
        if not entry or entry.get("n_samples", 0) == 0 or not entry.get("stable_mean") or not entry.get("stable_std"):
            still_bad.append(tid)

    if still_bad:
        raise RuntimeError(
            f"Calibration failed for targets {still_bad} in pair {pair_id}. "
            "Check dock_calibration.py output and data/calibration/scores/."
        )

    log_to_file(log_path, f"[{utc_timestamp()}] CALIBRATE {pair_id}: OK")
