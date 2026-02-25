"""Selective stage runner for the DualDocking pipeline.

Thin CLI wrapper around ``dualdocking.pipeline.PipelineRunner`` for running
individual pipeline stages or subsets, with preflight checks for required inputs.

Usage::

    # List available stages
    uv run python cli/pipeline/run_stages.py --list-stages

    # Run specific stages
    uv run python cli/pipeline/run_stages.py --stages analyze,report
    uv run python cli/pipeline/run_stages.py --stages all

    # Run with filters
    uv run python cli/pipeline/run_stages.py --stages score --pair CDK2_ERK2 --seeds 42
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import click
from loguru import logger

from dualdocking.pipeline import PipelineConfig, PipelineRunner, StageTask, checks, stages
from dualdocking.pipeline.core import detect_gpu_count
from dualdocking.pipeline.executor import TaskExecutor


@click.command()
@click.option("--stages", "stages_arg", default=None, help="Comma-separated stages to run (or 'all')")
@click.option("--list-stages", is_flag=True, help="List available stages and exit")
@click.option("--pair", default=None, help="Filter to a specific pair")
@click.option("--seeds", default=None, help="Comma-separated seed list")
@click.option("--molecules", default=None, type=click.Path(exists=True), help="Molecule file for scoring")
@click.option("--latex-mode", type=click.Choice(["off", "draft", "full"]), default="off")
@click.option("--parallel", default=0, type=int, help="Parallel tasks for GPU stages. 0 = auto-detect (# GPUs).")
@click.option("--resume", is_flag=True, default=False, help="Skip stage tasks when expected outputs already exist.")
@click.option("--dry-run", is_flag=True, help="Show what would be run without executing")
@click.option("--skip-preflight/--no-skip-preflight", default=False, help="Skip input checks")
def main(
    stages_arg: str | None,
    list_stages: bool,
    pair: str | None,
    seeds: str | None,
    molecules: str | None,
    latex_mode: str,
    parallel: int,
    resume: bool,
    dry_run: bool,
    skip_preflight: bool,
) -> None:
    """Run selected pipeline stages with preflight input validation.

    Each stage checks that its required inputs are available before
    executing, giving clear error messages about what to run first.

    Args:
        stages_arg: Comma-separated stage names to run, or ``"all"`` for every
            stage. ``None`` triggers an error with usage guidance.
        list_stages: If ``True``, print available stages and exit.
        pair: Optional pair identifier to restrict execution to a single pair.
        seeds: Comma-separated seed values. Defaults to the seeds defined in
            the experiment config.
        molecules: Optional filesystem path to a molecule file used by the
            scoring stage.
        latex_mode: LaTeX compilation mode (``"off"``, ``"draft"``, or
            ``"full"``).
        parallel: Number of concurrent task workers for task-based stages.
            ``0`` auto-detects from visible GPUs.
        resume: Skip per-task stage execution when expected outputs already
            exist.
        dry_run: If ``True``, log what would be run without executing.
        skip_preflight: If ``True``, skip prerequisite input checks before
            each stage.
    """
    if list_stages:
        click.echo("Available stages:\n")
        for stage_def in stages.STAGE_DEFINITIONS:
            deps = stage_def.dependencies
            dep_str = f" (requires: {', '.join(deps)})" if deps else ""
            click.echo(f"  {stage_def.name:<12} {stage_def.description}{dep_str}")
        return

    if stages_arg is None:
        click.echo("Error: --stages is required (use --list-stages to see options)")
        sys.exit(1)

    # Parse stage selection
    if stages_arg == "all":
        selected = list(stages.STAGE_ORDER)
    else:
        selected = [s.strip() for s in stages_arg.split(",")]
        for s in selected:
            if s not in stages.STAGE_ORDER:
                click.echo(f"Error: unknown stage '{s}'. Use --list-stages to see options.")
                sys.exit(1)

    # Load configuration
    config_dir = Path(__file__).resolve().parent.parent.parent / "configs"
    config = PipelineConfig.from_config_dir(config_dir)
    config.latex_mode = latex_mode  # type: ignore[assignment]

    # Parse filters
    seed_list = [int(s) for s in seeds.split(",")] if seeds else config.seeds
    pair_list = [pair] if pair else None

    # Generate tasks for stages that need them
    tasks = config.generate_tasks(pair_filter=pair_list, seed_filter=seed_list)

    if dry_run:
        logger.info("DRY RUN — Stages to run: {}", selected)
        logger.info("  Pair filter: {}", pair or "all")
        logger.info("  Seed filter: {}", seeds or "all")
        logger.info("  Total tasks (for generate/score): {}", len(tasks))
        logger.info("  Parallel workers: {}", _resolve_parallel(parallel))
        logger.info("  Resume mode: {}", resume)
        return

    # Create runner
    runner = PipelineRunner(config)

    # Execute each stage
    for stage in selected:
        logger.info("=== Stage: {} ===", stage)

        # Preflight checks
        if not skip_preflight:
            missing = checks.preflight_check(stage, config.project_root, tasks)
            if missing:
                logger.error("Preflight check failed for stage '{}'. Missing inputs:", stage)
                for m in missing[:10]:
                    logger.error("  - {}", m)
                if len(missing) > 10:
                    logger.error("  ... and {} more", len(missing) - 10)
                sys.exit(1)

        # Execute stage
        if stage == "fetch":
            runner.run_stage_fetch()
        elif stage == "prepare":
            runner.run_stage_prepare()
        elif stage == "calibrate":
            runner.run_stage_calibrate()
        elif stage == "generate":
            _run_generate_stage(
                runner,
                tasks,
                parallel=_resolve_parallel(parallel),
                resume=resume,
            )
        elif stage == "score":
            _run_score_stage(
                runner,
                tasks,
                molecules,
                parallel=_resolve_parallel(parallel),
                resume=resume,
            )
        elif stage == "analyze":
            runner.run_stage_analyze()
        elif stage == "report":
            runner.run_stage_report(latex_mode=latex_mode)  # type: ignore[arg-type]
        elif stage == "audit":
            runner.run_stage_audit()
        elif stage == "gnina":
            _run_gnina_stage(
                runner,
                tasks,
                parallel=_resolve_parallel(parallel),
                resume=resume,
            )
        elif stage == "md":
            _run_md_stage(runner, tasks)

        logger.info("=== Stage {} complete ===", stage)

    logger.info("All selected stages completed successfully")


def _resolve_parallel(parallel: int) -> int:
    """Resolve worker count from CLI value."""
    return detect_gpu_count() if parallel <= 0 else max(1, parallel)


def _task_has_stem_output(task: StageTask, project_root: Path, stem_name: str) -> bool:
    """Return whether a task run directory has a stem output (parquet/csv)."""
    from dualdocking.data.io_utils import stem_exists

    run_dir = task.run_dir(project_root)
    return stem_exists(run_dir / stem_name)


def _run_task_stage(
    runner: PipelineRunner,
    tasks: list[StageTask],
    *,
    stage_label: str,
    log_path: Path,
    parallel: int,
    build_cmd_fn: Callable[[StageTask], list[str]],
) -> None:
    """Run per-task stage commands sequentially or with GPU-aware parallelism."""
    if not tasks:
        logger.info("No tasks to run for stage '{}'", stage_label)
        return

    if parallel <= 1:
        for task in tasks:
            runner.run_cmd(build_cmd_fn(task), log_path, gpu_id=0)
        return

    n_gpus = detect_gpu_count()
    executor = TaskExecutor(n_gpus=n_gpus, max_workers=parallel)

    def _task_fn(task: StageTask, gpu_id: int | None) -> None:
        runner.run_cmd(build_cmd_fn(task), log_path, gpu_id=gpu_id)

    n_completed, n_failed = executor.execute(tasks, _task_fn)
    logger.info(
        "Stage '{}' task summary: {} completed, {} failed ({} total)",
        stage_label,
        n_completed,
        n_failed,
        len(tasks),
    )
    if n_failed > 0:
        msg = f"Stage '{stage_label}' failed for {n_failed} task(s)"
        raise RuntimeError(msg)


def _run_generate_stage(
    runner: PipelineRunner,
    tasks: list[StageTask],
    *,
    parallel: int,
    resume: bool,
) -> None:
    """Run generate stage for all tasks.

    Args:
        runner: Pipeline runner instance.
        tasks: List of tasks to execute.
        parallel: Number of concurrent workers.
        resume: Skip task if ``generated`` already exists.
    """
    log_path = runner.get_log_path("generate")
    run_tasks = tasks
    if resume:
        run_tasks = [task for task in tasks if not _task_has_stem_output(task, runner.config.project_root, "generated")]
        skipped = len(tasks) - len(run_tasks)
        if skipped > 0:
            logger.info("[RESUME] generate: skipped {} task(s) with existing outputs", skipped)

    _run_task_stage(
        runner,
        run_tasks,
        stage_label="generate",
        log_path=log_path,
        parallel=parallel,
        build_cmd_fn=stages.build_generate_command,
    )


def _run_score_stage(
    runner: PipelineRunner,
    tasks: list[StageTask],
    molecules: str | None,
    *,
    parallel: int,
    resume: bool,
) -> None:
    """Run score stage for all tasks.

    Args:
        runner: Pipeline runner instance.
        tasks: List of tasks to execute.
        molecules: Optional path to external molecule file.
        parallel: Number of concurrent workers.
        resume: Skip task if ``scored_joined`` already exists.
    """
    log_path = runner.get_log_path("score")
    run_tasks = tasks
    if resume and molecules is None:
        run_tasks = [task for task in tasks if not _task_has_stem_output(task, runner.config.project_root, "scored_joined")]
        skipped = len(tasks) - len(run_tasks)
        if skipped > 0:
            logger.info("[RESUME] score: skipped {} task(s) with existing outputs", skipped)
    elif resume and molecules is not None:
        logger.warning("--resume is ignored for score stage when --molecules override is set")

    _run_task_stage(
        runner,
        run_tasks,
        stage_label="score",
        log_path=log_path,
        parallel=parallel,
        build_cmd_fn=lambda task: stages.build_score_command(task, molecules=molecules),
    )


def _run_gnina_stage(
    runner: PipelineRunner,
    tasks: list[StageTask],
    *,
    parallel: int,
    resume: bool,
) -> None:
    """Run GNINA cross-validation for all tasks.

    Args:
        runner: Pipeline runner instance.
        tasks: List of tasks to execute.
        parallel: Number of concurrent workers.
        resume: Skip task if GNINA outputs already exist for both targets.
    """
    log_path = runner.get_log_path("gnina")
    run_tasks = tasks
    if resume:
        from dualdocking.data.io_utils import stem_exists

        def _has_gnina(task: StageTask) -> bool:
            run_dir = task.run_dir(runner.config.project_root)
            return stem_exists(run_dir / "gnina_scores_A") and stem_exists(run_dir / "gnina_scores_B")

        run_tasks = [task for task in tasks if not _has_gnina(task)]
        skipped = len(tasks) - len(run_tasks)
        if skipped > 0:
            logger.info("[RESUME] gnina: skipped {} task(s) with existing outputs", skipped)

    _run_task_stage(
        runner,
        run_tasks,
        stage_label="gnina",
        log_path=log_path,
        parallel=parallel,
        build_cmd_fn=stages.build_gnina_command,
    )


def _run_md_stage(runner: PipelineRunner, tasks: list[StageTask]) -> None:
    """Run MD validation for all unique pairs in tasks.

    Args:
        runner: Pipeline runner instance.
        tasks: List of tasks to execute.
    """
    log_path = runner.get_log_path("md")
    unique_pairs = list(dict.fromkeys(t.pair for t in tasks))
    for pair_id in unique_pairs:
        cmd = stages.build_md_command(pair_id)
        runner.run_cmd(cmd, log_path)


if __name__ == "__main__":
    main()
