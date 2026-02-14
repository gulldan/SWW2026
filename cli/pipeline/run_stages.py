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
from pathlib import Path

import click
from loguru import logger

from dualdocking.pipeline import PipelineConfig, PipelineRunner, StageTask, checks, stages


@click.command()
@click.option("--stages", "stages_arg", default=None, help="Comma-separated stages to run (or 'all')")
@click.option("--list-stages", is_flag=True, help="List available stages and exit")
@click.option("--pair", default=None, help="Filter to a specific pair")
@click.option("--seeds", default=None, help="Comma-separated seed list")
@click.option("--molecules", default=None, type=click.Path(exists=True), help="Molecule file for scoring")
@click.option("--latex-mode", type=click.Choice(["off", "draft", "full"]), default="off")
@click.option("--dry-run", is_flag=True, help="Show what would be run without executing")
@click.option("--skip-preflight/--no-skip-preflight", default=False, help="Skip input checks")
def main(
    stages_arg: str | None,
    list_stages: bool,
    pair: str | None,
    seeds: str | None,
    molecules: str | None,
    latex_mode: str,
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
        elif stage == "calibrate":
            runner.run_stage_calibrate()
        elif stage == "generate":
            _run_generate_stage(runner, tasks)
        elif stage == "score":
            _run_score_stage(runner, tasks, molecules)
        elif stage == "analyze":
            runner.run_stage_analyze()
        elif stage == "report":
            runner.run_stage_report(latex_mode=latex_mode)  # type: ignore[arg-type]
        elif stage == "audit":
            runner.run_stage_audit()
        elif stage == "gnina":
            _run_gnina_stage(runner, tasks)
        elif stage == "md":
            _run_md_stage(runner, tasks)

        logger.info("=== Stage {} complete ===", stage)

    logger.info("All selected stages completed successfully")


def _run_generate_stage(runner: PipelineRunner, tasks: list[StageTask]) -> None:
    """Run generate stage for all tasks.

    Args:
        runner: Pipeline runner instance.
        tasks: List of tasks to execute.
    """
    log_path = runner.get_log_path("generate")
    for task in tasks:
        cmd = stages.build_generate_command(task)
        runner.run_cmd(cmd, log_path)


def _run_score_stage(
    runner: PipelineRunner,
    tasks: list[StageTask],
    molecules: str | None,
) -> None:
    """Run score stage for all tasks.

    Args:
        runner: Pipeline runner instance.
        tasks: List of tasks to execute.
        molecules: Optional path to external molecule file.
    """
    log_path = runner.get_log_path("score")
    for task in tasks:
        cmd = stages.build_score_command(task, molecules=molecules)
        runner.run_cmd(cmd, log_path)


def _run_gnina_stage(runner: PipelineRunner, tasks: list[StageTask]) -> None:
    """Run GNINA cross-validation for all tasks.

    Args:
        runner: Pipeline runner instance.
        tasks: List of tasks to execute.
    """
    log_path = runner.get_log_path("gnina")
    for task in tasks:
        cmd = stages.build_gnina_command(task)
        runner.run_cmd(cmd, log_path)


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
