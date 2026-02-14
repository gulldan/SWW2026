"""Run full experiment across pairs, modes, and seeds.

Thin CLI wrapper around ``dualdocking.pipeline.PipelineRunner``.

Usage::

    # Run full experiment with default config
    uv run python scripts/pipeline/run_full_experiment.py

    # Run specific pair with custom seeds
    uv run python scripts/pipeline/run_full_experiment.py --pair CDK2_ERK2 --seeds 42,137

    # Run with post-processing and LaTeX report
    uv run python scripts/pipeline/run_full_experiment.py --with-post --latex-mode draft

    # Parallel execution with 4 GPUs
    uv run python scripts/pipeline/run_full_experiment.py --parallel 4
"""

from __future__ import annotations

from pathlib import Path

import click
from loguru import logger

from dualdocking.pipeline import PipelineConfig, PipelineRunner


@click.command()
@click.option("--dry-run", is_flag=True, default=False, help="Print configuration without executing.")
@click.option("--parallel", default=0, type=int, help="Parallel tasks. 0 = auto-detect (# GPUs).")
@click.option("--resume", is_flag=True, default=False, help="Skip generation if 'generated' file exists.")
@click.option(
    "--pair",
    multiple=True,
    default=None,
    help="Filter to specific pair(s). Can be repeated. Default: all pairs from config.",
)
@click.option(
    "--seeds",
    "seeds_override",
    default=None,
    help="Comma-separated seed list override. Default: all seeds from config.",
)
@click.option(
    "--prepare-freed-data/--skip-prepare-freed-data",
    default=True,
    show_default=True,
    help="Check/download FREED++ data dependencies before running.",
)
@click.option(
    "--prepare-baseline/--skip-prepare-baseline",
    default=False,
    show_default=True,
    help="Download baseline ZINC archive before running.",
)
@click.option(
    "--baseline-url",
    default=None,
    help="Custom URL for baseline archive (or use ZINC_BASELINE_URL env var).",
)
@click.option(
    "--with-post/--skip-post",
    default=False,
    show_default=True,
    help="Run post-processing (aggregate/metrics/plots/render/summarize/latex) after generate+score.",
)
@click.option(
    "--latex-mode",
    type=click.Choice(["off", "draft", "full"]),
    default="off",
    show_default=True,
    help="LaTeX compilation mode (only used with --with-post).",
)
def main(
    dry_run: bool,
    parallel: int,
    resume: bool,
    pair: tuple[str, ...],
    seeds_override: str | None,
    prepare_freed_data: bool,
    prepare_baseline: bool,
    baseline_url: str | None,
    with_post: bool,
    latex_mode: str,
) -> None:
    """Run the full DualDocking experiment pipeline.

    Loads configuration from ``configs/``, validates scorer backend,
    ensures calibration, then dispatches generate+score tasks across GPUs.
    Optionally runs post-processing (aggregate, metrics, plots, report).

    Args:
        dry_run: Print configuration and task count without executing.
        parallel: Number of parallel tasks (0 = auto-detect from GPU count).
        resume: Skip generation if ``generated`` file already exists.
        pair: Filter to specific pair(s); empty = all active pairs.
        seeds_override: Comma-separated seed list, or ``None`` for config default.
        prepare_freed_data: Fetch FREED++ data dependencies before running.
        prepare_baseline: Fetch ZINC baseline archive before running.
        baseline_url: Custom URL for baseline archive.
        with_post: Run post-processing pipeline after generate+score.
        latex_mode: LaTeX report mode (``"off"``, ``"draft"``, or ``"full"``).
    """
    # Load configuration
    config_dir = Path(__file__).resolve().parent.parent.parent / "configs"
    config = PipelineConfig.from_config_dir(config_dir)

    # Apply CLI overrides
    config.parallel = parallel
    config.resume = resume
    config.prepare_freed_data = prepare_freed_data
    config.prepare_baseline = prepare_baseline
    config.baseline_url = baseline_url
    config.latex_mode = latex_mode  # type: ignore[assignment]

    # Parse filters
    pair_list = list(pair) if pair else None
    seeds = [int(s) for s in seeds_override.split(",")] if seeds_override else None

    # Validate filters
    if pair_list:
        active_pairs = config.get_active_pairs()
        invalid = [p for p in pair_list if p not in active_pairs]
        if invalid:
            logger.error("Invalid pair(s): {}. Available: {}", invalid, list(active_pairs.keys()))
            raise SystemExit(1)

    # Generate task list
    tasks = config.generate_tasks(pair_filter=pair_list, seed_filter=seeds)

    if dry_run:
        logger.info("DRY RUN — Configuration:")
        logger.info("  Project root: {}", config.project_root)
        logger.info("  Pairs: {}", pair_list or "all")
        logger.info("  Seeds: {}", seeds or config.seeds)
        logger.info("  Modes: {}", config.modes)
        logger.info("  Scorer: {} (backend: {})", config.scorer.engine, config.scorer.backend)
        logger.info("  Parallel: {} (0 = auto-detect)", config.parallel)
        logger.info("  Resume: {}", config.resume)
        logger.info("  Prepare FREED data: {}", config.prepare_freed_data)
        logger.info("  Prepare baseline: {}", config.prepare_baseline)
        logger.info("  With post-processing: {}", with_post)
        logger.info("  LaTeX mode: {}", latex_mode)
        logger.info("  Total tasks: {}", len(tasks))
        return

    # Execute pipeline
    runner = PipelineRunner(config)
    runner.run_full_experiment(
        pairs=pair_list,
        seeds=seeds,
        resume=resume,
        parallel=parallel,
        with_post=with_post,
        latex_mode=latex_mode,  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    main()
