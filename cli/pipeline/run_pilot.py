"""Pilot matrix run: 1 pair x 3 modes x 2 seeds.

Thin wrapper around ``PipelineRunner.run_full_experiment`` with pilot-specific
defaults and artifact verification.

Usage::

    # Run default pilot (CDK2_ERK2, seeds 42,137)
    uv run python cli/pipeline/run_pilot.py

    # Run custom pilot configuration
    uv run python cli/pipeline/run_pilot.py --pair AChE_BChE --seeds 42,256

    # Verify existing pilot artifacts without re-running
    uv run python cli/pipeline/run_pilot.py --verify-only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from loguru import logger

from dualdocking.pipeline import PipelineConfig, PipelineRunner

PILOT_PAIR: str = "CDK2_ERK2"
"""Default target pair for the pilot run."""

PILOT_SEEDS: str = "42,137"
"""Default comma-separated seeds for the pilot run."""


def verify_pilot_artifacts(
    project_root: Path,
    pair: str,
    seeds: list[int],
    modes: list[str],
) -> list[str]:
    """Check that the pilot produced all expected non-empty artifacts.

    Verifies ``generated`` and ``scored_joined`` files for every (mode, seed)
    combination, plus Stage 2 report artifacts (``metrics.json``, ``stats.json``).

    Args:
        project_root: Project root directory.
        pair: Target pair identifier that was run.
        seeds: List of random seeds that were run.
        modes: List of experiment modes that were run.

    Returns:
        List of human-readable issue strings; empty if everything is OK.
    """
    from dualdocking.data.io_utils import stem_exists

    issues: list[str] = []
    results_root = project_root / "output" / "results"

    for mode in modes:
        for seed in seeds:
            run_dir = results_root / pair / mode / f"seed_{seed}"
            scored_stem = run_dir / "scored_joined"
            generated_stem = run_dir / "generated"

            if not stem_exists(generated_stem):
                issues.append(f"Missing: {run_dir.relative_to(project_root)}/generated.parquet")
            else:
                pq = generated_stem.with_suffix(".parquet")
                actual = pq if pq.exists() else generated_stem.with_suffix(".csv")
                if actual.stat().st_size < 100:
                    issues.append(f"Too small: {actual.relative_to(project_root)}")

            if not stem_exists(scored_stem):
                issues.append(f"Missing: {run_dir.relative_to(project_root)}/scored_joined.parquet")
            else:
                pq = scored_stem.with_suffix(".parquet")
                actual = pq if pq.exists() else scored_stem.with_suffix(".csv")
                if actual.stat().st_size < 100:
                    issues.append(f"Too small: {actual.relative_to(project_root)}")

    # Stage 2 artifacts
    report = project_root / "report"
    for artifact in ("metrics.json", "stats.json"):
        path = report / artifact
        if not path.exists():
            issues.append(f"Missing: report/{artifact}")
        elif path.stat().st_size < 10:
            issues.append(f"Empty: report/{artifact}")
        else:
            data = json.loads(path.read_text())
            if not data:
                issues.append(f"No data in: report/{artifact}")

    return issues


@click.command()
@click.option("--pair", default=PILOT_PAIR, show_default=True, help="Pair to use for pilot.")
@click.option("--seeds", default=PILOT_SEEDS, show_default=True, help="Comma-separated seeds.")
@click.option("--dry-run", is_flag=True, default=False, help="Print configuration without executing.")
@click.option("--parallel", default=1, type=int, help="Number of parallel tasks.")
@click.option("--verify-only", is_flag=True, default=False, help="Skip run, only verify artifacts.")
@click.option(
    "--latex-mode",
    type=click.Choice(["off", "draft", "full"]),
    default="draft",
    show_default=True,
    help="LaTeX report mode.",
)
def main(
    pair: str,
    seeds: str,
    dry_run: bool,
    parallel: int,
    verify_only: bool,
    latex_mode: str,
) -> None:
    """Run the pilot matrix and verify artifacts.

    Executes a minimal experiment configuration for testing/validation, then
    verifies that all expected output files are present and non-empty.

    Args:
        pair: Target pair identifier.
        seeds: Comma-separated random seeds.
        dry_run: Print configuration without executing.
        parallel: Number of parallel tasks.
        verify_only: Skip the run, only verify previously generated artifacts.
        latex_mode: LaTeX report mode (``"off"``, ``"draft"``, or ``"full"``).
    """
    config_dir = Path(__file__).resolve().parent.parent.parent / "configs"
    config = PipelineConfig.from_config_dir(config_dir)

    seed_list = [int(s) for s in seeds.split(",")]
    logger.info("Pilot: pair={}, seeds={}, modes={}", pair, seed_list, config.modes)

    if dry_run:
        logger.info("DRY RUN — Pilot configuration:")
        logger.info("  Pair: {}", pair)
        logger.info("  Seeds: {}", seed_list)
        logger.info("  Modes: {}", config.modes)
        logger.info("  Parallel: {}", parallel)
        logger.info("  LaTeX mode: {}", latex_mode)
        logger.info("  Verify only: {}", verify_only)
        return

    if not verify_only:
        runner = PipelineRunner(config)
        runner.run_full_experiment(
            pairs=[pair],
            seeds=seed_list,
            resume=True,
            parallel=parallel,
            with_post=True,
            latex_mode=latex_mode,  # type: ignore[arg-type]
        )

    # Verify artifacts
    logger.info("Verifying pilot artifacts...")
    issues = verify_pilot_artifacts(config.project_root, pair, seed_list, config.modes)
    if issues:
        for issue in issues:
            logger.error("  {}", issue)
        logger.error("Pilot verification FAILED: {} issue(s)", len(issues))
        sys.exit(1)
    else:
        logger.info("Pilot verification PASSED: all artifacts present and non-empty")


if __name__ == "__main__":
    main()
