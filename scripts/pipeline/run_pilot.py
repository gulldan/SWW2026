"""Pilot matrix run: 1 pair x 3 modes x 2 seeds.

Thin wrapper around ``run_full_experiment.py`` with pilot-specific defaults.
Runs CDK2_ERK2 with seeds [42, 137] through Stage 1 (generate+score) and
Stage 2 (post-processing), then verifies that all expected artifacts are
non-empty.

Usage::

    uv run python scripts/pipeline/run_pilot.py
    uv run python scripts/pipeline/run_pilot.py --dry-run
    uv run python scripts/pipeline/run_pilot.py --pair AChE_BChE --seeds 42,256
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from loguru import logger

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PILOT_PAIR: str = "CDK2_ERK2"
"""Default target pair for the pilot run."""

PILOT_SEEDS: str = "42,137"
"""Default comma-separated seeds for the pilot run."""


def _verify_pilot_artifacts(pair: str, seeds: list[int]) -> list[str]:
    """Check that the pilot produced all expected non-empty artifacts.

    Verifies ``generated.parquet/.csv`` and ``scored_joined.parquet/.csv``
    for every (mode, seed) combination, plus Stage 2 report artifacts
    (``metrics.json``, ``stats.json``).

    Args:
        pair: Target pair identifier that was run.
        seeds: List of random seeds that were run.

    Returns:
        List of human-readable issue strings; empty if everything is OK.
    """
    issues: list[str] = []
    results_root = PROJECT_ROOT / "output" / "results"

    for mode in ("single_a", "single_b", "double"):
        for seed in seeds:
            run_dir = results_root / pair / mode / f"seed_{seed}"
            from dualdocking.data.io_utils import stem_exists

            scored_stem = run_dir / "scored_joined"
            generated_stem = run_dir / "generated"

            if not stem_exists(generated_stem):
                issues.append(f"Missing: {run_dir.relative_to(PROJECT_ROOT)}/generated.parquet")
            else:
                pq = generated_stem.with_suffix(".parquet")
                actual = pq if pq.exists() else generated_stem.with_suffix(".csv")
                if actual.stat().st_size < 100:
                    issues.append(f"Too small: {actual.relative_to(PROJECT_ROOT)}")

            if not stem_exists(scored_stem):
                issues.append(f"Missing: {run_dir.relative_to(PROJECT_ROOT)}/scored_joined.parquet")
            else:
                pq = scored_stem.with_suffix(".parquet")
                actual = pq if pq.exists() else scored_stem.with_suffix(".csv")
                if actual.stat().st_size < 100:
                    issues.append(f"Too small: {actual.relative_to(PROJECT_ROOT)}")

    # Stage 2 artifacts
    report = PROJECT_ROOT / "report"
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
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--parallel", default=1, type=int)
@click.option("--verify-only", is_flag=True, default=False, help="Skip run, only verify artifacts.")
@click.option(
    "--latex-mode",
    type=click.Choice(["off", "draft", "full"]),
    default="draft",
    show_default=True,
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

    Delegates to ``run_full_experiment.main`` with pilot-scoped defaults,
    then checks that all expected output files are present and non-empty.

    Args:
        pair: Target pair identifier.
        seeds: Comma-separated random seeds.
        dry_run: Print commands without executing.
        parallel: Number of parallel tasks.
        verify_only: Skip the run, only verify previously generated artifacts.
        latex_mode: LaTeX report mode (``"off"``, ``"draft"``, or ``"full"``).
    """
    seed_list = [int(s) for s in seeds.split(",")]
    logger.info("Pilot: pair={}, seeds={}, modes=[single_a, single_b, double]", pair, seed_list)

    if not verify_only:
        from scripts.pipeline.run_full_experiment import main as run_main

        # Build args for run_full_experiment
        args = [
            "--pair",
            pair,
            "--seeds",
            seeds,
            "--resume",
            "--with-post",
            "--latex-mode",
            latex_mode,
            "--parallel",
            str(parallel),
        ]
        if dry_run:
            args.append("--dry-run")

        run_main.main(args, standalone_mode=False)

    if dry_run:
        logger.info("Dry run — skipping artifact verification")
        return

    # Verify artifacts
    logger.info("Verifying pilot artifacts...")
    issues = _verify_pilot_artifacts(pair, seed_list)
    if issues:
        for issue in issues:
            logger.error("  {}", issue)
        logger.error("Pilot verification FAILED: {} issue(s)", len(issues))
        sys.exit(1)
    else:
        logger.info("Pilot verification PASSED: all artifacts present and non-empty")


if __name__ == "__main__":
    main()
