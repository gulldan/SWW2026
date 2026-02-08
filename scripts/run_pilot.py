"""Pilot matrix run: 1 pair x 3 modes x 2 seeds (SCI-09 / KVA-273).

Thin wrapper around run_full_experiment.py with pilot-specific defaults.
Runs CDK2_ERK2 with seeds [42, 137] through Stage 1 + Stage 2, then
verifies that all expected artifacts are non-empty.

Usage:
    uv run python scripts/run_pilot.py
    uv run python scripts/run_pilot.py --dry-run
    uv run python scripts/run_pilot.py --pair AChE_BChE --seeds 42,256
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PILOT_PAIR = "CDK2_ERK2"
PILOT_SEEDS = "42,137"


def _verify_pilot_artifacts(pair: str, seeds: list[int]) -> list[str]:
    """Check that pilot produced all expected non-empty artifacts."""
    issues: list[str] = []
    results_root = PROJECT_ROOT / "results"

    for mode in ("single_a", "single_b", "double"):
        for seed in seeds:
            run_dir = results_root / pair / mode / f"seed_{seed}"
            scored = run_dir / "scored_joined.csv"
            generated = run_dir / "generated.csv"

            if not generated.exists():
                issues.append(f"Missing: {generated.relative_to(PROJECT_ROOT)}")
            elif generated.stat().st_size < 100:
                issues.append(f"Too small: {generated.relative_to(PROJECT_ROOT)}")

            if not scored.exists():
                issues.append(f"Missing: {scored.relative_to(PROJECT_ROOT)}")
            elif scored.stat().st_size < 100:
                issues.append(f"Too small: {scored.relative_to(PROJECT_ROOT)}")

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
    """Run pilot matrix and verify artifacts."""
    seed_list = [int(s) for s in seeds.split(",")]
    logger.info("Pilot: pair={}, seeds={}, modes=[single_a, single_b, double]", pair, seed_list)

    if not verify_only:
        from scripts.run_full_experiment import main as run_main

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
