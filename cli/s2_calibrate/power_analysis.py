"""Statistical power analysis for the dual-target experiment.

Computes minimum detectable effect size (MDE) given the experimental
design (m=5 RL seeds per mode), reports Cohen's d thresholds, and
calculates the number of seeds required to detect a medium effect
(d=0.5) at 80% power.

Usage::

    uv run python cli/s2_calibrate/power_analysis.py
    uv run python cli/s2_calibrate/power_analysis.py --output output/data/calibration/power_analysis.json
    uv run python cli/s2_calibrate/power_analysis.py --alpha 0.01
"""

from __future__ import annotations

from pathlib import Path

import click

from dualdocking.analysis.calibration.power import (
    cliffs_delta_to_cohens_d,
    power_at_effect,
    required_seeds,
    run_power_analysis,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@click.command()
@click.option(
    "--output",
    default="output/data/calibration/power_analysis.json",
    help="Output JSON path relative to project root.",
)
@click.option("--seeds", default=5, type=int, help="Number of RL seeds (m).")
@click.option("--alpha", default=0.05, type=float, help="Significance level.")
@click.option("--power", default=0.80, type=float, help="Desired statistical power.")
def main(output: str, seeds: int, alpha: float, power: float) -> None:
    """Run power analysis and write results."""
    manifest = run_power_analysis(
        project_root=PROJECT_ROOT,
        output=output,
        seeds=seeds,
        alpha=alpha,
        power=power,
    )

    m = seeds
    mde_two = manifest["minimum_detectable_effect"]["two_sided_cohens_d"]
    mde_one = manifest["minimum_detectable_effect"]["one_sided_cohens_d"]

    effect_sizes = {"small": 0.2, "medium": 0.5, "large": 0.8}
    power_at_m = {label: power_at_effect(m, d, alpha) for label, d in effect_sizes.items()}
    seeds_required = {label: required_seeds(d, alpha, power) for label, d in effect_sizes.items()}

    cliffs_medium = 0.33
    d_from_cliffs_medium = cliffs_delta_to_cohens_d(cliffs_medium)
    power_for_cliffs_medium = power_at_effect(m, d_from_cliffs_medium, alpha)
    seeds_for_cliffs_medium = required_seeds(d_from_cliffs_medium, alpha, power)

    click.echo("\n=== Power Analysis Summary ===")
    click.echo(f"Seeds (m): {m}  |  alpha: {alpha}  |  target power: {power:.0%}")
    click.echo()
    click.echo(f"Minimum detectable effect (two-sided):  Cohen's d = {mde_two:.2f}")
    click.echo(f"Minimum detectable effect (one-sided):  Cohen's d = {mde_one:.2f}")
    click.echo()
    click.echo("Power at m=5 for standard effect sizes:")
    for label, d in effect_sizes.items():
        click.echo(f"  {label:>8} (d={d:.1f}): power = {power_at_m[label]:.1%}  (need {seeds_required[label]} seeds for 80%)")
    click.echo()
    click.echo("Cliff's delta mapping (from goal.md success criteria):")
    click.echo(
        f"  |delta| = {cliffs_medium}: "
        f"~Cohen's d = {d_from_cliffs_medium:.2f}, "
        f"power = {power_for_cliffs_medium:.1%}, "
        f"need {seeds_for_cliffs_medium} seeds"
    )
    click.echo()
    click.echo("Key limitation: m=5 provides adequate power ONLY for very large effects.")
    click.echo("Rely on effect size + cross-pair consistency, not p-values alone.")
    click.echo(f"\nFull results: {PROJECT_ROOT / output}")


if __name__ == "__main__":
    main()
