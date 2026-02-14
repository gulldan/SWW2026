"""Run post-processing stage: analytics and report generation.

Thin CLI wrapper around ``dualdocking.pipeline.PipelineRunner.run_post_pipeline``.

Usage::

    # Run post-processing with draft LaTeX report
    uv run python cli/pipeline/run_post_pipeline.py

    # Skip LaTeX generation
    uv run python cli/pipeline/run_post_pipeline.py --latex-mode off

    # Full LaTeX compilation
    uv run python cli/pipeline/run_post_pipeline.py --latex-mode full
"""

from __future__ import annotations

from pathlib import Path

import click

from dualdocking.pipeline import PipelineRunner


@click.command()
@click.option("--dry-run", is_flag=True, default=False, help="Print commands without executing.")
@click.option(
    "--latex-mode",
    type=click.Choice(["off", "draft", "full"]),
    default="draft",
    show_default=True,
    help="LaTeX report generation mode.",
)
def main(dry_run: bool, latex_mode: str) -> None:
    """Run the post-processing pipeline (aggregate, metrics, plots, report).

    Executes the full analysis and reporting pipeline: aggregation, metrics
    computation, result plotting, molecule rendering, summarization, and
    optional LaTeX report generation.

    Args:
        dry_run: Print pipeline steps without executing.
        latex_mode: LaTeX report mode (``"off"``, ``"draft"``, or ``"full"``).
    """
    if dry_run:
        click.echo("DRY RUN — Post-processing pipeline steps:")
        click.echo("  1. Aggregate results")
        click.echo("  2. Compute metrics")
        click.echo("  3. Plot results")
        click.echo("  4. Render molecules")
        click.echo("  5. Summarize")
        if latex_mode != "off":
            click.echo(f"  6. Generate LaTeX report (mode: {latex_mode})")
        return

    config_dir = Path(__file__).resolve().parent.parent.parent / "configs"
    runner = PipelineRunner.from_config_dir(config_dir)
    runner.run_post_pipeline(latex_mode=latex_mode)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
