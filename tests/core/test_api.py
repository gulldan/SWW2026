"""Smoke tests for programmatic workflow API."""

from __future__ import annotations

from dualdocking.api import run_full_experiment, run_post_pipeline, run_stages


def test_run_stages_api_dry_run() -> None:
    """Programmatic stage runner should support dry-run."""
    run_stages(
        stages="analyze",
        dry_run=True,
        skip_preflight=True,
    )


def test_run_post_pipeline_api_dry_run() -> None:
    """Programmatic post runner should support dry-run."""
    run_post_pipeline(dry_run=True, latex_mode="off")


def test_run_full_experiment_api_dry_run() -> None:
    """Programmatic full pipeline runner should support dry-run."""
    run_full_experiment(
        dry_run=True,
        pairs=["CDK2_ERK2"],
        seeds=[42],
        with_post=False,
    )
