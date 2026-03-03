"""Calibrate normalization statistics for docking scores.

Loads per-target calibration scores, runs bootstrap stability checks,
and writes a ``normalization.yaml`` config used by downstream scoring
to z-score-normalize docking results.
"""

from __future__ import annotations

from pathlib import Path

import click

from dualdocking.analysis.calibration.normalization import (
    compute_normalization,
    docking_version_fallback,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Backward-compatible alias
_docking_version = docking_version_fallback


@click.command()
@click.option("--calibration-config", default="configs/calibration.yaml")
@click.option("--calibration-set", default=None)
@click.option("--subset-name", default="zinc_druglike_calibration_1000_v2024")
@click.option("--n", default=1000, type=int)
@click.option("--seed", default=137, type=int)
@click.option("--scores-dir", default="output/data/calibration/scores")
@click.option("--output", default="configs/normalization.yaml")
@click.option("--docking-program", default="unidock_gpu")
@click.option("--docking-binary", default=None, help="Binary to query for version (optional)")
@click.option("--docking-version", default=None, help="Override docking version (optional)")
def main(
    calibration_config: str,
    calibration_set: str | None,
    subset_name: str,
    n: int,
    seed: int,
    scores_dir: str,
    output: str,
    docking_program: str,
    docking_binary: str | None,
    docking_version: str | None,
) -> None:
    """Compute normalization statistics and write normalization.yaml."""
    compute_normalization(
        project_root=PROJECT_ROOT,
        calibration_config=calibration_config,
        calibration_set=calibration_set,
        subset_name=subset_name,
        n=n,
        seed=seed,
        scores_dir=scores_dir,
        output=output,
        docking_program=docking_program,
        docking_binary=docking_binary,
        docking_version_override=docking_version,
    )


if __name__ == "__main__":
    main()
