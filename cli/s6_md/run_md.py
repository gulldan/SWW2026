"""CLI wrapper for MD validation.

Thin Click interface around :func:`dualdocking.validation.md_pipeline.run_md_validation`.
All logic lives in the library modules.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import click
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@click.command()
@click.option("--pair", required=True, help="Pair ID from configs/pairs.yaml")
@click.option("--mode", default=None, help="Generation mode (default: all configured modes)")
@click.option("--top-n", default=3, type=int, show_default=True, help="Top-N molecules per mode")
@click.option("--config", default="configs/pairs.yaml")
@click.option("--experiment-config", default="configs/experiment.yaml")
@click.option("--mmpbsa", is_flag=True, default=False, help="Run MM-GBSA via gmx_MMPBSA")
@click.option("--dry-run", is_flag=True, default=False, help="Print plan without executing")
@click.option("--simulation-time", type=float, default=None, help="Override simulation time (ns)")
@click.option("--no-gpu", is_flag=True, default=False, help="Disable automatic GPU acceleration")
@click.option("--docker", is_flag=True, default=False, help="Run inside Docker container")
@click.option("--docker-image", default="dualdocking/md:latest", show_default=True, help="Docker image tag")
def main(  # noqa: PLR0913
    pair: str,
    mode: str | None,
    top_n: int,
    config: str,
    experiment_config: str,
    mmpbsa: bool,
    dry_run: bool,
    simulation_time: float | None,
    no_gpu: bool,
    docker: bool,
    docker_image: str,
) -> None:
    """Run MD validation for top candidate molecules.

    Selects the top-N molecules per mode from scored results, runs GROMACS
    MD for each molecule against both targets, and writes per-frame RMSD,
    hydrogen bond, and interaction energy data.
    """
    if docker:
        from dualdocking.validation.md_docker import build_md_image, run_md_in_docker

        if not dry_run:
            ok = build_md_image(PROJECT_ROOT, image=docker_image)
            if not ok:
                logger.error("Failed to build Docker image {}", docker_image)
                sys.exit(1)

        result = run_md_in_docker(
            pair=pair,
            mode=mode,
            top_n=top_n,
            config=config,
            experiment_config=experiment_config,
            mmpbsa=mmpbsa,
            simulation_time=simulation_time,
            no_gpu=no_gpu,
            image=docker_image,
            project_root=PROJECT_ROOT,
            dry_run=dry_run,
        )
        if result.stdout:
            click.echo(result.stdout)
        if result.stderr:
            click.echo(result.stderr, err=True)
        sys.exit(result.returncode)

    # Check prerequisites (only when running locally)
    for tool in ("gmx", "antechamber", "acpype", "obabel"):
        if not shutil.which(tool):
            logger.error("{} not found in PATH. Install prerequisites first.", tool)
            sys.exit(1)

    from dualdocking.config import load_experiment, load_pairs
    from dualdocking.validation.md_config import MDConfig
    from dualdocking.validation.md_pipeline import (
        run_md_validation,
        select_top_molecules,
        write_summary,
    )

    pairs_cfg = load_pairs(PROJECT_ROOT / config)["pairs"]
    exp_cfg = load_experiment(PROJECT_ROOT / experiment_config)

    if pair not in pairs_cfg:
        logger.error("Pair '{}' not found", pair)
        sys.exit(1)

    pair_cfg = pairs_cfg[pair]
    seeds = exp_cfg.get("seeds", [42, 137, 256, 512, 1024])
    modes = [mode] if mode else exp_cfg.get("modes", ["single_a", "single_b", "double"])

    # Build config kwargs, applying CLI overrides
    config_kwargs: dict = {
        "pair_id": pair,
        "top_n": top_n,
        "modes": modes,
        "seeds": seeds,
        "mmpbsa": mmpbsa,
        "output_dir": PROJECT_ROOT / "output" / "md",
        "results_dir": PROJECT_ROOT / "output" / "results",
        "targets_dir": PROJECT_ROOT / "output" / "data" / "targets",
    }
    if simulation_time is not None:
        config_kwargs["simulation_time_ns"] = simulation_time
    if no_gpu:
        config_kwargs["gpu_auto"] = False

    md_config = MDConfig(**config_kwargs)

    if dry_run:
        for m in md_config.modes:
            logger.info("=== Mode: {} ===", m)
            top_df = select_top_molecules(
                pair, m, top_n, seeds, md_config.results_dir
            )
            if top_df.empty:
                logger.warning("No scored molecules found for {}/{}", pair, m)
                continue
            logger.info("Selected {} molecules for MD", len(top_df))
            for i, row in top_df.iterrows():
                logger.info("  [DRY RUN] mol_{}: {:.80s}", i, row["smiles"])
        return

    results = run_md_validation(md_config, pair_cfg)

    if not results:
        logger.error("No MD results produced for pair {}", pair)
        sys.exit(1)

    summary_path = PROJECT_ROOT / "output" / "md" / "md_summary.parquet"
    write_summary(results, summary_path)

    logger.info("MD validation complete for pair {}", pair)


if __name__ == "__main__":
    main()
