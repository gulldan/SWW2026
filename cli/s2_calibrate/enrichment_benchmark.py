"""Enrichment benchmark: validate that UniDock separates actives from decoys.

For each target, docks ChEMBL actives and compares their score distribution
to the MOSES calibration decoy set. Computes AUROC, BEDROC(alpha=20), and
enrichment factor metrics.  Targets below quality thresholds are flagged.

Usage::

    uv run python cli/s2_calibrate/enrichment_benchmark.py --dry-run
    uv run python cli/s2_calibrate/enrichment_benchmark.py --target-id BACE1 --target-id CDK2
    uv run python cli/s2_calibrate/enrichment_benchmark.py
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import click
from loguru import logger

from dualdocking.analysis.calibration.enrichment import (
    load_active_smiles,
    load_decoy_scores,
    print_summary,
    process_target,
    write_outputs,
)
from dualdocking.config import (
    collect_all_target_ids,
    collect_blocked_target_ids,
    collect_target_pdb_map,
)
from dualdocking.scoring.docking import (
    DockerRunner,
    DockingParams,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Backward-compatible aliases
_collect_blocked_target_ids = collect_blocked_target_ids
_collect_all_target_ids = collect_all_target_ids
_collect_target_pdb_map = collect_target_pdb_map
_load_decoy_scores = load_decoy_scores
_load_active_smiles = load_active_smiles
_print_summary = print_summary


def _dry_run_report(
    all_target_ids: list[str],
    blocked_ids: set[str],
    target_pdb_map: dict[str, str],
    chembl_dir: Path,
    calibration_scores_dir: Path,
    scores_dir: Path,
    max_actives: int,
    seed: int,
    auroc_threshold: float,
    bedroc_threshold: float,
) -> None:
    """Print a dry-run target summary without performing any docking."""
    click.echo("--- Enrichment Benchmark (dry run) ---")
    click.echo(f"Targets: {len(all_target_ids)} ({len(blocked_ids)} blocked)")
    click.echo(f"Thresholds: AUROC>{auroc_threshold}, BEDROC>{bedroc_threshold}")
    click.echo(f"Max actives: {max_actives}, Seed: {seed}")
    click.echo()
    for tid in all_target_ids:
        blocked = " [BLOCKED]" if tid in blocked_ids else ""
        pdb_id = target_pdb_map.get(tid, "?")
        try:
            actives = _load_active_smiles(tid, chembl_dir, max_actives, seed)
            n_actives = len(actives)
        except Exception:
            n_actives = 0
        from dualdocking.data.io_utils import stem_exists

        decoy_stem = calibration_scores_dir / tid
        n_decoys = len(_load_decoy_scores(decoy_stem)) if stem_exists(decoy_stem) else 0
        has_active_scores = stem_exists(scores_dir / f"{tid}_actives")
        click.echo(
            f"  {tid:<10} PDB={pdb_id:<5} actives={n_actives:>4} "
            f"decoys={n_decoys:>5} active_scores={'YES' if has_active_scores else 'NO'}"
            f"{blocked}"
        )


@click.command()
@click.option("--calibration-config", default="configs/calibration.yaml")
@click.option("--metrics-config", default="configs/metrics.yaml")
@click.option("--target-id", multiple=True, help="Specific target IDs (repeatable)")
@click.option("--max-actives", default=200, type=int, help="Max actives per target")
@click.option("--seed", default=137, type=int, help="Sampling seed")
@click.option("--docker-image", default="dualdocking/freedpp:latest")
@click.option("--gpus", default="all", help="Docker --gpus value")
@click.option("--output", default="output/data/enrichment/enrichment_results.json")
@click.option("--csv-output", default="report/data/enrichment_metrics.csv")
@click.option("--figures-dir", default="report/figures/enrichment")
@click.option("--dry-run", is_flag=True, help="Validate config without docking")
@click.option("--skip-docking", is_flag=True, help="Use existing scores only")
def main(  # noqa: PLR0913
    calibration_config: str,
    metrics_config: str,
    target_id: tuple[str, ...],
    max_actives: int,
    seed: int,
    docker_image: str,
    gpus: str,
    output: str,
    csv_output: str,
    figures_dir: str,
    dry_run: bool,
    skip_docking: bool,
) -> None:
    """Run enrichment benchmark: dock actives vs decoys per target."""
    import yaml

    from dualdocking.analysis.enrichment import EnrichmentResult
    from dualdocking.config import load_pairs

    pairs = load_pairs(PROJECT_ROOT / "configs")["pairs"]
    blocked_ids = _collect_blocked_target_ids(pairs)
    all_target_ids = _collect_all_target_ids(pairs)
    target_pdb_map = _collect_target_pdb_map(pairs)

    if target_id:
        all_target_ids = [t for t in all_target_ids if t in set(target_id)]
        if not all_target_ids:
            raise click.ClickException("No target IDs matched the provided --target-id values")

    metrics_cfg = yaml.safe_load((PROJECT_ROOT / metrics_config).read_text())
    enrichment_cfg = metrics_cfg.get("enrichment_quality", {})
    auroc_threshold = float(enrichment_cfg.get("auroc_threshold", 0.6))
    bedroc_threshold = float(enrichment_cfg.get("bedroc_alpha20_threshold", 0.3))
    alpha = float(enrichment_cfg.get("alpha", 20.0))

    cal_cfg = yaml.safe_load((PROJECT_ROOT / calibration_config).read_text())
    docking_cfg = cal_cfg.get("docking", {})

    chembl_dir = PROJECT_ROOT / "output" / "data" / "chembl"
    scores_dir = PROJECT_ROOT / "output" / "data" / "enrichment" / "scores"
    receptors_dir = PROJECT_ROOT / "output" / "data" / "enrichment" / "receptors"
    calibration_scores_dir = PROJECT_ROOT / "output" / "data" / "calibration" / "scores"
    figures_path = PROJECT_ROOT / figures_dir

    scores_dir.mkdir(parents=True, exist_ok=True)
    receptors_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        _dry_run_report(
            all_target_ids,
            blocked_ids,
            target_pdb_map,
            chembl_dir,
            calibration_scores_dir,
            scores_dir,
            max_actives,
            seed,
            auroc_threshold,
            bedroc_threshold,
        )
        return

    params = DockingParams(
        exhaustiveness=int(docking_cfg.get("exhaustiveness", 8)),
        num_modes=int(docking_cfg.get("num_modes", 9)),
        energy_range=float(docking_cfg.get("energy_range", 3.0)),
        timeout_seconds=int(docking_cfg.get("timeout_seconds", 120)),
        box_padding=float(docking_cfg.get("box_padding", 10.0)),
    )
    batch_size = int(docking_cfg.get("batch_size", 64))
    binary = "unidock"

    results: list[EnrichmentResult] = []
    for tid in sorted(blocked_ids & set(all_target_ids)):
        results.append(
            EnrichmentResult(
                target_id=tid,
                n_actives=0,
                n_decoys=0,
                auroc=0.0,
                bedroc_alpha20=0.0,
                ef_1pct=0.0,
                ef_5pct=0.0,
                passed=False,
                status="skipped:blocked",
                error="Blocked pair (non-human species)",
            )
        )
        logger.info("[SKIP] {} -- blocked pair", tid)

    from dualdocking.data.io_utils import stem_exists

    active_targets = [t for t in all_target_ids if t not in blocked_ids]
    need_docker = not skip_docking and any(
        not stem_exists(scores_dir / f"{tid}_actives") or not stem_exists(calibration_scores_dir / tid) for tid in active_targets
    )

    if need_docker:
        from dualdocking.generation.freedpp import _ensure_docker_image

        _ensure_docker_image(docker_image)

    runner_ctx: Any = DockerRunner(docker_image, PROJECT_ROOT, gpus=gpus) if need_docker else nullcontext(None)
    with runner_ctx as runner:
        for tid in active_targets:
            result = process_target(
                tid,
                target_pdb_map,
                runner,
                binary,
                params,
                cal_cfg,
                seed,
                batch_size,
                max_actives,
                alpha,
                auroc_threshold,
                bedroc_threshold,
                chembl_dir,
                scores_dir,
                receptors_dir,
                calibration_scores_dir,
                figures_path,
                skip_docking,
                PROJECT_ROOT,
            )
            results.append(result)

    results.sort(key=lambda r: r.target_id)
    write_outputs(
        results,
        target_pdb_map,
        output,
        csv_output,
        figures_path,
        seed,
        max_actives,
        auroc_threshold,
        bedroc_threshold,
        alpha,
        PROJECT_ROOT,
    )
    _print_summary(results, auroc_threshold, bedroc_threshold, alpha)


if __name__ == "__main__":
    main()
