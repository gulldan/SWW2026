"""Re-docking validation: dock native co-crystallized ligands back to receptors.

For each target in pairs.yaml, extracts the co-crystallized ligand, docks it
back using UniDock GPU with identical parameters to the experiment, and computes
heavy-atom RMSD between the docked and crystal poses.

RMSD <= 2.0 A indicates the docking setup correctly reproduces the known
binding mode -- a standard sanity check for computational docking studies.

Targets from blocked pairs and apo structures (no co-crystal ligand) are
reported with status "skipped" and excluded from pass/fail counting.

Usage::

    uv run python cli/s2_calibrate/validate_redocking.py
    uv run python cli/s2_calibrate/validate_redocking.py --target-id BACE1 --target-id GSK3B
    uv run python cli/s2_calibrate/validate_redocking.py --include-blocked
    uv run python cli/s2_calibrate/validate_redocking.py --dry-run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
from loguru import logger

from dualdocking.analysis.calibration.redocking import (
    RMSD_THRESHOLD,
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIPPED_BLOCKED,
    RedockResult,
    resolve_pdb_for_target,
    result_to_dict,
    validate_target,
    write_csv,
)
from dualdocking.config import collect_blocked_target_ids
from dualdocking.scoring.docking import (
    DockerRunner,
    DockingParams,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Backward-compatible aliases
_result_to_dict = result_to_dict
_write_csv = write_csv


def _resolve_pdb_for_target(target_id: str, pdb_id: str) -> Path:
    """Find PDB file for a target across all pair directories."""
    return resolve_pdb_for_target(target_id, pdb_id, PROJECT_ROOT)


def _validate_target(
    target_id: str,
    target_cfg: dict[str, Any],
    runner: Any,
    params: Any,
    work_dir: Path,
    rmsd_threshold: float,
) -> RedockResult:
    """Run re-docking validation for a single target."""
    return validate_target(target_id, target_cfg, runner, params, work_dir, rmsd_threshold, PROJECT_ROOT)


@click.command()
@click.option("--config", default="configs/pairs.yaml", help="Path to pairs config")
@click.option("--target-id", multiple=True, help="Specific target IDs (repeatable)")
@click.option("--docker-image", default="dualdocking/freedpp:latest")
@click.option("--gpus", default="all")
@click.option("--rmsd-threshold", default=RMSD_THRESHOLD, type=float)
@click.option("--output", default="output/data/validation/redocking_results.json")
@click.option("--csv-output", default="report/data/redocking_metrics.csv")
@click.option("--include-blocked", is_flag=True, help="Include targets from blocked pairs")
@click.option("--dry-run", is_flag=True, help="Show targets without docking")
def main(
    config: str,
    target_id: tuple[str, ...],
    docker_image: str,
    gpus: str,
    rmsd_threshold: float,
    output: str,
    csv_output: str,
    include_blocked: bool,
    dry_run: bool,
) -> None:
    """Validate docking setup by re-docking co-crystallized ligands."""
    import yaml

    from dualdocking.config import load_pairs
    from dualdocking.scoring.binding_box import extract_ligand_info

    cal_cfg = yaml.safe_load((PROJECT_ROOT / config.replace("pairs", "calibration")).read_text())
    docking_cfg = cal_cfg.get("docking", {})
    params = DockingParams(
        exhaustiveness=int(docking_cfg.get("exhaustiveness", 8)),
        num_modes=int(docking_cfg.get("num_modes", 9)),
        energy_range=float(docking_cfg.get("energy_range", 3.0)),
        timeout_seconds=int(docking_cfg.get("timeout_seconds", 120)),
        box_padding=float(docking_cfg.get("box_padding", 10.0)),
    )

    pairs = load_pairs(PROJECT_ROOT / "configs")["pairs"]
    blocked_ids = collect_blocked_target_ids(pairs)

    targets: dict[str, dict[str, Any]] = {}
    for pair_cfg in pairs.values():
        for role in ("target_a", "target_b"):
            t = pair_cfg[role]
            targets[t["id"]] = t

    if target_id:
        targets = {k: v for k, v in targets.items() if k in set(target_id)}

    if dry_run:
        click.echo("--- Re-docking targets (dry run) ---")
        for tid, t in sorted(targets.items()):
            blocked = " [BLOCKED]" if tid in blocked_ids else ""
            pdb_path = _resolve_pdb_for_target(tid, t["pdb_id"])
            info = extract_ligand_info(pdb_path) if pdb_path.exists() else None
            resname = info.resname if info else "?"
            n_atoms = len(info.coords) if info else 0
            click.echo(f"  {tid:<10} PDB={t['pdb_id']} ligand={resname} atoms={n_atoms}{blocked}")
        return

    work_dir = PROJECT_ROOT / "output" / "data" / "validation" / "redocking_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    results: list[RedockResult] = []

    # Emit skipped results for blocked targets.
    for tid in sorted(blocked_ids & set(targets)):
        t = targets[tid]
        results.append(
            RedockResult(
                target_id=tid,
                pdb_id=t["pdb_id"],
                ligand_resname="--",
                ligand_smiles="",
                n_crystal_atoms=0,
                n_docked_atoms=0,
                docking_score=None,
                rmsd=None,
                centroid_dist=None,
                passed=False,
                status=STATUS_SKIPPED_BLOCKED,
                error="Blocked pair (non-human species)",
            )
        )
        logger.info("[SKIP] {} -- blocked pair", tid)

    active_targets = {k: v for k, v in targets.items() if include_blocked or k not in blocked_ids}

    runner_ctx = DockerRunner(docker_image, PROJECT_ROOT, gpus=gpus)
    with runner_ctx as runner:
        for tid, t in sorted(active_targets.items()):
            result = _validate_target(tid, t, runner, params, work_dir, rmsd_threshold)
            results.append(result)

            rmsd_str = f"{result.rmsd:.2f}" if result.rmsd is not None else "N/A"
            score_str = f"{result.docking_score:.2f}" if result.docking_score is not None else "N/A"
            logger.info(
                "[{}] {} ligand={} score={} RMSD={} A", result.status.upper(), tid, result.ligand_resname, score_str, rmsd_str
            )

    # Sort results by target_id for deterministic output.
    results.sort(key=lambda r: r.target_id)

    # Write JSON.
    out_path = PROJECT_ROOT / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([_result_to_dict(r) for r in results], indent=2))
    logger.info("JSON results saved to {}", out_path)

    # Write Parquet (+ CSV for compatibility).
    import pandas as pd

    from dualdocking.data.io_utils import write_df

    metrics_df = pd.DataFrame([_result_to_dict(r) for r in results])
    metrics_stem = (PROJECT_ROOT / csv_output).with_suffix("")
    write_df(metrics_df, metrics_stem, csv_too=True)
    logger.info("Metrics saved to {}", metrics_stem.with_suffix(".parquet"))

    # --- Summary table ---
    click.echo("\n--- Re-docking Validation Summary ---")
    click.echo(f"RMSD threshold: {rmsd_threshold:.1f} A")
    click.echo(f"{'Target':<10} {'PDB':<6} {'Ligand':<8} {'Score':>8} {'RMSD':>8} {'Status':<16}")
    click.echo("-" * 64)
    for r in results:
        rmsd_s = f"{r.rmsd:.2f}" if r.rmsd is not None else "N/A"
        score_s = f"{r.docking_score:.2f}" if r.docking_score is not None else "N/A"
        click.echo(f"{r.target_id:<10} {r.pdb_id:<6} {r.ligand_resname:<8} {score_s:>8} {rmsd_s:>8} {r.status:<16}")

    evaluated = [r for r in results if r.status in (STATUS_PASS, STATUS_FAIL)]
    skipped = [r for r in results if r.status.startswith("skipped")]
    failed = [r for r in results if r.status == STATUS_FAIL]
    errored = [r for r in results if r.status == STATUS_ERROR]
    n_pass = sum(1 for r in evaluated if r.passed)

    click.echo(f"\n{n_pass}/{len(evaluated)} evaluated targets passed")
    if skipped:
        click.echo(f"{len(skipped)} skipped: {', '.join(r.target_id + ' (' + r.status + ')' for r in skipped)}")

    # --- Exclusion summary for primary claims ---
    excluded = failed + errored
    if excluded:
        click.echo("\n--- Targets EXCLUDED from primary claims ---")
        for r in excluded:
            click.echo(
                f"  {r.target_id} (PDB {r.pdb_id}): RMSD={r.rmsd:.2f} A"
                if r.rmsd is not None
                else f"  {r.target_id} (PDB {r.pdb_id}): {r.error}"
            )
        sys.exit(1)
    else:
        click.echo("\nAll evaluated targets passed -- none excluded from primary claims.")


if __name__ == "__main__":
    main()
