"""Enrichment benchmark: validate that UniDock separates actives from decoys.

For each target, docks ChEMBL actives and compares their score distribution
to the MOSES calibration decoy set. Computes AUROC, BEDROC(alpha=20), and
enrichment factor metrics.  Targets below quality thresholds are flagged.

Blocks KVA-270 (threshold calibration) and KVA-276 (M2 milestone).

Usage::

    uv run python scripts/enrichment_benchmark.py --dry-run
    uv run python scripts/enrichment_benchmark.py --target-id BACE1 --target-id CDK2
    uv run python scripts/enrichment_benchmark.py
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import sys
from contextlib import nullcontext
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import numpy as np
from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

DOCKER_BIN = shutil.which("docker") or "docker"

# PDB ID mapping for targets missing local PDB files.
# These are downloaded from RCSB when not present on disk.
TARGET_PDB_MAP: dict[str, str] = {
    "CDK2": "1HCL",
    "ERK2": "4QTA",
    "EGFR": "1M17",
    "HER2": "3PP0",
    "AChE": "4EY7",
    "BChE": "4BDS",
    "BACE1": "6EQM",
    "GSK3B": "5KPK",
    "CDK5": "1UNL",
    "MAOB": "2V5Z",
    "COX1": "1EQG",
    "COX2": "3LN1",
}

CSV_COLUMNS = [
    "target_id",
    "pdb_id",
    "n_actives",
    "n_decoys",
    "auroc",
    "bedroc_alpha20",
    "ef_1pct",
    "ef_5pct",
    "status",
    "passed",
]


def _collect_blocked_target_ids(pairs: dict[str, Any]) -> set[str]:
    """Return target IDs belonging to blocked pairs."""
    blocked: set[str] = set()
    for pair_cfg in pairs.values():
        if pair_cfg.get("set") == "blocked":
            for role in ("target_a", "target_b"):
                blocked.add(pair_cfg[role]["id"])
    return blocked


def _collect_all_target_ids(pairs: dict[str, Any]) -> list[str]:
    """Collect unique target IDs from all pairs."""
    ids: list[str] = []
    for pair_cfg in pairs.values():
        ids.append(pair_cfg["target_a"]["id"])
        ids.append(pair_cfg["target_b"]["id"])
    return sorted(set(ids))


def _collect_target_pdb_map(pairs: dict[str, Any]) -> dict[str, str]:
    """Build target_id -> pdb_id mapping from pairs config."""
    mapping: dict[str, str] = {}
    for pair_cfg in pairs.values():
        for role in ("target_a", "target_b"):
            t = pair_cfg[role]
            mapping[t["id"]] = t["pdb_id"]
    return mapping


def _download_pdb(pdb_id: str, dest: Path) -> None:
    """Download PDB from RCSB if not already present."""
    import requests

    if dest.exists():
        return
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    logger.info("Downloading PDB {} from RCSB to {}", pdb_id, dest)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(resp.text)


def _resolve_pdb_path_or_download(
    target_id: str,
    pdb_id: str,
    receptors_dir: Path,
) -> Path:
    """Try to find PDB via generator_freedpp path resolution, fall back to RCSB download."""
    try:
        from dualdocking.generator_freedpp import _resolve_pdb_path

        return _resolve_pdb_path(target_id)
    except FileNotFoundError:
        dest = receptors_dir / f"{pdb_id.lower()}.pdb"
        _download_pdb(pdb_id, dest)
        return dest


def _load_decoy_scores(scores_path: Path) -> list[float]:
    """Load docking scores from a calibration CSV file."""
    scores: list[float] = []
    with scores_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                scores.append(float(row["score"]))
            except (ValueError, KeyError):
                continue
    return scores


def _load_active_smiles(
    target_id: str,
    cache_dir: Path,
    max_actives: int,
    seed: int,
) -> list[str]:
    """Load and sample ChEMBL actives for a target."""
    from dualdocking.chembl_validation import fetch_actives

    all_actives = fetch_actives(target_id, cache_dir=cache_dir)
    if len(all_actives) <= max_actives:
        return all_actives
    rng = random.Random(seed)  # noqa: S311
    return rng.sample(all_actives, max_actives)


def _dock_smiles_list(
    runner: Any,
    binary: str,
    receptor_path: Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    smiles_list: list[str],
    params: Any,
    work_dir: Path,
    seed: int,
    batch_size: int,
) -> tuple[list[float], int]:
    """Dock a list of SMILES and return (scores, n_failures)."""
    from dock_calibration import (
        _dock_batch_unidock,
        _ensure_ligand_pdbqt,
        _safe_unlink,
    )

    ligands_dir = work_dir / "ligands"
    ligands_dir.mkdir(parents=True, exist_ok=True)
    output_dir = work_dir / "dock_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    ligand_paths: list[Path] = []
    prep_failures = 0
    for idx, smi in enumerate(tqdm(smiles_list, desc="Prep ligands", unit="lig")):
        sdf_path = ligands_dir / f"active_{idx:05d}.sdf"
        lig_path = ligands_dir / f"active_{idx:05d}.pdbqt"
        ok, err = _ensure_ligand_pdbqt(
            runner, smi, sdf_path, lig_path,
            timeout=params.timeout_seconds, seed=seed + idx,
        )
        if not ok:
            prep_failures += 1
            logger.debug("Ligand prep failed: {} ({})", smi[:60], err)
            continue
        ligand_paths.append(lig_path)

    if prep_failures:
        logger.warning("{} ligand prep failures out of {}", prep_failures, len(smiles_list))

    scores: list[float] = []
    dock_failures = 0
    use_batch = "unidock" in Path(binary).name.lower()

    if use_batch:
        for batch_id, start in enumerate(
            tqdm(range(0, len(ligand_paths), batch_size), desc="Dock", unit="batch")
        ):
            batch = ligand_paths[start : start + batch_size]
            try:
                batch_scores = _dock_batch_unidock(
                    runner, binary, receptor_path, batch, center, size,
                    params, output_dir, batch_id=batch_id,
                    seed=seed + start, timeout=params.timeout_seconds,
                )
            except Exception as exc:
                dock_failures += len(batch)
                logger.error("Batch docking failed: {}", exc)
                continue
            for lig in batch:
                score = batch_scores.get(lig)
                if score is None:
                    dock_failures += 1
                else:
                    scores.append(score)
                _safe_unlink(lig, runner)
    else:
        from dock_calibration import _dock_ligand

        for idx, lig_path in enumerate(tqdm(ligand_paths, desc="Dock", unit="lig")):
            log_path = output_dir / f"active_{idx:05d}.log"
            out_path = output_dir / f"active_{idx:05d}.pdbqt"
            try:
                score = _dock_ligand(
                    runner, binary, receptor_path, lig_path,
                    center, size, params, log_path, out_path,
                )
                scores.append(score)
            except Exception as exc:
                dock_failures += 1
                logger.error("Docking failed for #{}: {}", idx, exc)
            finally:
                for p in (log_path, out_path, lig_path):
                    if p.exists():
                        _safe_unlink(p, runner)

    return scores, prep_failures + dock_failures


def _save_active_scores(scores: list[float], smiles: list[str], path: Path) -> None:
    """Save active docking scores to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "score"])
        for smi, score in zip(smiles, scores):
            writer.writerow([smi, f"{score:.6f}"])


def _make_roc_figure(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    target_id: str,
    out_path: Path,
) -> None:
    """Save an ROC curve figure."""
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping ROC figure for {}", target_id)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"AUROC = {auroc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {target_id}")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _make_summary_figure(
    results: list[dict[str, Any]],
    auroc_threshold: float,
    bedroc_threshold: float,
    out_path: Path,
) -> None:
    """Save a summary bar chart of AUROC + BEDROC with threshold lines."""
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping summary figure")
        return

    evaluated = [r for r in results if r["status"] in ("pass", "fail")]
    if not evaluated:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    targets = [r["target_id"] for r in evaluated]
    aurocs = [r["auroc"] for r in evaluated]
    bedrocs = [r["bedroc_alpha20"] for r in evaluated]

    x = np.arange(len(targets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(targets) * 1.2), 5))
    ax.bar(x - width / 2, aurocs, width, label="AUROC", color="steelblue")
    ax.bar(x + width / 2, bedrocs, width, label="BEDROC(α=20)", color="darkorange")
    ax.axhline(y=auroc_threshold, color="steelblue", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(y=bedroc_threshold, color="darkorange", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Enrichment Benchmark Summary")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_or_dock_decoys(  # noqa: PLR0913
    tid: str,
    pdb_id: str,
    runner: Any,
    binary: str,
    params: Any,
    cal_cfg: dict[str, Any],
    seed: int,
    batch_size: int,
    receptors_dir: Path,
    calibration_scores_dir: Path,
    skip_docking: bool,
) -> list[float] | str:
    """Load existing decoy scores or dock calibration set.  Returns scores or error string."""
    from dualdocking.binding_box import extract_binding_box

    decoy_scores_path = calibration_scores_dir / f"{tid}.csv"
    if decoy_scores_path.exists():
        scores = _load_decoy_scores(decoy_scores_path)
        logger.info("Loaded {} decoy scores for {} from calibration", len(scores), tid)
        return scores

    if skip_docking:
        return f"No calibration scores for {tid} and --skip-docking set"

    from dock_calibration import (
        _ensure_receptor_pdbqt,
        _load_calibration_smiles,
        _parse_vec,
        _sample_smiles,
    )

    logger.info("Docking calibration set for {} (no existing scores)", tid)
    pdb_path = _resolve_pdb_path_or_download(tid, pdb_id, receptors_dir)
    receptor_path = receptors_dir / f"{tid}.pdbqt"
    _ensure_receptor_pdbqt(runner, pdb_path, receptor_path, timeout=params.timeout_seconds)

    center_str, size_str = extract_binding_box(pdb_path, padding=params.box_padding)
    center = _parse_vec(center_str)
    size = _parse_vec(size_str)

    cal_path = PROJECT_ROOT / cal_cfg.get("output_path", "data/calibration/zinc_druglike_calibration_1000_v2024.smi")
    all_smiles = _load_calibration_smiles(cal_path)
    sample = _sample_smiles(all_smiles, 1000, seed)

    work_dir = PROJECT_ROOT / "data" / "enrichment" / "work" / tid / "decoys"
    scores, _n_fail = _dock_smiles_list(
        runner, binary, receptor_path, center, size,
        sample, params, work_dir, seed, batch_size,
    )

    decoy_scores_path.parent.mkdir(parents=True, exist_ok=True)
    with decoy_scores_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["score"])
        for s in scores:
            writer.writerow([f"{s:.6f}"])
    logger.info("Saved {} decoy scores for {}", len(scores), tid)
    return scores


def _load_or_dock_actives(  # noqa: PLR0913
    tid: str,
    pdb_id: str,
    runner: Any,
    binary: str,
    params: Any,
    seed: int,
    batch_size: int,
    max_actives: int,
    chembl_dir: Path,
    scores_dir: Path,
    receptors_dir: Path,
    skip_docking: bool,
) -> list[float] | str:
    """Load existing active scores or dock ChEMBL actives.  Returns scores or error string."""
    from dualdocking.binding_box import extract_binding_box

    active_scores_path = scores_dir / f"{tid}_actives.csv"
    active_smiles = _load_active_smiles(tid, chembl_dir, max_actives, seed)

    if active_scores_path.exists() and skip_docking:
        data: list[float] = []
        with active_scores_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    data.append(float(row["score"]))
                except (ValueError, KeyError):
                    continue
        logger.info("Loaded {} active scores for {} from cache", len(data), tid)
        return data

    if skip_docking:
        return f"No active scores for {tid} and --skip-docking set"

    from dock_calibration import _ensure_receptor_pdbqt, _parse_vec

    pdb_path = _resolve_pdb_path_or_download(tid, pdb_id, receptors_dir)
    receptor_path = receptors_dir / f"{tid}.pdbqt"
    _ensure_receptor_pdbqt(runner, pdb_path, receptor_path, timeout=params.timeout_seconds)

    center_str, size_str = extract_binding_box(pdb_path, padding=params.box_padding)
    center = _parse_vec(center_str)
    size = _parse_vec(size_str)

    work_dir = PROJECT_ROOT / "data" / "enrichment" / "work" / tid / "actives"
    scores, _n_fail = _dock_smiles_list(
        runner, binary, receptor_path, center, size,
        active_smiles, params, work_dir, seed, batch_size,
    )

    success_rate = len(scores) / len(active_smiles) if active_smiles else 0
    if success_rate < 0.80:
        logger.warning("Active docking success rate {:.1%} < 80% for {}", success_rate, tid)

    _save_active_scores(scores, active_smiles[:len(scores)], active_scores_path)
    logger.info("Saved {} active scores for {}", len(scores), tid)
    return scores


def _print_summary(results: list[Any], auroc_threshold: float, bedroc_threshold: float, alpha: float) -> None:
    """Print enrichment benchmark summary table."""
    click.echo("\n--- Enrichment Benchmark Summary ---")
    click.echo(f"Thresholds: AUROC>{auroc_threshold}, BEDROC(α={alpha:.0f})>{bedroc_threshold}")
    click.echo(
        f"{'Target':<10} {'n_act':>6} {'n_dec':>6} {'AUROC':>7} {'BEDROC':>7} "
        f"{'EF@1%':>7} {'EF@5%':>7} {'Status':<16}"
    )
    click.echo("-" * 76)
    for r in results:
        show_metrics = r.status not in ("error", "skipped:blocked")
        auroc_s = f"{r.auroc:.3f}" if show_metrics else "N/A"
        bedroc_s = f"{r.bedroc_alpha20:.3f}" if show_metrics else "N/A"
        ef1_s = f"{r.ef_1pct:.1f}" if show_metrics else "N/A"
        ef5_s = f"{r.ef_5pct:.1f}" if show_metrics else "N/A"
        click.echo(
            f"{r.target_id:<10} {r.n_actives:>6} {r.n_decoys:>6} {auroc_s:>7} "
            f"{bedroc_s:>7} {ef1_s:>7} {ef5_s:>7} {r.status:<16}"
        )

    evaluated = [r for r in results if r.status in ("pass", "fail")]
    n_pass = sum(1 for r in evaluated if r.passed)
    click.echo(f"\n{n_pass}/{len(evaluated)} targets passed enrichment thresholds")

    excluded = [r for r in results if r.status == "fail"]
    if excluded:
        click.echo("\n--- Targets below enrichment threshold ---")
        for r in excluded:
            click.echo(f"  {r.target_id}: AUROC={r.auroc:.3f} BEDROC={r.bedroc_alpha20:.3f}")


@click.command()
@click.option("--calibration-config", default="configs/calibration.yaml")
@click.option("--metrics-config", default="configs/metrics.yaml")
@click.option("--target-id", multiple=True, help="Specific target IDs (repeatable)")
@click.option("--max-actives", default=200, type=int, help="Max actives per target")
@click.option("--seed", default=137, type=int, help="Sampling seed")
@click.option("--docker-image", default="dualdocking/freedpp:latest")
@click.option("--gpus", default="all", help="Docker --gpus value")
@click.option("--output", default="data/enrichment/enrichment_results.json")
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

    from dualdocking.config import load_pairs
    from dualdocking.enrichment import (
        EnrichmentResult,
        compute_auroc,
        compute_bedroc,
        compute_enrichment_factor,
        compute_roc_curve,
    )

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

    chembl_dir = PROJECT_ROOT / "data" / "chembl"
    scores_dir = PROJECT_ROOT / "data" / "enrichment" / "scores"
    receptors_dir = PROJECT_ROOT / "data" / "enrichment" / "receptors"
    calibration_scores_dir = PROJECT_ROOT / "data" / "calibration" / "scores"
    figures_path = PROJECT_ROOT / figures_dir

    scores_dir.mkdir(parents=True, exist_ok=True)
    receptors_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        _dry_run_report(all_target_ids, blocked_ids, target_pdb_map, chembl_dir,
                        calibration_scores_dir, scores_dir, max_actives, seed,
                        auroc_threshold, bedroc_threshold)
        return

    from dock_calibration import DockerRunner, DockingParams

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
        results.append(EnrichmentResult(
            target_id=tid, n_actives=0, n_decoys=0,
            auroc=0.0, bedroc_alpha20=0.0, ef_1pct=0.0, ef_5pct=0.0,
            passed=False, status="skipped:blocked",
            error="Blocked pair (non-human species)",
        ))
        logger.info("[SKIP] {} — blocked pair", tid)

    active_targets = [t for t in all_target_ids if t not in blocked_ids]
    need_docker = not skip_docking and any(
        not (scores_dir / f"{tid}_actives.csv").exists()
        or not (calibration_scores_dir / f"{tid}.csv").exists()
        for tid in active_targets
    )

    if need_docker:
        from dualdocking.generator_freedpp import _ensure_docker_image
        _ensure_docker_image(docker_image)

    runner_ctx: Any = DockerRunner(docker_image, PROJECT_ROOT, gpus=gpus) if need_docker else nullcontext(None)
    with runner_ctx as runner:
        for tid in active_targets:
            result = _process_target(
                tid, target_pdb_map, runner, binary, params, cal_cfg, seed, batch_size,
                max_actives, alpha, auroc_threshold, bedroc_threshold,
                chembl_dir, scores_dir, receptors_dir, calibration_scores_dir,
                figures_path, skip_docking,
                compute_auroc, compute_bedroc, compute_enrichment_factor,
                compute_roc_curve, EnrichmentResult,
            )
            results.append(result)

    results.sort(key=lambda r: r.target_id)
    _write_outputs(results, target_pdb_map, output, csv_output, figures_path, seed,
                   max_actives, auroc_threshold, bedroc_threshold, alpha)
    _print_summary(results, auroc_threshold, bedroc_threshold, alpha)


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
    """Print dry-run target summary."""
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
        decoy_path = calibration_scores_dir / f"{tid}.csv"
        n_decoys = len(_load_decoy_scores(decoy_path)) if decoy_path.exists() else 0
        has_active_scores = (scores_dir / f"{tid}_actives.csv").exists()
        click.echo(
            f"  {tid:<10} PDB={pdb_id:<5} actives={n_actives:>4} "
            f"decoys={n_decoys:>5} active_scores={'YES' if has_active_scores else 'NO'}"
            f"{blocked}"
        )


def _process_target(  # noqa: PLR0913
    tid: str,
    target_pdb_map: dict[str, str],
    runner: Any,
    binary: str,
    params: Any,
    cal_cfg: dict[str, Any],
    seed: int,
    batch_size: int,
    max_actives: int,
    alpha: float,
    auroc_threshold: float,
    bedroc_threshold: float,
    chembl_dir: Path,
    scores_dir: Path,
    receptors_dir: Path,
    calibration_scores_dir: Path,
    figures_path: Path,
    skip_docking: bool,
    compute_auroc_fn: Any,
    compute_bedroc_fn: Any,
    compute_ef_fn: Any,
    compute_roc_fn: Any,
    result_cls: type,
) -> Any:
    """Process a single target: load/dock scores, compute metrics."""
    pdb_id = target_pdb_map.get(tid, TARGET_PDB_MAP.get(tid, ""))
    err_result = lambda msg: result_cls(  # noqa: E731
        target_id=tid, n_actives=0, n_decoys=0,
        auroc=0.0, bedroc_alpha20=0.0, ef_1pct=0.0, ef_5pct=0.0,
        passed=False, status="error", error=msg,
    )
    if not pdb_id:
        return err_result(f"No PDB ID for target {tid}")

    logger.info("Processing target {}", tid)

    decoy_result = _load_or_dock_decoys(
        tid, pdb_id, runner, binary, params, cal_cfg, seed, batch_size,
        receptors_dir, calibration_scores_dir, skip_docking,
    )
    if isinstance(decoy_result, str):
        return err_result(decoy_result)
    decoy_scores = decoy_result
    if not decoy_scores:
        return err_result(f"No decoy scores available for {tid}")

    active_result = _load_or_dock_actives(
        tid, pdb_id, runner, binary, params, seed, batch_size,
        max_actives, chembl_dir, scores_dir, receptors_dir, skip_docking,
    )
    if isinstance(active_result, str):
        return result_cls(
            target_id=tid, n_actives=0, n_decoys=len(decoy_scores),
            auroc=0.0, bedroc_alpha20=0.0, ef_1pct=0.0, ef_5pct=0.0,
            passed=False, status="error", error=active_result,
        )
    active_scores = active_result
    if len(active_scores) < 20:
        return result_cls(
            target_id=tid, n_actives=len(active_scores), n_decoys=len(decoy_scores),
            auroc=0.0, bedroc_alpha20=0.0, ef_1pct=0.0, ef_5pct=0.0,
            passed=False, status="error",
            error=f"Too few valid active scores ({len(active_scores)} < 20)",
        )

    all_scores = np.array(active_scores + decoy_scores)
    labels = np.array([1] * len(active_scores) + [0] * len(decoy_scores))

    try:
        auroc = compute_auroc_fn(all_scores, labels, lower_is_better=True)
        bedroc = compute_bedroc_fn(all_scores, labels, alpha=alpha, lower_is_better=True)
        ef_1 = compute_ef_fn(all_scores, labels, fraction=0.01, lower_is_better=True)
        ef_5 = compute_ef_fn(all_scores, labels, fraction=0.05, lower_is_better=True)
        fpr, tpr = compute_roc_fn(all_scores, labels, lower_is_better=True)
    except Exception as exc:
        return result_cls(
            target_id=tid, n_actives=len(active_scores), n_decoys=len(decoy_scores),
            auroc=0.0, bedroc_alpha20=0.0, ef_1pct=0.0, ef_5pct=0.0,
            passed=False, status="error", error=f"Metric computation failed: {exc}",
        )

    passed = auroc >= auroc_threshold and bedroc >= bedroc_threshold
    status = "pass" if passed else "fail"

    logger.info(
        "[{}] {} AUROC={:.3f} BEDROC={:.3f} EF@1%={:.1f} (n_act={}, n_dec={})",
        status.upper(), tid, auroc, bedroc, ef_1, len(active_scores), len(decoy_scores),
    )
    _make_roc_figure(fpr, tpr, auroc, tid, figures_path / f"roc_{tid}.png")

    return result_cls(
        target_id=tid, n_actives=len(active_scores), n_decoys=len(decoy_scores),
        auroc=auroc, bedroc_alpha20=bedroc, ef_1pct=ef_1, ef_5pct=ef_5,
        passed=passed, status=status,
    )


def _write_outputs(
    results: list[Any],
    target_pdb_map: dict[str, str],
    output: str,
    csv_output: str,
    figures_path: Path,
    seed: int,
    max_actives: int,
    auroc_threshold: float,
    bedroc_threshold: float,
    alpha: float,
) -> None:
    """Write JSON, CSV, and summary figure outputs."""
    json_path = PROJECT_ROOT / output
    json_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "date": datetime.now(UTC).date().isoformat(),
        "seed": seed,
        "max_actives": max_actives,
        "auroc_threshold": auroc_threshold,
        "bedroc_threshold": bedroc_threshold,
        "alpha": alpha,
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(manifest, indent=2))
    logger.info("JSON results saved to {}", json_path)

    csv_path = PROJECT_ROOT / csv_output
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            d = asdict(r)
            d["pdb_id"] = target_pdb_map.get(r.target_id, TARGET_PDB_MAP.get(r.target_id, ""))
            writer.writerow({k: d.get(k, "") for k in CSV_COLUMNS})
    logger.info("CSV results saved to {}", csv_path)

    _make_summary_figure(
        [asdict(r) for r in results],
        auroc_threshold, bedroc_threshold,
        figures_path / "enrichment_summary.png",
    )


if __name__ == "__main__":
    main()
