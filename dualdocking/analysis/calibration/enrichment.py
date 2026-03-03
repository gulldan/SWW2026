"""Enrichment benchmark: validate that UniDock separates actives from decoys.

For each target, docks ChEMBL actives and compares their score distribution
to the MOSES calibration decoy set. Computes AUROC, BEDROC(alpha=20), and
enrichment factor metrics. Targets below quality thresholds are flagged.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from tqdm import tqdm

from dualdocking.data.http_utils import get_with_retry
from dualdocking.scoring.docking import (
    ensure_ligand_pdbqt,
    ensure_receptor_pdbqt,
    parse_vec,
)

CSV_COLUMNS: list[str] = [
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


def download_pdb(pdb_id: str, dest: Path) -> None:
    """Download a PDB file from RCSB if not already present.

    Args:
        pdb_id: Four-character PDB accession code.
        dest: Local destination path.
    """
    if dest.exists():
        return
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    logger.info("Downloading PDB {} from RCSB to {}", pdb_id, dest)
    resp = get_with_retry(url, timeout=60)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(resp.text)


def resolve_pdb_path_or_download(
    target_id: str,
    pdb_id: str,
    receptors_dir: Path,
) -> Path:
    """Resolve PDB path via the generator module, falling back to RCSB download.

    Args:
        target_id: Internal target identifier.
        pdb_id: PDB accession code.
        receptors_dir: Directory for downloaded PDB files.

    Returns:
        Path to the PDB file (existing or freshly downloaded).
    """
    try:
        from dualdocking.generation.freedpp import _resolve_pdb_path

        return _resolve_pdb_path(target_id)
    except FileNotFoundError:
        dest = receptors_dir / f"{pdb_id.lower()}.pdb"
        download_pdb(pdb_id, dest)
        return dest


def load_decoy_scores(scores_stem: Path) -> list[float]:
    """Load docking scores from a calibration Parquet or CSV file.

    Args:
        scores_stem: Path without extension (e.g. ``calibration/scores/CDK2``).

    Returns:
        List of parsed docking scores.
    """
    from dualdocking.data.io_utils import read_df

    try:
        df = read_df(scores_stem)
    except FileNotFoundError:
        return []
    return [float(v) for v in df["score"].dropna()]


def load_active_smiles(
    target_id: str,
    cache_dir: Path,
    max_actives: int,
    seed: int,
) -> list[str]:
    """Load and sample ChEMBL actives for a target.

    Args:
        target_id: Internal target identifier.
        cache_dir: Directory for cached ChEMBL actives.
        max_actives: Maximum number of actives to return.
        seed: Random seed for subsampling.

    Returns:
        List of active SMILES strings (at most *max_actives*).
    """
    from dualdocking.data.chembl_validation import fetch_actives

    all_actives = fetch_actives(target_id, cache_dir=cache_dir)
    if len(all_actives) <= max_actives:
        return all_actives
    rng = random.Random(seed)  # noqa: S311
    return rng.sample(all_actives, max_actives)


def dock_smiles_list(  # noqa: PLR0913
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
    project_root: Path,
) -> tuple[list[float], int]:
    """Dock a list of SMILES and return scores with failure count.

    Args:
        runner: Docker runner (or None for local execution).
        binary: Docking binary name.
        receptor_path: Receptor PDBQT file.
        center: Binding box center (x, y, z).
        size: Binding box size (x, y, z).
        smiles_list: SMILES strings to dock.
        params: Docking parameters (``DockingParams``).
        work_dir: Working directory for intermediate files.
        seed: Random seed for 3D embedding.
        batch_size: UniDock batch size.
        project_root: Project root directory.

    Returns:
        Tuple of (score_list, total_failures).
    """
    ligands_dir = work_dir / "ligands"
    ligands_dir.mkdir(parents=True, exist_ok=True)
    output_dir = work_dir / "dock_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    ligand_paths: list[Path] = []
    prep_failures = 0
    for idx, smi in enumerate(tqdm(smiles_list, desc="Prep ligands", unit="lig")):
        sdf_path = ligands_dir / f"active_{idx:05d}.sdf"
        lig_path = ligands_dir / f"active_{idx:05d}.pdbqt"
        ok, err = ensure_ligand_pdbqt(
            runner,
            smi,
            sdf_path,
            lig_path,
            timeout=params.timeout_seconds,
            seed=seed + idx,
            project_root=project_root,
        )
        if not ok:
            prep_failures += 1
            logger.debug("Ligand prep failed: {} ({})", smi[:60], err)
            continue
        ligand_paths.append(lig_path)

    if prep_failures:
        logger.warning("{} ligand prep failures out of {}", prep_failures, len(smiles_list))

    from dualdocking.analysis.calibration.docking import dock_batch_ligands, dock_sequential_ligands

    scores: list[float]
    dock_failures: int
    use_batch = "unidock" in Path(binary).name.lower()

    target_label = receptor_path.stem or "target"
    if use_batch:
        scores, dock_failures = dock_batch_ligands(
            runner,
            binary,
            receptor_path,
            ligand_paths,
            center,
            size,
            params,
            output_dir,
            target_label,
            batch_size,
            seed,
            project_root,
            cleanup_ligands=True,
        )
    else:
        scores, dock_failures = dock_sequential_ligands(
            runner,
            binary,
            receptor_path,
            ligand_paths,
            center,
            size,
            params,
            output_dir,
            target_label,
            dry_run=False,
            project_root=project_root,
            cleanup_ligands=True,
        )

    return scores, prep_failures + dock_failures


def save_active_scores(scores: list[float], smiles: list[str], stem: Path) -> None:
    """Save active docking scores to Parquet (with CSV fallback).

    Args:
        scores: Docking scores.
        smiles: Corresponding SMILES strings.
        stem: Output path without extension.
    """
    import pandas as pd

    from dualdocking.data.io_utils import write_df

    df = pd.DataFrame({"smiles": smiles, "score": scores})
    write_df(df, stem, csv_too=True)


def make_roc_figure(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    target_id: str,
    out_path: Path,
) -> None:
    """Save an ROC curve figure as PNG.

    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        auroc: Area under the ROC curve.
        target_id: Target identifier (used in the title).
        out_path: Output PNG path.
    """
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


def make_summary_figure(
    results: list[dict[str, Any]],
    auroc_threshold: float,
    bedroc_threshold: float,
    out_path: Path,
) -> None:
    """Save a summary bar chart of AUROC and BEDROC with threshold lines.

    Args:
        results: List of result dicts.
        auroc_threshold: AUROC pass threshold (drawn as dashed line).
        bedroc_threshold: BEDROC pass threshold (drawn as dashed line).
        out_path: Output PNG path.
    """
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
    ax.bar(x + width / 2, bedrocs, width, label="BEDROC(a=20)", color="darkorange")
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


def load_or_dock_decoys(  # noqa: PLR0913
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
    project_root: Path,
) -> list[float] | str:
    """Load existing decoy scores or dock the calibration set.

    Args:
        tid: Target identifier.
        pdb_id: PDB accession code.
        runner: Docker runner (or None).
        binary: Docking binary.
        params: Docking parameters.
        cal_cfg: Calibration YAML config dict.
        seed: Random seed.
        batch_size: UniDock batch size.
        receptors_dir: Directory for receptor files.
        calibration_scores_dir: Directory with existing calibration scores.
        skip_docking: If True, return an error string when scores are missing.
        project_root: Project root directory.

    Returns:
        List of decoy scores, or an error string on failure.
    """
    from dualdocking.scoring.binding_box import extract_binding_box

    decoy_scores_stem = calibration_scores_dir / tid
    from dualdocking.data.io_utils import stem_exists

    if stem_exists(decoy_scores_stem):
        scores = load_decoy_scores(decoy_scores_stem)
        logger.info("Loaded {} decoy scores for {} from calibration", len(scores), tid)
        return scores

    if skip_docking:
        return f"No calibration scores for {tid} and --skip-docking set"

    from dualdocking.analysis.calibration.docking import (
        load_calibration_smiles,
        sample_smiles,
    )

    logger.info("Docking calibration set for {} (no existing scores)", tid)
    pdb_path = resolve_pdb_path_or_download(tid, pdb_id, receptors_dir)
    receptor_path = receptors_dir / f"{tid}.pdbqt"
    ensure_receptor_pdbqt(runner, pdb_path, receptor_path, timeout=params.timeout_seconds, project_root=project_root)

    center_str, size_str = extract_binding_box(pdb_path, padding=params.box_padding)
    center = parse_vec(center_str)
    size = parse_vec(size_str)

    cal_path = project_root / cal_cfg.get("output_path", "output/data/calibration/zinc_druglike_calibration_1000_v2024.smi")
    all_smiles = load_calibration_smiles(cal_path)
    sample = sample_smiles(all_smiles, 1000, seed)

    work_dir = project_root / "output" / "data" / "enrichment" / "work" / tid / "decoys"
    scores, _n_fail = dock_smiles_list(
        runner,
        binary,
        receptor_path,
        center,
        size,
        sample,
        params,
        work_dir,
        seed,
        batch_size,
        project_root,
    )

    import pandas as pd

    from dualdocking.data.io_utils import write_df

    df = pd.DataFrame({"score": scores})
    write_df(df, decoy_scores_stem, csv_too=True)
    logger.info("Saved {} decoy scores for {}", len(scores), tid)
    return scores


def load_or_dock_actives(  # noqa: PLR0913
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
    project_root: Path,
) -> list[float] | str:
    """Load existing active scores or dock ChEMBL actives.

    Args:
        tid: Target identifier.
        pdb_id: PDB accession code.
        runner: Docker runner (or None).
        binary: Docking binary.
        params: Docking parameters.
        seed: Random seed.
        batch_size: UniDock batch size.
        max_actives: Maximum number of actives to dock.
        chembl_dir: Directory with cached ChEMBL actives.
        scores_dir: Directory for output active score CSVs.
        receptors_dir: Directory for receptor files.
        skip_docking: If True, return an error string when scores are missing.
        project_root: Project root directory.

    Returns:
        List of active scores, or an error string on failure.
    """
    from dualdocking.scoring.binding_box import extract_binding_box

    active_scores_stem = scores_dir / f"{tid}_actives"
    active_smiles = load_active_smiles(tid, chembl_dir, max_actives, seed)

    from dualdocking.data.io_utils import read_df, stem_exists

    if stem_exists(active_scores_stem) and skip_docking:
        try:
            df = read_df(active_scores_stem)
            data: list[float] = [float(v) for v in df["score"].dropna()]
            logger.info("Loaded {} active scores for {} from cache", len(data), tid)
            return data
        except Exception:
            logger.debug("Failed to load cached active scores for {}, will re-dock", tid)

    if skip_docking:
        return f"No active scores for {tid} and --skip-docking set"

    pdb_path = resolve_pdb_path_or_download(tid, pdb_id, receptors_dir)
    receptor_path = receptors_dir / f"{tid}.pdbqt"
    ensure_receptor_pdbqt(runner, pdb_path, receptor_path, timeout=params.timeout_seconds, project_root=project_root)

    center_str, size_str = extract_binding_box(pdb_path, padding=params.box_padding)
    center = parse_vec(center_str)
    size = parse_vec(size_str)

    work_dir = project_root / "output" / "data" / "enrichment" / "work" / tid / "actives"
    scores, _n_fail = dock_smiles_list(
        runner,
        binary,
        receptor_path,
        center,
        size,
        active_smiles,
        params,
        work_dir,
        seed,
        batch_size,
        project_root,
    )

    success_rate = len(scores) / len(active_smiles) if active_smiles else 0
    if success_rate < 0.80:
        logger.warning("Active docking success rate {:.1%} < 80% for {}", success_rate, tid)

    save_active_scores(scores, active_smiles[: len(scores)], active_scores_stem)
    logger.info("Saved {} active scores for {}", len(scores), tid)
    return scores


def process_target(  # noqa: PLR0913
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
    project_root: Path,
) -> Any:
    """Process a single target: load/dock scores, compute metrics.

    Args:
        tid: Target identifier.
        target_pdb_map: Mapping from target ID to PDB ID.
        runner: Docker runner (or None).
        binary: Docking binary.
        params: Docking parameters.
        cal_cfg: Calibration config dict.
        seed: Random seed.
        batch_size: UniDock batch size.
        max_actives: Maximum actives to dock.
        alpha: BEDROC alpha parameter.
        auroc_threshold: AUROC pass threshold.
        bedroc_threshold: BEDROC pass threshold.
        chembl_dir: ChEMBL actives cache directory.
        scores_dir: Enrichment scores directory.
        receptors_dir: Receptor files directory.
        calibration_scores_dir: Calibration scores directory.
        figures_path: Figures output directory.
        skip_docking: Only use existing score files.
        project_root: Project root directory.

    Returns:
        An ``EnrichmentResult`` instance.
    """
    from dualdocking.analysis.enrichment import (
        EnrichmentResult,
        compute_auroc,
        compute_bedroc,
        compute_enrichment_factor,
        compute_roc_curve,
    )

    pdb_id = target_pdb_map.get(tid, "")
    err_result = lambda msg: EnrichmentResult(  # noqa: E731
        target_id=tid,
        n_actives=0,
        n_decoys=0,
        auroc=0.0,
        bedroc_alpha20=0.0,
        ef_1pct=0.0,
        ef_5pct=0.0,
        passed=False,
        status="error",
        error=msg,
    )
    if not pdb_id:
        return err_result(f"No PDB ID for target {tid}")

    logger.info("Processing target {}", tid)

    decoy_result = load_or_dock_decoys(
        tid,
        pdb_id,
        runner,
        binary,
        params,
        cal_cfg,
        seed,
        batch_size,
        receptors_dir,
        calibration_scores_dir,
        skip_docking,
        project_root,
    )
    if isinstance(decoy_result, str):
        return err_result(decoy_result)
    decoy_scores = decoy_result
    if not decoy_scores:
        return err_result(f"No decoy scores available for {tid}")

    active_result = load_or_dock_actives(
        tid,
        pdb_id,
        runner,
        binary,
        params,
        seed,
        batch_size,
        max_actives,
        chembl_dir,
        scores_dir,
        receptors_dir,
        skip_docking,
        project_root,
    )
    if isinstance(active_result, str):
        return EnrichmentResult(
            target_id=tid,
            n_actives=0,
            n_decoys=len(decoy_scores),
            auroc=0.0,
            bedroc_alpha20=0.0,
            ef_1pct=0.0,
            ef_5pct=0.0,
            passed=False,
            status="error",
            error=active_result,
        )
    active_scores = active_result
    if len(active_scores) < 20:
        return EnrichmentResult(
            target_id=tid,
            n_actives=len(active_scores),
            n_decoys=len(decoy_scores),
            auroc=0.0,
            bedroc_alpha20=0.0,
            ef_1pct=0.0,
            ef_5pct=0.0,
            passed=False,
            status="error",
            error=f"Too few valid active scores ({len(active_scores)} < 20)",
        )

    all_scores = np.array(active_scores + decoy_scores)
    labels = np.array([1] * len(active_scores) + [0] * len(decoy_scores))

    try:
        auroc = compute_auroc(all_scores, labels, lower_is_better=True)
        bedroc = compute_bedroc(all_scores, labels, alpha=alpha, lower_is_better=True)
        ef_1 = compute_enrichment_factor(all_scores, labels, fraction=0.01, lower_is_better=True)
        ef_5 = compute_enrichment_factor(all_scores, labels, fraction=0.05, lower_is_better=True)
        fpr, tpr = compute_roc_curve(all_scores, labels, lower_is_better=True)
    except Exception as exc:
        return EnrichmentResult(
            target_id=tid,
            n_actives=len(active_scores),
            n_decoys=len(decoy_scores),
            auroc=0.0,
            bedroc_alpha20=0.0,
            ef_1pct=0.0,
            ef_5pct=0.0,
            passed=False,
            status="error",
            error=f"Metric computation failed: {exc}",
        )

    passed = auroc >= auroc_threshold and bedroc >= bedroc_threshold
    status = "pass" if passed else "fail"

    logger.info(
        "[{}] {} AUROC={:.3f} BEDROC={:.3f} EF@1%={:.1f} (n_act={}, n_dec={})",
        status.upper(),
        tid,
        auroc,
        bedroc,
        ef_1,
        len(active_scores),
        len(decoy_scores),
    )
    make_roc_figure(fpr, tpr, auroc, tid, figures_path / f"roc_{tid}.png")

    return EnrichmentResult(
        target_id=tid,
        n_actives=len(active_scores),
        n_decoys=len(decoy_scores),
        auroc=auroc,
        bedroc_alpha20=bedroc,
        ef_1pct=ef_1,
        ef_5pct=ef_5,
        passed=passed,
        status=status,
    )


def write_outputs(  # noqa: PLR0913
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
    project_root: Path,
) -> None:
    """Write JSON, CSV, and summary figure outputs.

    Args:
        results: List of ``EnrichmentResult`` objects.
        target_pdb_map: Mapping from target ID to PDB ID.
        output: JSON output path (relative to project_root).
        csv_output: CSV output path (relative to project_root).
        figures_path: Directory for figure outputs.
        seed: Random seed used.
        max_actives: Maximum actives per target.
        auroc_threshold: AUROC pass threshold.
        bedroc_threshold: BEDROC pass threshold.
        alpha: BEDROC alpha parameter.
        project_root: Project root directory.
    """
    json_path = project_root / output
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

    import pandas as pd

    from dualdocking.data.io_utils import write_df

    rows = []
    for r in results:
        d = asdict(r)
        d["pdb_id"] = target_pdb_map.get(r.target_id, "")
        rows.append({k: d.get(k, "") for k in CSV_COLUMNS})
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    csv_stem = (project_root / csv_output).with_suffix("")
    write_df(df, csv_stem, csv_too=True)
    logger.info("Results saved to {}.parquet (+.csv)", csv_stem)

    make_summary_figure(
        [asdict(r) for r in results],
        auroc_threshold,
        bedroc_threshold,
        figures_path / "enrichment_summary.png",
    )


def print_summary(results: list[Any], auroc_threshold: float, bedroc_threshold: float, alpha: float) -> None:
    """Print the enrichment benchmark summary table to stdout.

    Args:
        results: List of ``EnrichmentResult`` objects.
        auroc_threshold: AUROC pass threshold.
        bedroc_threshold: BEDROC pass threshold.
        alpha: BEDROC alpha parameter.
    """
    import click

    click.echo("\n--- Enrichment Benchmark Summary ---")
    click.echo(f"Thresholds: AUROC>{auroc_threshold}, BEDROC(a={alpha:.0f})>{bedroc_threshold}")
    click.echo(f"{'Target':<10} {'n_act':>6} {'n_dec':>6} {'AUROC':>7} {'BEDROC':>7} {'EF@1%':>7} {'EF@5%':>7} {'Status':<16}")
    click.echo("-" * 76)
    for r in results:
        show_metrics = r.status not in ("error", "skipped:blocked")
        auroc_s = f"{r.auroc:.3f}" if show_metrics else "N/A"
        bedroc_s = f"{r.bedroc_alpha20:.3f}" if show_metrics else "N/A"
        ef1_s = f"{r.ef_1pct:.1f}" if show_metrics else "N/A"
        ef5_s = f"{r.ef_5pct:.1f}" if show_metrics else "N/A"
        click.echo(
            f"{r.target_id:<10} {r.n_actives:>6} {r.n_decoys:>6} {auroc_s:>7} {bedroc_s:>7} {ef1_s:>7} {ef5_s:>7} {r.status:<16}"
        )

    evaluated = [r for r in results if r.status in ("pass", "fail")]
    n_pass = sum(1 for r in evaluated if r.passed)
    click.echo(f"\n{n_pass}/{len(evaluated)} targets passed enrichment thresholds")

    excluded = [r for r in results if r.status == "fail"]
    if excluded:
        click.echo("\n--- Targets below enrichment threshold ---")
        for r in excluded:
            click.echo(f"  {r.target_id}: AUROC={r.auroc:.3f} BEDROC={r.bedroc_alpha20:.3f}")
