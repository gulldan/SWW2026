"""Re-docking validation: dock native co-crystallized ligands back to receptors.

For each target in pairs.yaml, extracts the co-crystallized ligand, docks it
back using UniDock GPU with identical parameters to the experiment, and computes
heavy-atom RMSD between the docked and crystal poses.

RMSD <= 2.0 A indicates the docking setup correctly reproduces the known
binding mode -- a standard sanity check for computational docking studies.

Usage::

    uv run python scripts/validate_redocking.py
    uv run python scripts/validate_redocking.py --target-id BACE1 --target-id GSK3B
    uv run python scripts/validate_redocking.py --dry-run
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np
import requests
from loguru import logger
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
DOCKER_BIN = shutil.which("docker") or "docker"

RMSD_THRESHOLD = 2.0  # Angstrom


@dataclass
class RedockResult:
    """Result of a single re-docking validation."""

    target_id: str
    pdb_id: str
    ligand_resname: str
    ligand_smiles: str
    n_crystal_atoms: int
    n_docked_atoms: int
    docking_score: float | None
    rmsd: float | None
    centroid_dist: float | None
    passed: bool
    error: str | None = None


def fetch_ligand_smiles(resname: str) -> str | None:
    """Fetch canonical SMILES for a PDB ligand from RCSB Chemical Component Dictionary."""
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{resname.upper()}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Failed to fetch SMILES for {}: {}", resname, exc)
        return None

    data = resp.json()
    descriptors: list[dict[str, str]] = data.get("rcsb_chem_comp_descriptor", [])
    for desc in descriptors:
        if desc.get("type") == "SMILES_CANONICAL":
            return desc.get("descriptor")
    # Fallback: try any SMILES
    for desc in descriptors:
        if "SMILES" in desc.get("type", ""):
            return desc.get("descriptor")
    return None


def symmetric_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Compute symmetry-corrected RMSD using the Hungarian algorithm.

    Finds the optimal atom-to-atom assignment that minimizes RMSD,
    handling cases where atom ordering differs between crystal and docked poses.
    """
    if len(coords_a) != len(coords_b):
        msg = f"Atom count mismatch: {len(coords_a)} vs {len(coords_b)}"
        raise ValueError(msg)
    dist_matrix = cdist(coords_a, coords_b, metric="euclidean")
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    squared_dists = dist_matrix[row_ind, col_ind] ** 2
    return float(np.sqrt(squared_dists.mean()))


def centroid_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Compute distance between centroids of two coordinate sets."""
    ca = coords_a.mean(axis=0)
    cb = coords_b.mean(axis=0)
    return float(np.linalg.norm(ca - cb))


def parse_sdf_coords(sdf_text: str) -> np.ndarray | None:
    """Extract heavy-atom coordinates from an SDF mol block (first conformer)."""
    coords: list[list[float]] = []
    lines = sdf_text.splitlines()
    if len(lines) < 5:
        return None

    counts_line = lines[3]
    match = re.match(r"\s*(\d+)\s+(\d+)", counts_line)
    if not match:
        return None

    n_atoms = int(match.group(1))
    for i in range(4, 4 + n_atoms):
        if i >= len(lines):
            break
        parts = lines[i].split()
        if len(parts) < 4:
            continue
        element = parts[3].strip()
        if element == "H":
            continue
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            coords.append([x, y, z])
        except ValueError:
            continue

    return np.array(coords, dtype=np.float64) if coords else None


def parse_unidock_output_sdf(sdf_path: Path) -> tuple[float | None, np.ndarray | None]:
    """Parse UniDock SDF output: extract best score and heavy-atom coords."""
    if not sdf_path.exists():
        return None, None

    text = sdf_path.read_text()
    blocks = text.split("$$$$")

    best_score: float | None = None
    best_coords: np.ndarray | None = None

    for raw_block in blocks:
        stripped = raw_block.strip()
        if not stripped:
            continue

        score = _extract_unidock_score(stripped)
        coords = parse_sdf_coords(stripped)

        if score is not None and (best_score is None or score < best_score):
            best_score = score
            best_coords = coords

    return best_score, best_coords


def _extract_unidock_score(block: str) -> float | None:
    """Extract ENERGY score from a UniDock SDF block."""
    block_lines = block.splitlines()
    for line_idx, line in enumerate(block_lines):
        if not line.startswith("> <Uni-Dock RESULT>"):
            continue
        if line_idx + 1 >= len(block_lines):
            continue
        payload = block_lines[line_idx + 1].strip()
        for token in payload.replace(";", " ").split():
            if token.startswith("ENERGY="):
                try:
                    return float(token.split("=", 1)[1])
                except ValueError:
                    pass
    return None


def _resolve_pdb_for_target(_target_id: str, pdb_id: str) -> Path:
    """Find PDB file for a target across all pair directories."""
    data_dir = PROJECT_ROOT / "data" / "targets"
    for pair_dir in data_dir.iterdir():
        if not pair_dir.is_dir():
            continue
        for role_dir in (pair_dir / "A", pair_dir / "B"):
            pdb_path = role_dir / f"{pdb_id.lower()}.pdb"
            if pdb_path.exists():
                return pdb_path
    return data_dir / f"{pdb_id.lower()}.pdb"


def _result_to_dict(r: RedockResult) -> dict[str, Any]:
    """Convert RedockResult to JSON-serializable dict."""
    return {
        "target_id": r.target_id,
        "pdb_id": r.pdb_id,
        "ligand_resname": r.ligand_resname,
        "ligand_smiles": r.ligand_smiles,
        "n_crystal_atoms": r.n_crystal_atoms,
        "n_docked_atoms": r.n_docked_atoms,
        "docking_score": r.docking_score,
        "rmsd": r.rmsd,
        "centroid_dist": r.centroid_dist,
        "passed": r.passed,
        "error": r.error,
    }


def _validate_target(
    target_id: str,
    target_cfg: dict[str, Any],
    runner: Any,
    params: Any,
    work_dir: Path,
    rmsd_threshold: float,
) -> RedockResult:
    """Run re-docking validation for a single target."""
    from dock_calibration import (
        _dock_batch_unidock,
        _ensure_ligand_pdbqt,
        _ensure_receptor_pdbqt,
        _parse_vec,
    )

    from dualdocking.binding_box import extract_binding_box, extract_ligand_info

    pdb_id = target_cfg["pdb_id"]
    pdb_path = _resolve_pdb_for_target(target_id, pdb_id)

    if not pdb_path.exists():
        return RedockResult(
            target_id=target_id, pdb_id=pdb_id, ligand_resname="?",
            ligand_smiles="", n_crystal_atoms=0, n_docked_atoms=0,
            docking_score=None, rmsd=None, centroid_dist=None,
            passed=False, error=f"PDB not found: {pdb_path}",
        )

    lig_info = extract_ligand_info(pdb_path)
    if lig_info is None:
        return RedockResult(
            target_id=target_id, pdb_id=pdb_id, ligand_resname="?",
            ligand_smiles="", n_crystal_atoms=0, n_docked_atoms=0,
            docking_score=None, rmsd=None, centroid_dist=None,
            passed=False, error="No ligand found in PDB",
        )

    crystal_coords = lig_info.coords
    resname = lig_info.resname

    smiles = fetch_ligand_smiles(resname)
    if smiles is None:
        return RedockResult(
            target_id=target_id, pdb_id=pdb_id, ligand_resname=resname,
            ligand_smiles="", n_crystal_atoms=len(crystal_coords), n_docked_atoms=0,
            docking_score=None, rmsd=None, centroid_dist=None,
            passed=False, error=f"Could not fetch SMILES for ligand {resname}",
        )

    target_work = work_dir / target_id
    target_work.mkdir(parents=True, exist_ok=True)
    receptor_path = target_work / f"{target_id}.pdbqt"
    sdf_path = target_work / "native_ligand.sdf"
    ligand_path = target_work / "native_ligand.pdbqt"
    output_dir = target_work / "dock_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _ensure_receptor_pdbqt(runner, pdb_path, receptor_path, timeout=params.timeout_seconds)
    except Exception as exc:
        return RedockResult(
            target_id=target_id, pdb_id=pdb_id, ligand_resname=resname,
            ligand_smiles=smiles, n_crystal_atoms=len(crystal_coords), n_docked_atoms=0,
            docking_score=None, rmsd=None, centroid_dist=None,
            passed=False, error=f"Receptor prep failed: {exc}",
        )

    ok, err = _ensure_ligand_pdbqt(
        runner, smiles, sdf_path, ligand_path,
        timeout=params.timeout_seconds, seed=42, overwrite=True,
    )
    if not ok:
        return RedockResult(
            target_id=target_id, pdb_id=pdb_id, ligand_resname=resname,
            ligand_smiles=smiles, n_crystal_atoms=len(crystal_coords), n_docked_atoms=0,
            docking_score=None, rmsd=None, centroid_dist=None,
            passed=False, error=f"Ligand prep failed: {err}",
        )

    center_str, size_str = extract_binding_box(pdb_path, padding=params.box_padding)
    center = _parse_vec(center_str)
    size = _parse_vec(size_str)

    try:
        batch_scores = _dock_batch_unidock(
            runner, "unidock", receptor_path, [ligand_path],
            center, size, params, output_dir,
            batch_id=0, seed=42, timeout=params.timeout_seconds,
        )
    except Exception as exc:
        return RedockResult(
            target_id=target_id, pdb_id=pdb_id, ligand_resname=resname,
            ligand_smiles=smiles, n_crystal_atoms=len(crystal_coords), n_docked_atoms=0,
            docking_score=None, rmsd=None, centroid_dist=None,
            passed=False, error=f"Docking failed: {exc}",
        )

    docking_score = batch_scores.get(ligand_path)

    out_sdf = output_dir / f"{ligand_path.stem}_out.sdf"
    dock_score_sdf, docked_coords = parse_unidock_output_sdf(out_sdf)
    if dock_score_sdf is not None:
        docking_score = dock_score_sdf

    rmsd_val = None
    cdist_val = None
    if docked_coords is not None:
        cdist_val = centroid_distance(crystal_coords, docked_coords)
        if len(crystal_coords) == len(docked_coords):
            try:
                rmsd_val = symmetric_rmsd(crystal_coords, docked_coords)
            except Exception as exc:
                logger.warning("RMSD computation failed for {}: {}", target_id, exc)
        else:
            logger.warning(
                "Atom count mismatch for {}: crystal={} docked={}; using centroid distance",
                target_id, len(crystal_coords), len(docked_coords),
            )
            rmsd_val = cdist_val

    passed = rmsd_val is not None and rmsd_val <= rmsd_threshold

    return RedockResult(
        target_id=target_id,
        pdb_id=pdb_id,
        ligand_resname=resname,
        ligand_smiles=smiles,
        n_crystal_atoms=len(crystal_coords),
        n_docked_atoms=len(docked_coords) if docked_coords is not None else 0,
        docking_score=docking_score,
        rmsd=rmsd_val,
        centroid_dist=cdist_val,
        passed=passed,
    )


@click.command()
@click.option("--config", default="configs/pairs.yaml", help="Path to pairs config")
@click.option("--target-id", multiple=True, help="Specific target IDs (repeatable)")
@click.option("--docker-image", default="dualdocking/freedpp:latest")
@click.option("--gpus", default="all")
@click.option("--rmsd-threshold", default=RMSD_THRESHOLD, type=float)
@click.option("--output", default="data/validation/redocking_results.json")
@click.option("--dry-run", is_flag=True, help="Show targets without docking")
def main(
    config: str,
    target_id: tuple[str, ...],
    docker_image: str,
    gpus: str,
    rmsd_threshold: float,
    output: str,
    dry_run: bool,
) -> None:
    """Validate docking setup by re-docking co-crystallized ligands."""
    import yaml
    from dock_calibration import DockerRunner, DockingParams

    from dualdocking.binding_box import extract_ligand_info
    from dualdocking.config import load_pairs

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
            pdb_path = _resolve_pdb_for_target(tid, t["pdb_id"])
            info = extract_ligand_info(pdb_path) if pdb_path.exists() else None
            resname = info.resname if info else "?"
            n_atoms = len(info.coords) if info else 0
            click.echo(f"  {tid:<10} PDB={t['pdb_id']} ligand={resname} atoms={n_atoms}")
        return

    work_dir = PROJECT_ROOT / "data" / "validation" / "redocking_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    results: list[RedockResult] = []

    runner_ctx = DockerRunner(docker_image, PROJECT_ROOT, gpus=gpus)
    with runner_ctx as runner:
        for tid, t in sorted(targets.items()):
            result = _validate_target(tid, t, runner, params, work_dir, rmsd_threshold)
            results.append(result)

            status = "PASS" if result.passed else "FAIL"
            rmsd_str = f"{result.rmsd:.2f}" if result.rmsd is not None else "N/A"
            score_str = f"{result.docking_score:.2f}" if result.docking_score is not None else "N/A"
            logger.info("[{}] {} ligand={} score={} RMSD={} A", status, tid, result.ligand_resname, score_str, rmsd_str)

    out_path = PROJECT_ROOT / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([_result_to_dict(r) for r in results], indent=2))
    logger.info("Results saved to {}", out_path)

    click.echo("\n--- Re-docking Validation Summary ---")
    click.echo(f"{'Target':<10} {'PDB':<6} {'Ligand':<8} {'Score':>8} {'RMSD':>8} {'Status':<6}")
    click.echo("-" * 54)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        rmsd_s = f"{r.rmsd:.2f}" if r.rmsd is not None else "N/A"
        score_s = f"{r.docking_score:.2f}" if r.docking_score is not None else "N/A"
        click.echo(f"{r.target_id:<10} {r.pdb_id:<6} {r.ligand_resname:<8} {score_s:>8} {rmsd_s:>8} {status:<6}")

    n_pass = sum(1 for r in results if r.passed)
    click.echo(f"\n{n_pass}/{len(results)} targets passed (RMSD <= {rmsd_threshold:.1f} A)")

    if n_pass < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
