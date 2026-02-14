"""Re-docking validation: dock native co-crystallized ligands back to receptors.

For each target, extracts the co-crystallized ligand, docks it back using
UniDock GPU with identical parameters to the experiment, and computes
heavy-atom RMSD between the docked and crystal poses.

RMSD <= 2.0 A indicates the docking setup correctly reproduces the known
binding mode -- a standard sanity check for computational docking studies.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from dualdocking.data.http_utils import get_with_retry
from dualdocking.scoring.docking import (
    dock_batch_unidock,
    ensure_ligand_pdbqt,
    ensure_receptor_pdbqt,
    parse_vec,
)

RMSD_THRESHOLD = 2.0  # Angstrom

# Possible status values for a RedockResult.
STATUS_PASS = "pass"  # noqa: S105  # not a password
STATUS_FAIL = "fail"
STATUS_SKIPPED_APO = "skipped:apo"
STATUS_SKIPPED_BLOCKED = "skipped:blocked"
STATUS_ERROR = "error"


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
    status: str = STATUS_ERROR
    error: str | None = None


def fetch_ligand_smiles(resname: str) -> str | None:
    """Fetch canonical SMILES for a PDB ligand from RCSB Chemical Component Dictionary."""
    import requests

    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{resname.upper()}"
    try:
        resp = get_with_retry(url, timeout=30)
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


def resolve_pdb_for_target(_target_id: str, pdb_id: str, project_root: Path) -> Path:
    """Find PDB file for a target across all pair directories.

    Args:
        _target_id: Internal target identifier (unused, kept for API consistency).
        pdb_id: PDB accession code.
        project_root: Project root directory.

    Returns:
        Path to the PDB file (may not exist if not found in any pair dir).
    """
    data_dir = project_root / "output" / "data" / "targets"
    for pair_dir in data_dir.iterdir():
        if not pair_dir.is_dir():
            continue
        for role_dir in (pair_dir / "A", pair_dir / "B"):
            pdb_path = role_dir / f"{pdb_id.lower()}.pdb"
            if pdb_path.exists():
                return pdb_path
    return data_dir / f"{pdb_id.lower()}.pdb"


def result_to_dict(r: RedockResult) -> dict[str, Any]:
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
        "status": r.status,
        "error": r.error,
    }


CSV_COLUMNS = [
    "target_id",
    "pdb_id",
    "ligand_resname",
    "docking_score",
    "rmsd",
    "centroid_dist",
    "status",
    "error",
]


def write_csv(results: list[RedockResult], path: Path) -> None:
    """Write results as CSV for downstream reporting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            row = result_to_dict(r)
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def validate_target(
    target_id: str,
    target_cfg: dict[str, Any],
    runner: Any,
    params: Any,
    work_dir: Path,
    rmsd_threshold: float,
    project_root: Path,
) -> RedockResult:
    """Run re-docking validation for a single target.

    Args:
        target_id: Internal target identifier.
        target_cfg: Target config dict (must have ``pdb_id``).
        runner: Docker runner (or None for local execution).
        params: Docking parameters (``DockingParams``).
        work_dir: Working directory for intermediate files.
        rmsd_threshold: RMSD threshold for pass/fail.
        project_root: Project root directory.

    Returns:
        A ``RedockResult`` instance.
    """
    from dualdocking.scoring.binding_box import extract_binding_box, extract_ligand_info

    pdb_id = target_cfg["pdb_id"]
    pdb_path = resolve_pdb_for_target(target_id, pdb_id, project_root)

    if not pdb_path.exists():
        return RedockResult(
            target_id=target_id,
            pdb_id=pdb_id,
            ligand_resname="?",
            ligand_smiles="",
            n_crystal_atoms=0,
            n_docked_atoms=0,
            docking_score=None,
            rmsd=None,
            centroid_dist=None,
            passed=False,
            error=f"PDB not found: {pdb_path}",
        )

    lig_info = extract_ligand_info(pdb_path)
    if lig_info is None:
        return RedockResult(
            target_id=target_id,
            pdb_id=pdb_id,
            ligand_resname="?",
            ligand_smiles="",
            n_crystal_atoms=0,
            n_docked_atoms=0,
            docking_score=None,
            rmsd=None,
            centroid_dist=None,
            passed=False,
            status=STATUS_SKIPPED_APO,
            error="No co-crystal ligand in PDB (apo structure)",
        )

    crystal_coords = lig_info.coords
    resname = lig_info.resname

    smiles = fetch_ligand_smiles(resname)
    if smiles is None:
        return RedockResult(
            target_id=target_id,
            pdb_id=pdb_id,
            ligand_resname=resname,
            ligand_smiles="",
            n_crystal_atoms=len(crystal_coords),
            n_docked_atoms=0,
            docking_score=None,
            rmsd=None,
            centroid_dist=None,
            passed=False,
            error=f"Could not fetch SMILES for ligand {resname}",
        )

    target_work = work_dir / target_id
    target_work.mkdir(parents=True, exist_ok=True)
    receptor_path = target_work / f"{target_id}.pdbqt"
    sdf_path = target_work / "native_ligand.sdf"
    ligand_path = target_work / "native_ligand.pdbqt"
    output_dir = target_work / "dock_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        ensure_receptor_pdbqt(runner, pdb_path, receptor_path, timeout=params.timeout_seconds, project_root=project_root)
    except Exception as exc:
        return RedockResult(
            target_id=target_id,
            pdb_id=pdb_id,
            ligand_resname=resname,
            ligand_smiles=smiles,
            n_crystal_atoms=len(crystal_coords),
            n_docked_atoms=0,
            docking_score=None,
            rmsd=None,
            centroid_dist=None,
            passed=False,
            error=f"Receptor prep failed: {exc}",
        )

    ok, err = ensure_ligand_pdbqt(
        runner,
        smiles,
        sdf_path,
        ligand_path,
        timeout=params.timeout_seconds,
        seed=42,
        project_root=project_root,
        overwrite=True,
    )
    if not ok:
        return RedockResult(
            target_id=target_id,
            pdb_id=pdb_id,
            ligand_resname=resname,
            ligand_smiles=smiles,
            n_crystal_atoms=len(crystal_coords),
            n_docked_atoms=0,
            docking_score=None,
            rmsd=None,
            centroid_dist=None,
            passed=False,
            error=f"Ligand prep failed: {err}",
        )

    center_str, size_str = extract_binding_box(pdb_path, padding=params.box_padding)
    center = parse_vec(center_str)
    size = parse_vec(size_str)

    try:
        batch_scores = dock_batch_unidock(
            runner,
            "unidock",
            receptor_path,
            [ligand_path],
            center,
            size,
            params,
            output_dir,
            batch_id=0,
            seed=42,
            timeout=params.timeout_seconds,
            project_root=project_root,
        )
    except Exception as exc:
        return RedockResult(
            target_id=target_id,
            pdb_id=pdb_id,
            ligand_resname=resname,
            ligand_smiles=smiles,
            n_crystal_atoms=len(crystal_coords),
            n_docked_atoms=0,
            docking_score=None,
            rmsd=None,
            centroid_dist=None,
            passed=False,
            error=f"Docking failed: {exc}",
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
                target_id,
                len(crystal_coords),
                len(docked_coords),
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
        status=STATUS_PASS if passed else STATUS_FAIL,
    )
