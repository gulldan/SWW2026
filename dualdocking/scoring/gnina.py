"""GNINA CNN-based molecular docking scorer (optional cross-validation).

Provides ``score_batch()`` for docking and scoring a list of SMILES against a
receptor using the GNINA Docker image.  GNINA uses convolutional neural
networks for pose scoring and affinity prediction on top of the smina/Vina
scoring function.

GNINA runs in full docking mode (not ``--score_only``), making it independent
of UniDock/Boltz-2 poses.  A binding box must be provided either via an
autobox ligand or explicit center/size coordinates.

This module is intended for post-hoc cross-validation of UniDock/Boltz-2
scores, not as an RL reward signal.
"""

from __future__ import annotations

import contextlib
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem

_DOCKER_IMAGE = "gnina/gnina:v1.1"
_SCORE_TIMEOUT = 300  # seconds per batch
_EXHAUSTIVENESS = 8  # GNINA default

_NAN = float("nan")


@dataclass(frozen=True, slots=True)
class GninaResult:
    """GNINA scoring result for a single molecule."""

    smiles: str
    cnn_score: float  # CNN pose score (higher = better pose)
    cnn_affinity: float  # CNN affinity prediction (more negative = better)
    vina_score: float  # Vina/smina score (kcal/mol, more negative = better)
    valid: bool


def _invalid_result(smi: str) -> GninaResult:
    return GninaResult(smiles=smi, cnn_score=_NAN, cnn_affinity=_NAN, vina_score=_NAN, valid=False)


def generate_conformer(smiles: str, seed: int = 42) -> Chem.Mol | None:
    """Generate 3D conformer from SMILES using RDKit ETKDGv3.

    Returns RDKit Mol with 3D coordinates, or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        status = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if status != 0:
            return None
    with contextlib.suppress(Exception):
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    return mol


def _mol_to_sdf(mol: Chem.Mol, path: Path) -> None:
    """Write RDKit Mol to SDF file."""
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()


def _parse_gnina_output(sdf_path: Path) -> list[tuple[float, float, float]]:
    """Parse GNINA output SDF, extract CNNscore, CNNaffinity, and minimizedAffinity.

    Returns list of (cnn_score, cnn_affinity, vina_score) tuples.
    """
    results: list[tuple[float, float, float]] = []
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for mol in supplier:
        if mol is None:
            results.append((_NAN, _NAN, _NAN))
            continue
        try:
            cnn_score = float(mol.GetProp("CNNscore"))
            cnn_affinity = float(mol.GetProp("CNNaffinity"))
        except KeyError:
            cnn_score = _NAN
            cnn_affinity = _NAN
        try:
            vina_score = float(mol.GetProp("minimizedAffinity"))
        except KeyError:
            vina_score = _NAN
        results.append((cnn_score, cnn_affinity, vina_score))
    return results


def score_batch(
    smiles_list: list[str],
    receptor_pdb: Path,
    *,
    autobox_ligand: Path | None = None,
    center: tuple[float, float, float] | None = None,
    box_size: tuple[float, float, float] | float = 25.0,
    exhaustiveness: int = _EXHAUSTIVENESS,
    docker_image: str = _DOCKER_IMAGE,
    timeout: int = _SCORE_TIMEOUT,
    seed: int = 42,
) -> list[GninaResult]:
    """Dock and score a batch of SMILES against a receptor using GNINA.

    Runs GNINA in full docking mode (not ``--score_only``), so it generates
    its own poses independently of any other docking engine.

    Args:
        smiles_list: List of SMILES to dock and score.
        receptor_pdb: Path to receptor PDB file.
        autobox_ligand: Optional ligand for autobox (determines binding site).
        center: Manual box center (x, y, z). Used if autobox_ligand is None.
        box_size: Box size in Angstroms. Either a single float for cubic box,
            or a (sx, sy, sz) tuple for rectangular.
        exhaustiveness: Search exhaustiveness (higher = slower but more thorough).
        docker_image: GNINA Docker image.
        timeout: Timeout in seconds.
        seed: Random seed for conformer generation and docking.

    Returns:
        List of GninaResult, one per input SMILES.
    """
    if autobox_ligand is None and center is None:
        logger.error("GNINA requires either autobox_ligand or center for binding box")
        return [_invalid_result(smi) for smi in smiles_list]

    with tempfile.TemporaryDirectory(prefix="gnina_") as tmpdir:
        tmp = Path(tmpdir)

        # Generate conformers and write combined SDF
        valid_indices: list[int] = []
        combined_sdf = tmp / "ligands.sdf"
        writer = Chem.SDWriter(str(combined_sdf))
        for i, smi in enumerate(smiles_list):
            mol = generate_conformer(smi, seed=seed + i)
            if mol is not None:
                mol.SetProp("_Name", f"mol_{i}")
                writer.write(mol)
                valid_indices.append(i)
        writer.close()

        if not valid_indices:
            return [_invalid_result(smi) for smi in smiles_list]

        # Build Docker command
        output_sdf = tmp / "output.sdf"
        cmd: list[str] = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{receptor_pdb.parent}:/receptor:ro",
            "-v",
            f"{tmp}:/work",
        ]

        gnina_args: list[str] = [
            docker_image,
            "gnina",
            "-r",
            f"/receptor/{receptor_pdb.name}",
            "-l",
            "/work/ligands.sdf",
            "-o",
            "/work/output.sdf",
            "--exhaustiveness",
            str(exhaustiveness),
            "--seed",
            str(seed),
        ]

        if autobox_ligand is not None:
            cmd.extend(["-v", f"{autobox_ligand.parent}:/autobox:ro"])
            gnina_args.extend(["--autobox_ligand", f"/autobox/{autobox_ligand.name}"])
        elif center is not None:
            if isinstance(box_size, (list, tuple)):
                sx, sy, sz = box_size
            else:
                sx = sy = sz = float(box_size)
            gnina_args.extend(
                [
                    "--center_x",
                    str(center[0]),
                    "--center_y",
                    str(center[1]),
                    "--center_z",
                    str(center[2]),
                    "--size_x",
                    str(sx),
                    "--size_y",
                    str(sy),
                    "--size_z",
                    str(sz),
                ]
            )

        cmd.extend(gnina_args)

        try:
            proc = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if proc.returncode != 0:
                logger.warning("GNINA failed (rc={}): {}", proc.returncode, proc.stderr[:500])
                return [_invalid_result(smi) for smi in smiles_list]
        except subprocess.TimeoutExpired:
            logger.warning("GNINA timed out after {}s", timeout)
            return [_invalid_result(smi) for smi in smiles_list]

        # Parse results
        scored = _parse_gnina_output(output_sdf) if output_sdf.exists() else []

        # Map back to original indices
        score_map = {idx: (cs, ca, vs) for idx, (cs, ca, vs) in zip(valid_indices, scored)}

        results: list[GninaResult] = []
        for i, smi in enumerate(smiles_list):
            if i in score_map:
                cs, ca, vs = score_map[i]
                results.append(
                    GninaResult(
                        smiles=smi,
                        cnn_score=cs,
                        cnn_affinity=ca,
                        vina_score=vs,
                        valid=not (np.isnan(cs) or np.isnan(ca)),
                    )
                )
            else:
                results.append(_invalid_result(smi))

    return results
