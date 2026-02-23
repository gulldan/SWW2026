"""GNINA CNN-based molecular docking scorer (optional cross-validation).

Provides ``score_batch()`` for docking and scoring a list of SMILES against a
receptor using either a local GNINA binary or a Docker image.  GNINA uses
convolutional neural networks for pose scoring and affinity prediction on top
of the smina/Vina scoring function.

GNINA runs in full docking mode (not ``--score_only``), making it independent
of UniDock/Boltz-2 poses.  A binding box must be provided either via an
autobox ligand or explicit center/size coordinates.

This module is intended for post-hoc cross-validation of UniDock/Boltz-2
scores, not as an RL reward signal.

Set ``GNINA_BINARY`` to the path of a local gnina executable to bypass Docker.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem

_DOCKER_IMAGE = "gnina/gnina:v1.3.2"
_SCORE_TIMEOUT = 300  # seconds per batch
_EXHAUSTIVENESS = 8  # GNINA default
_SUB_BATCH_SIZE = 50  # molecules per GNINA invocation
_TIMEOUT_PER_MOL = 30  # seconds per molecule for timeout calculation

# Local binary path (overrides Docker when set)
_GNINA_BINARY: str | None = os.environ.get("GNINA_BINARY")

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


def _build_local_cmd(
    gnina_bin: str,
    receptor_pdb: Path,
    ligands_sdf: Path,
    output_sdf: Path,
    *,
    autobox_ligand: Path | None,
    center: tuple[float, float, float] | None,
    box_size: tuple[float, float, float] | float,
    exhaustiveness: int,
    seed: int,
) -> list[str]:
    """Build command for local gnina binary."""
    cmd = [
        gnina_bin,
        "-r",
        str(receptor_pdb),
        "-l",
        str(ligands_sdf),
        "-o",
        str(output_sdf),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        "1",
        "--seed",
        str(seed),
    ]
    if autobox_ligand is not None:
        cmd.extend(["--autobox_ligand", str(autobox_ligand)])
    elif center is not None:
        if isinstance(box_size, (list, tuple)):
            sx, sy, sz = box_size
        else:
            sx = sy = sz = float(box_size)
        cmd.extend(
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
    return cmd


def _build_docker_cmd(
    docker_image: str,
    receptor_pdb: Path,
    tmp: Path,
    output_sdf_name: str,
    *,
    autobox_ligand: Path | None,
    center: tuple[float, float, float] | None,
    box_size: tuple[float, float, float] | float,
    exhaustiveness: int,
    seed: int,
) -> list[str]:
    """Build command for Docker-based gnina execution."""
    cmd: list[str] = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{receptor_pdb.parent}:/receptor:ro",
        "-v",
        f"{tmp}:/work",
    ]
    gnina_args: list[str] = [
        docker_image,
        "-r",
        f"/receptor/{receptor_pdb.name}",
        "-l",
        "/work/ligands.sdf",
        "-o",
        f"/work/{output_sdf_name}",
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        "1",
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
    return cmd


def _run_gnina_on_sdf(  # noqa: PLR0913
    combined_sdf: Path,
    receptor_pdb: Path,
    output_sdf: Path,
    tmp: Path,
    *,
    autobox_ligand: Path | None,
    center: tuple[float, float, float] | None,
    box_size: tuple[float, float, float] | float,
    exhaustiveness: int,
    docker_image: str,
    timeout: int,
    seed: int,
) -> list[tuple[float, float, float]]:
    """Run a single GNINA invocation on a prepared SDF and return parsed scores."""
    use_local = _GNINA_BINARY is not None

    if use_local and _GNINA_BINARY is not None:
        cmd = _build_local_cmd(
            _GNINA_BINARY,
            receptor_pdb,
            combined_sdf,
            output_sdf,
            autobox_ligand=autobox_ligand,
            center=center,
            box_size=box_size,
            exhaustiveness=exhaustiveness,
            seed=seed,
        )
    else:
        cmd = _build_docker_cmd(
            docker_image,
            receptor_pdb,
            tmp,
            output_sdf.name,
            autobox_ligand=autobox_ligand,
            center=center,
            box_size=box_size,
            exhaustiveness=exhaustiveness,
            seed=seed,
        )

    mode_str = f"local ({_GNINA_BINARY})" if use_local else f"docker ({docker_image})"
    logger.debug("GNINA cmd ({}): {}", mode_str, " ".join(cmd))

    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            logger.warning("GNINA failed (rc={}, {}): {}", proc.returncode, mode_str, proc.stderr[:500])
            return []
    except subprocess.TimeoutExpired:
        logger.warning("GNINA timed out after {}s ({})", timeout, mode_str)
        return []

    return _parse_gnina_output(output_sdf) if output_sdf.exists() else []


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
    sub_batch_size: int = _SUB_BATCH_SIZE,
) -> list[GninaResult]:
    """Dock and score a batch of SMILES against a receptor using GNINA.

    Runs GNINA in full docking mode (not ``--score_only``), so it generates
    its own poses independently of any other docking engine.

    Large batches are automatically split into sub-batches of
    ``sub_batch_size`` molecules to avoid timeouts.  The timeout for each
    sub-batch is ``max(timeout, sub_batch_size * 30)`` seconds.

    Args:
        smiles_list: List of SMILES to dock and score.
        receptor_pdb: Path to receptor PDB file.
        autobox_ligand: Optional ligand for autobox (determines binding site).
        center: Manual box center (x, y, z). Used if autobox_ligand is None.
        box_size: Box size in Angstroms. Either a single float for cubic box,
            or a (sx, sy, sz) tuple for rectangular.
        exhaustiveness: Search exhaustiveness (higher = slower but more thorough).
        docker_image: GNINA Docker image.
        timeout: Minimum timeout in seconds (actual = max of this and per-mol estimate).
        seed: Random seed for conformer generation and docking.
        sub_batch_size: Max molecules per GNINA invocation.

    Returns:
        List of GninaResult, one per input SMILES.
    """
    if autobox_ligand is None and center is None:
        logger.error("GNINA requires either autobox_ligand or center for binding box")
        return [_invalid_result(smi) for smi in smiles_list]

    n_total = len(smiles_list)

    # Pre-generate all conformers
    conformers: list[Chem.Mol | None] = []
    for i, smi in enumerate(smiles_list):
        conformers.append(generate_conformer(smi, seed=seed + i))

    n_conf = sum(1 for m in conformers if m is not None)
    if n_conf == 0:
        logger.warning("No valid conformers generated for {} molecules", n_total)
        return [_invalid_result(smi) for smi in smiles_list]

    logger.info("Generated conformers for {}/{} molecules", n_conf, n_total)

    # Build index of (global_idx, mol) for molecules with valid conformers
    work: list[tuple[int, Chem.Mol]] = [(i, mol) for i, mol in enumerate(conformers) if mol is not None]

    # Split into sub-batches
    chunks: list[list[tuple[int, Chem.Mol]]] = [work[i : i + sub_batch_size] for i in range(0, len(work), sub_batch_size)]
    n_chunks = len(chunks)
    logger.info("Split {} molecules into {} sub-batches of <={}", n_conf, n_chunks, sub_batch_size)

    # Run each sub-batch
    score_map: dict[int, tuple[float, float, float]] = {}

    for chunk_idx, chunk in enumerate(chunks):
        chunk_size = len(chunk)
        chunk_timeout = max(timeout, chunk_size * _TIMEOUT_PER_MOL)

        with tempfile.TemporaryDirectory(prefix="gnina_") as tmpdir:
            tmp = Path(tmpdir)
            combined_sdf = tmp / "ligands.sdf"
            output_sdf = tmp / "output.sdf"

            writer = Chem.SDWriter(str(combined_sdf))
            chunk_indices: list[int] = []
            for global_idx, mol in chunk:
                mol.SetProp("_Name", f"mol_{global_idx}")
                writer.write(mol)
                chunk_indices.append(global_idx)
            writer.close()

            logger.info(
                "Sub-batch {}/{}: {} mols, timeout={}s",
                chunk_idx + 1,
                n_chunks,
                chunk_size,
                chunk_timeout,
            )

            scored = _run_gnina_on_sdf(
                combined_sdf,
                receptor_pdb,
                output_sdf,
                tmp,
                autobox_ligand=autobox_ligand,
                center=center,
                box_size=box_size,
                exhaustiveness=exhaustiveness,
                docker_image=docker_image,
                timeout=chunk_timeout,
                seed=seed,
            )

            score_map.update(dict(zip(chunk_indices, scored)))

            n_ok = sum(1 for s in scored if not (np.isnan(s[0]) or np.isnan(s[1])))
            logger.info(
                "Sub-batch {}/{}: {}/{} valid scores",
                chunk_idx + 1,
                n_chunks,
                n_ok,
                chunk_size,
            )

    # Assemble final results
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
