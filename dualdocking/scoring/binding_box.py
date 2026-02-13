"""Binding-box extraction from PDB files.

Determines docking box coordinates from a co-crystallized ligand (HETATM
records).  Falls back to the protein centre-of-mass with a cubic
30x30x30 Angstrom box when no ligand is found.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class LigandInfo:
    """Metadata and coordinates of a co-crystallized ligand.

    Attributes:
        resname: PDB residue name (e.g. ``"ATP"``).
        chain: Chain identifier.
        resseq: Residue sequence number (string, may contain insertion code).
        coords: Heavy-atom coordinates with shape ``(N, 3)``.
    """

    resname: str
    chain: str
    resseq: str
    coords: NDArray[np.float64]


# HETATM records that are NOT ligands
_EXCLUDED_RESNAMES: set[str] = {
    # Water
    "HOH",
    "WAT",
    "H2O",
    # Ions
    "NA",
    "CL",
    "MG",
    "CA",
    "ZN",
    "MN",
    "FE",
    "CU",
    "CO",
    "NI",
    "K",
    # Polyatomic ions
    "SO4",
    "PO4",
    "NO3",
    "CO3",
    # Buffers and cryoprotectants
    "GOL",
    "EDO",
    "PEG",
    "DMS",
    "ACT",
    "BME",
    "TRS",
    "MPD",
    "FMT",
    "IPA",
    "MES",
    "EPE",
    "CIT",
    "TAR",
    "SUC",
    "MLI",
}


def extract_ligand_coords(pdb_path: str | Path) -> NDArray[np.float64] | None:
    """Extract heavy-atom coordinates of the co-crystallized ligand from a PDB.

    Parses HETATM records, excluding water, ions, buffers and other
    non-crystallographic molecules listed in ``_EXCLUDED_RESNAMES``.
    When multiple ligand residues exist, the one with the most heavy
    atoms is selected.

    Args:
        pdb_path: Path to the PDB file.

    Returns:
        Coordinate array of shape ``(N, 3)`` for the chosen ligand, or
        ``None`` if no ligand residue is found.
    """
    residues: dict[tuple[str, str, str, str], list[list[float]]] = {}
    pdb_path = Path(pdb_path)

    with pdb_path.open() as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            if resname in _EXCLUDED_RESNAMES:
                continue
            # Heavy atoms only (exclude hydrogen)
            element = line[76:78].strip()
            if element == "H":
                continue
            chain = line[21:22].strip()
            resseq = line[22:26].strip()
            icode = line[26:27].strip()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                key = (resname, chain, resseq, icode)
                residues.setdefault(key, []).append([x, y, z])
            except (ValueError, IndexError):
                continue

    if not residues:
        return None

    # Choose the ligand with the most heavy atoms to avoid oversized boxes
    best_coords = max(residues.values(), key=len)
    return np.array(best_coords, dtype=np.float64)


def extract_ligand_info(pdb_path: str | Path) -> LigandInfo | None:
    """Extract the co-crystallized ligand with full residue metadata.

    Same selection logic as :func:`extract_ligand_coords` but returns a
    :class:`LigandInfo` dataclass with residue name, chain and sequence
    number needed for re-docking validation.

    Args:
        pdb_path: Path to the PDB file.

    Returns:
        :class:`LigandInfo` for the largest ligand, or ``None`` if no
        ligand residue is found.
    """
    residues: dict[tuple[str, str, str, str], list[list[float]]] = {}
    pdb_path = Path(pdb_path)

    with pdb_path.open() as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            if resname in _EXCLUDED_RESNAMES:
                continue
            element = line[76:78].strip()
            if element == "H":
                continue
            chain = line[21:22].strip()
            resseq = line[22:26].strip()
            icode = line[26:27].strip()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                key = (resname, chain, resseq, icode)
                residues.setdefault(key, []).append([x, y, z])
            except (ValueError, IndexError):
                continue

    if not residues:
        return None

    best_key = max(residues, key=lambda k: len(residues[k]))
    best_coords = residues[best_key]
    resname, chain, resseq, _icode = best_key
    return LigandInfo(
        resname=resname,
        chain=chain,
        resseq=resseq,
        coords=np.array(best_coords, dtype=np.float64),
    )


def compute_box(
    coords: NDArray[np.float64],
    padding: float = 10.0,
) -> tuple[str, str]:
    """Compute docking box centre and size from atom coordinates.

    Args:
        coords: Atom coordinates with shape ``(N, 3)``.
        padding: Padding in Angstroms added to each side of the
            bounding box.

    Returns:
        ``(center_str, size_str)`` -- comma-separated coordinate strings,
        e.g. ``("12.30,4.50,6.70", "25.00,20.00,22.00")``.
    """
    center = coords.mean(axis=0)
    extent = coords.max(axis=0) - coords.min(axis=0) + 2 * padding
    center_str = ",".join(f"{v:.2f}" for v in center)
    size_str = ",".join(f"{v:.2f}" for v in extent)
    return center_str, size_str


def extract_binding_box(
    pdb_path: str | Path,
    padding: float = 10.0,
) -> tuple[str, str]:
    """Extract a docking binding box from a PDB file.

    First attempts to locate a co-crystallized ligand via HETATM records.
    If none is found, falls back to the protein centre-of-mass with a
    fixed 30x30x30 Angstrom cubic box.

    Args:
        pdb_path: Path to the PDB file.
        padding: Padding in Angstroms for the ligand-based box (ignored
            during fallback).

    Returns:
        ``(center_str, size_str)`` comma-separated coordinate strings.

    Raises:
        ValueError: If no coordinates (neither HETATM nor ATOM) can be
            extracted from the file.
    """
    ligand_coords = extract_ligand_coords(pdb_path)
    if ligand_coords is not None and len(ligand_coords) > 0:
        return compute_box(ligand_coords, padding=padding)

    # Fallback: protein centre of mass + 30×30×30 cube
    protein_coords: list[list[float]] = []
    pdb_path = Path(pdb_path)
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                protein_coords.append([x, y, z])
            except (ValueError, IndexError):
                continue

    if not protein_coords:
        raise ValueError(f"Failed to extract coordinates from PDB: {pdb_path}")

    center = np.array(protein_coords, dtype=np.float64).mean(axis=0)
    center_str = ",".join(f"{v:.2f}" for v in center)
    size_str = "30.00,30.00,30.00"
    return center_str, size_str
