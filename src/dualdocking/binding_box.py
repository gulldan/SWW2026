"""Извлечение binding box из PDB файлов.

Используется для автоматического определения координат связывающего кармана
по со-кристаллизованному лиганду. Если лиганд не найден, используется
fallback на центр масс белка с кубическим боксом 30×30×30 Å.
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
    """Metadata and coordinates of a co-crystallized ligand."""

    resname: str
    chain: str
    resseq: str
    coords: NDArray  # (N, 3) heavy-atom coordinates

# HETATM-записи, которые НЕ являются лигандами
_EXCLUDED_RESNAMES: set[str] = {
    # Вода
    "HOH",
    "WAT",
    "H2O",
    # Ионы
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
    # Многоатомные ионы
    "SO4",
    "PO4",
    "NO3",
    "CO3",
    # Буферы и криопротекторы
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


def extract_ligand_coords(pdb_path: str | Path) -> NDArray | None:
    """Извлечь координаты тяжёлых атомов лиганда из PDB.

    Парсит HETATM записи, исключая воду, ионы, буферы и другие
    некристаллографические молекулы.

    Parameters
    ----------
    pdb_path : str | Path
        Путь к PDB файлу с со-кристаллизованным лигандом.

    Returns:
    -------
    NDArray | None
        Массив координат (N, 3) или None если лиганд не найден.
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
            # Только тяжёлые атомы (не водород)
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
    """Extract the co-crystallized ligand with residue metadata.

    Same logic as :func:`extract_ligand_coords` but returns
    :class:`LigandInfo` with residue name, chain and sequence number
    needed for re-docking validation.
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
    coords: NDArray,
    padding: float = 10.0,
) -> tuple[str, str]:
    """Вычислить центр и размер бокса для докинга.

    Parameters
    ----------
    coords : NDArray
        Координаты атомов (N, 3).
    padding : float
        Отступ от крайних атомов в каждом направлении (Å).

    Returns:
    -------
    tuple[str, str]
        (center_str, size_str) — координаты через запятую.
        Пример: ("12.3,4.5,6.7", "25.0,20.0,22.0")
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
    """High-level: извлечь binding box из PDB файла.

    Сначала пытается найти со-кристаллизованный лиганд (HETATM).
    Если лиганд не найден, использует центр масс белка (ATOM записи)
    с кубическим боксом 30×30×30 Å.

    Parameters
    ----------
    pdb_path : str | Path
        Путь к PDB файлу.
    padding : float
        Отступ для бокса лиганда (Å). Игнорируется при fallback.

    Returns:
    -------
    tuple[str, str]
        (center_str, size_str)
    """
    ligand_coords = extract_ligand_coords(pdb_path)
    if ligand_coords is not None and len(ligand_coords) > 0:
        return compute_box(ligand_coords, padding=padding)

    # Fallback: центр масс белка + куб 30×30×30
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
        raise ValueError(f"Не удалось извлечь координаты из PDB: {pdb_path}")

    center = np.array(protein_coords, dtype=np.float64).mean(axis=0)
    center_str = ",".join(f"{v:.2f}" for v in center)
    size_str = "30.00,30.00,30.00"
    return center_str, size_str
