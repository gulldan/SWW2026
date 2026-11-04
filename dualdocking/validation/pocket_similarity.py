"""Pocket similarity from co-crystal PDB structures.

Computes sequence identity between ligand-contacting pocket residues for two
PDB structures. Pocket residues are defined as protein residues having at least
one heavy atom within a configurable radius (default 5.0 A) of heavy ligand
atoms.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from dualdocking.data.http_utils import get_with_retry
from dualdocking.scoring.binding_box import extract_ligand_info
from dualdocking.validation.nw_align import nw_align

_RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

_AA3_TO_1: dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    # Common modified/selenated residues frequently seen in crystal structures
    "MSE": "M",
    "SEC": "C",
}


@dataclass
class PocketSimilarityReport:
    """Pocket-sequence identity report for two PDB entries."""

    pdb_a: str
    pdb_b: str
    radius_angstrom: float = 5.0
    chain_a: str = ""
    chain_b: str = ""
    ligand_a: str = ""
    ligand_b: str = ""
    pocket_len_a: int = 0
    pocket_len_b: int = 0
    identity_pct: float = 0.0
    aligned_length: int = 0
    matches: int = 0
    gaps: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _PocketSequence:
    chain: str
    ligand_label: str
    sequence: str
    residue_count: int


def _safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _cache_key(pdb_a: str, pdb_b: str, radius_angstrom: float) -> str:
    return f"pocket_{_safe_id(pdb_a.upper())}_{_safe_id(pdb_b.upper())}_{radius_angstrom:.2f}.json"


def _fetch_pdb_if_missing(pdb_id: str, cache_dir: Path) -> Path:
    pdb_id = pdb_id.upper()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{pdb_id}.pdb"
    if path.exists():
        return path
    url = _RCSB_PDB_URL.format(pdb_id=pdb_id)
    logger.info("Downloading PDB {} -> {}", pdb_id, path)
    resp = get_with_retry(url, timeout=30)
    path.write_text(resp.text, encoding="utf-8")
    return path


def _parse_protein_residues(
    pdb_path: Path,
) -> dict[str, dict[tuple[int, str], tuple[str, list[np.ndarray[Any, np.dtype[np.float64]]]]]]:
    chains: dict[str, dict[tuple[int, str], tuple[str, list[np.ndarray[Any, np.dtype[np.float64]]]]]] = {}

    with pdb_path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            altloc = line[16:17].strip()
            if altloc not in {"", "A"}:
                continue

            resname = line[17:20].strip().upper()
            aa = _AA3_TO_1.get(resname)
            if aa is None:
                continue

            chain = line[21:22].strip() or "_"
            resseq_txt = line[22:26].strip()
            icode = line[26:27].strip()

            try:
                resseq = int(resseq_txt)
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            element = line[76:78].strip().upper()
            if element == "H":
                continue

            key = (resseq, icode)
            chain_res = chains.setdefault(chain, {})
            if key not in chain_res:
                chain_res[key] = (aa, [np.array([x, y, z], dtype=np.float64)])
            else:
                prev_aa, atoms = chain_res[key]
                atoms.append(np.array([x, y, z], dtype=np.float64))
                chain_res[key] = (prev_aa, atoms)

    return chains


def _residue_contact(
    residue_atoms: list[np.ndarray[Any, np.dtype[np.float64]]],
    ligand_coords: np.ndarray[Any, np.dtype[np.float64]],
    radius_sq: float,
) -> bool:
    coords = np.array(residue_atoms, dtype=np.float64)
    deltas = coords[:, None, :] - ligand_coords[None, :, :]
    dist_sq = np.sum(deltas * deltas, axis=2)
    return bool(np.any(dist_sq <= radius_sq))


def _build_pocket_sequence_from_pdb(
    pdb_path: Path,
    *,
    radius_angstrom: float,
) -> _PocketSequence:
    lig = extract_ligand_info(pdb_path)
    if lig is None:
        msg = f"No ligand detected in {pdb_path}"
        raise ValueError(msg)

    chains = _parse_protein_residues(pdb_path)
    if not chains:
        msg = f"No protein ATOM residues parsed from {pdb_path}"
        raise ValueError(msg)

    radius_sq = float(radius_angstrom) ** 2
    best_chain = ""
    best_contacts: list[tuple[int, str]] = []
    best_chain_len = 0

    for chain, residues in chains.items():
        ordered = sorted(residues, key=lambda k: (k[0], k[1] or " "))
        contacts: list[tuple[int, str]] = []
        for key in ordered:
            aa, atom_coords = residues[key]
            if not aa:
                continue
            if _residue_contact(atom_coords, lig.coords, radius_sq):
                contacts.append(key)

        if not contacts:
            continue

        if len(contacts) > len(best_contacts) or (len(contacts) == len(best_contacts) and len(ordered) > best_chain_len):
            best_chain = chain
            best_contacts = contacts
            best_chain_len = len(ordered)

    if not best_contacts or not best_chain:
        msg = f"No protein residues within {radius_angstrom:.1f} A of ligand in {pdb_path}"
        raise ValueError(msg)

    residues = chains[best_chain]
    pocket_seq = "".join(residues[key][0] for key in best_contacts)
    ligand_label = f"{lig.resname}:{lig.chain}{lig.resseq}"
    return _PocketSequence(
        chain=best_chain,
        ligand_label=ligand_label,
        sequence=pocket_seq,
        residue_count=len(best_contacts),
    )


def compute_pocket_similarity_from_pdb_paths(
    pdb_path_a: str | Path,
    pdb_path_b: str | Path,
    *,
    pdb_id_a: str = "",
    pdb_id_b: str = "",
    radius_angstrom: float = 5.0,
) -> PocketSimilarityReport:
    """Compute pocket identity from two local PDB paths."""
    pa = Path(pdb_path_a)
    pb = Path(pdb_path_b)
    pocket_a = _build_pocket_sequence_from_pdb(pa, radius_angstrom=radius_angstrom)
    pocket_b = _build_pocket_sequence_from_pdb(pb, radius_angstrom=radius_angstrom)

    aln = nw_align(pocket_a.sequence, pocket_b.sequence)
    return PocketSimilarityReport(
        pdb_a=(pdb_id_a or pa.stem).upper(),
        pdb_b=(pdb_id_b or pb.stem).upper(),
        radius_angstrom=radius_angstrom,
        chain_a=pocket_a.chain,
        chain_b=pocket_b.chain,
        ligand_a=pocket_a.ligand_label,
        ligand_b=pocket_b.ligand_label,
        pocket_len_a=pocket_a.residue_count,
        pocket_len_b=pocket_b.residue_count,
        identity_pct=aln.identity_pct,
        aligned_length=aln.aligned_length,
        matches=aln.matches,
        gaps=aln.gaps,
        notes=[],
    )


def compute_pocket_similarity(
    pdb_a: str,
    pdb_b: str,
    *,
    cache_dir: Path | None = None,
    radius_angstrom: float = 5.0,
) -> PocketSimilarityReport:
    """Compute pocket identity for a pair of PDB IDs with optional disk cache."""
    pdb_a = pdb_a.upper().strip()
    pdb_b = pdb_b.upper().strip()
    cache_file: Path | None = None
    pdb_cache_dir: Path | None = None

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / _cache_key(pdb_a, pdb_b, radius_angstrom)
        pdb_cache_dir = cache_dir / "pdb"
        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return PocketSimilarityReport(**data)
    else:
        pdb_cache_dir = Path.cwd() / ".pocket_pdb_cache"
        pdb_cache_dir.mkdir(parents=True, exist_ok=True)

    path_a = _fetch_pdb_if_missing(pdb_a, pdb_cache_dir)
    path_b = _fetch_pdb_if_missing(pdb_b, pdb_cache_dir)
    report = compute_pocket_similarity_from_pdb_paths(
        path_a,
        path_b,
        pdb_id_a=pdb_a,
        pdb_id_b=pdb_b,
        radius_angstrom=radius_angstrom,
    )

    if cache_file is not None:
        cache_file.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")

    return report
