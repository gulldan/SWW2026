"""SMILES validation, deduplication, and basic statistics.

Pipeline: raw SMILES -> RDKit parse -> canonicalization -> deduplication -> stats.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger
from rdkit import Chem
from rdkit.Chem import Descriptors

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class SanitizeStats:
    """Summary statistics from SMILES sanitization.

    Attributes:
        total: Total number of input SMILES.
        valid: Number of SMILES that passed RDKit parsing.
        unique: Number of unique canonical SMILES after deduplication.
        invalid_indices: Zero-based indices of inputs that failed validation.
    """

    total: int
    valid: int
    unique: int
    invalid_indices: list[int]

    @property
    def validity_rate(self) -> float:
        """Fraction of valid molecules among all inputs."""
        return self.valid / self.total if self.total > 0 else 0.0

    @property
    def uniqueness_rate(self) -> float:
        """Fraction of unique molecules among valid ones."""
        return self.unique / self.valid if self.valid > 0 else 0.0


def canonicalize(smi: str) -> str | None:
    """Return canonical SMILES, or ``None`` if the input is invalid.

    Args:
        smi: Raw SMILES string.

    Returns:
        Canonical SMILES string, or ``None`` for unparsable input.
    """
    if not smi or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    canon = Chem.MolToSmiles(mol, canonical=True)
    return canon or None


def mol_hash(smi: str) -> str:
    """Deterministic SHA-256 hash of a SMILES string for deduplication.

    Args:
        smi: Canonical SMILES string.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(smi.encode()).hexdigest()


def sanitize_smiles(
    smiles: Sequence[str],
    *,
    max_heavy_atoms: int = 80,
    max_mw: float = 800.0,
) -> tuple[list[str], SanitizeStats]:
    """Validate, filter, and deduplicate a list of SMILES.

    Args:
        smiles: Raw SMILES strings (one per molecule).
        max_heavy_atoms: Reject molecules exceeding this heavy-atom count.
        max_mw: Reject molecules with molecular weight above this threshold.

    Returns:
        Tuple of ``(clean_smiles, stats)`` where *clean_smiles* are canonical,
        unique, and filtered SMILES, and *stats* summarizes the sanitization.
    """
    seen: set[str] = set()
    clean: list[str] = []
    invalid_indices: list[int] = []

    for idx, smi in enumerate(smiles):
        canon = canonicalize(smi.strip())
        if canon is None:
            invalid_indices.append(idx)
            continue

        mol = Chem.MolFromSmiles(canon)
        if mol is None:
            invalid_indices.append(idx)
            continue

        if mol.GetNumHeavyAtoms() > max_heavy_atoms:
            logger.debug("Пропуск idx={}: {} тяжёлых атомов", idx, mol.GetNumHeavyAtoms())
            invalid_indices.append(idx)
            continue

        mw = Descriptors.MolWt(mol)  # type: ignore[attr-defined]
        if mw > max_mw:
            logger.debug("Пропуск idx={}: MW={:.1f}", idx, mw)
            invalid_indices.append(idx)
            continue

        h = mol_hash(canon)
        if h in seen:
            continue
        seen.add(h)
        clean.append(canon)

    stats = SanitizeStats(
        total=len(smiles),
        valid=len(smiles) - len(invalid_indices),
        unique=len(clean),
        invalid_indices=invalid_indices,
    )
    logger.info(
        "Санитизация: {}/{} валидных, {}/{} уникальных",
        stats.valid,
        stats.total,
        stats.unique,
        stats.valid,
    )
    return clean, stats
