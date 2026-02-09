"""Валидация, дедупликация молекул и базовая статистика.

Пайплайн: сырые SMILES - парсинг RDKit - каноникализация - дедупликация - статистика.
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
    """Сводка санитарной проверки SMILES."""

    total: int
    valid: int
    unique: int
    invalid_indices: list[int]

    @property
    def validity_rate(self) -> float:
        """Доля валидных молекул среди всех входных."""
        return self.valid / self.total if self.total > 0 else 0.0

    @property
    def uniqueness_rate(self) -> float:
        """Доля уникальных молекул среди валидных."""
        return self.unique / self.valid if self.valid > 0 else 0.0


def canonicalize(smi: str) -> str | None:
    """Вернуть канонический SMILES или ``None``, если невалидный."""
    if not smi or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    canon = Chem.MolToSmiles(mol, canonical=True)
    return canon or None


def mol_hash(smi: str) -> str:
    """Детерминированный хеш для дедупликации."""
    return hashlib.sha256(smi.encode()).hexdigest()


def sanitize_smiles(
    smiles: Sequence[str],
    *,
    max_heavy_atoms: int = 80,
    max_mw: float = 800.0,
) -> tuple[list[str], SanitizeStats]:
    """Валидация, фильтрация и дедупликация списка SMILES.

    Parameters
    ----------
    smiles:
        Сырые SMILES (по одному на молекулу).
    max_heavy_atoms:
        Отклонять молекулы с большим числом тяжёлых атомов.
    max_mw:
        Отклонять молекулы с молекулярной массой выше порога.

    Returns:
    -------
    clean:
        Канонические, уникальные, отфильтрованные SMILES.
    stats:
        Сводная статистика.
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
