"""Diversity metrics for generated molecules.

Computes uniqueness, scaffold diversity, internal Tanimoto diversity,
and shared nearest-neighbor (SNN) similarity for sets of SMILES.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from dualdocking.data.chem_sanitize import canonicalize


@dataclass(frozen=True, slots=True)
class DiversityResult:
    """Container for diversity metrics.

    Attributes:
        uniqueness: Fraction of unique canonical SMILES among all inputs.
        scaffold_diversity: Ratio of unique Murcko scaffolds to unique molecules.
        internal_diversity: Mean pairwise Tanimoto distance (1 - similarity).
        internal_diversity_stderr: Standard error of internal diversity (sampled),
            or ``None`` if exact computation was used.
        tanimoto_snn: Mean k-nearest-neighbor Tanimoto similarity.
    """

    uniqueness: float
    scaffold_diversity: float
    internal_diversity: float
    internal_diversity_stderr: float | None
    tanimoto_snn: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to a JSON-ready dict.

        Returns:
            Dict with float values for all metrics.
        """
        out = {
            "uniqueness": float(self.uniqueness),
            "scaffold_diversity": float(self.scaffold_diversity),
            "internal_diversity": float(self.internal_diversity),
            "tanimoto_snn": float(self.tanimoto_snn),
        }
        if self.internal_diversity_stderr is not None:
            out["internal_diversity_stderr"] = float(self.internal_diversity_stderr)
        return out


_MORGAN_GENERATOR: Any = None


def _get_morgan_generator() -> Any:
    """Lazily initialize and return a Morgan fingerprint generator.

    Returns:
        An ``rdFingerprintGenerator`` Morgan generator, or ``None`` if the
        new-style API is unavailable.
    """
    global _MORGAN_GENERATOR
    if _MORGAN_GENERATOR is not None:
        return _MORGAN_GENERATOR
    try:
        from rdkit.Chem import rdFingerprintGenerator

        _MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    except Exception:
        _MORGAN_GENERATOR = None
    return _MORGAN_GENERATOR


def _fingerprints(smiles: list[str]) -> list[DataStructs.cDataStructs.ExplicitBitVect]:
    """Compute Morgan fingerprints (radius=2, 2048 bits) for valid SMILES.

    Args:
        smiles: Canonical SMILES strings.

    Returns:
        List of fingerprint bit vectors (invalid SMILES are silently skipped).
    """
    fps: list[DataStructs.cDataStructs.ExplicitBitVect] = []
    generator = _get_morgan_generator()
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        if generator is not None:
            fp = generator.GetFingerprint(mol)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)  # type: ignore[attr-defined]
        fps.append(fp)
    return fps


def uniqueness(smiles: list[str]) -> float:
    """Fraction of unique canonical SMILES among all inputs.

    Args:
        smiles: Raw SMILES strings.

    Returns:
        Ratio of unique valid SMILES to total input count (0.0 if empty).
    """
    if not smiles:
        return 0.0
    canon = [canonicalize(s) for s in smiles]
    uniq = {s for s in canon if s}
    return float(len(uniq) / len(smiles))


def scaffold_diversity(smiles: list[str]) -> float:
    """Ratio of unique Murcko scaffolds to unique molecules.

    Args:
        smiles: Raw SMILES strings.

    Returns:
        Scaffold diversity in [0, 1]. 0.0 if no valid molecules.
    """
    canon = [canonicalize(s) for s in smiles]
    canon = [s for s in canon if s]
    if not canon:
        return 0.0
    scaffolds: set[str] = set()
    for smi in canon:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            continue
        scaffolds.add(Chem.MolToSmiles(scaffold, canonical=True))
    return float(len(scaffolds) / len(set(canon))) if canon else 0.0


def internal_diversity_tanimoto(
    smiles: list[str],
    *,
    max_pairs: int = 10000,
    rng_seed: int = 42,
) -> tuple[float, float | None]:
    """Mean pairwise Tanimoto distance (1 - similarity).

    If n > 316, sample *max_pairs* pairs and return standard error.

    Args:
        smiles: Canonical SMILES strings.
        max_pairs: Maximum number of random pairs to sample when n > 316.
        rng_seed: Random seed for pair sampling.

    Returns:
        Tuple of ``(mean_diversity, stderr)``. ``stderr`` is ``None`` when
        exact (exhaustive) computation is used.
    """
    fps = _fingerprints(smiles)
    n = len(fps)
    if n < 2:
        return 0.0, None

    rng = np.random.default_rng(rng_seed)

    if n <= 316:
        sims: list[float] = []
        for i in range(n):
            sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :]))
        sims_arr = np.asarray(sims, dtype=float)
        diversity = float(1.0 - np.mean(sims_arr))
        return diversity, None

    # Sample pairs
    pairs = rng.integers(0, n, size=(max_pairs, 2))
    sims = []
    for i, j in pairs:
        if i == j:
            continue
        sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    sims_arr = np.asarray(sims, dtype=float)
    diversity_vals = 1.0 - sims_arr
    mean_div = float(np.mean(diversity_vals))
    stderr = float(np.std(diversity_vals, ddof=1) / np.sqrt(len(diversity_vals))) if len(diversity_vals) > 1 else 0.0
    return mean_div, stderr


def tanimoto_snn(smiles: list[str], k: int = 10) -> float:
    """Mean k-nearest-neighbor Tanimoto similarity.

    Args:
        smiles: Canonical SMILES strings.
        k: Number of nearest neighbors to average over.

    Returns:
        Mean SNN similarity across all molecules. 0.0 if fewer than 2
        valid molecules.
    """
    fps = _fingerprints(smiles)
    n = len(fps)
    if n == 0:
        return 0.0
    k = min(k, n - 1) if n > 1 else 1
    if k <= 0:
        return 0.0

    snn_vals: list[float] = []
    for _i, fp in enumerate(fps):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
        sims_sorted = sorted(sims, reverse=True)
        # skip self-similarity at index 0
        neighbors = sims_sorted[1 : k + 1] if len(sims_sorted) > 1 else []
        if neighbors:
            snn_vals.append(float(np.mean(neighbors)))
    return float(np.mean(snn_vals)) if snn_vals else 0.0


def compute_all_diversity_metrics(smiles: list[str]) -> DiversityResult:
    """Compute all diversity metrics for a set of SMILES.

    Args:
        smiles: Raw SMILES strings.

    Returns:
        A ``DiversityResult`` with uniqueness, scaffold diversity, internal
        diversity, and SNN similarity.
    """
    uniq = uniqueness(smiles)
    scaffold = scaffold_diversity(smiles)
    internal, stderr = internal_diversity_tanimoto(smiles, max_pairs=10000, rng_seed=42)
    snn = tanimoto_snn(smiles, k=10)
    return DiversityResult(
        uniqueness=uniq,
        scaffold_diversity=scaffold,
        internal_diversity=internal,
        internal_diversity_stderr=stderr,
        tanimoto_snn=snn,
    )
