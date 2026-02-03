"""Diversity metrics for generated molecules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from dualdocking.chem_sanitize import canonicalize


@dataclass(frozen=True, slots=True)
class DiversityResult:
    """Container for diversity metrics."""

    uniqueness: float
    scaffold_diversity: float
    internal_diversity: float
    internal_diversity_stderr: float | None
    tanimoto_snn: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to a JSON-ready dict."""
        out = {
            "uniqueness": float(self.uniqueness),
            "scaffold_diversity": float(self.scaffold_diversity),
            "internal_diversity": float(self.internal_diversity),
            "tanimoto_snn": float(self.tanimoto_snn),
        }
        if self.internal_diversity_stderr is not None:
            out["internal_diversity_stderr"] = float(self.internal_diversity_stderr)
        return out


def _fingerprints(smiles: list[str]) -> list[DataStructs.cDataStructs.ExplicitBitVect]:
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)  # type: ignore[attr-defined]
        fps.append(fp)
    return fps


def uniqueness(smiles: list[str]) -> float:
    if not smiles:
        return 0.0
    canon = [canonicalize(s) for s in smiles]
    uniq = {s for s in canon if s}
    return float(len(uniq) / len(smiles))


def scaffold_diversity(smiles: list[str]) -> float:
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

    If n > 316, sample max_pairs pairs and return stderr.
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
