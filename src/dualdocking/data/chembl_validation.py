"""ChEMBL-based validation for dual-target molecular generation.

Fetches known actives from ChEMBL REST API and computes validation metrics:
- Tanimoto nearest-neighbor similarity
- Enrichment factor at K
- Dual-active overlap analysis
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import requests
from loguru import logger
from rdkit import DataStructs
from rdkit.Chem import AllChem, MolFromSmiles

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Target ID → ChEMBL target ID mapping
TARGET_CHEMBL_MAP: dict[str, str] = {
    "BACE1": "CHEMBL4822",
    "GSK3B": "CHEMBL262",
    "AChE": "CHEMBL220",
    "BChE": "CHEMBL1914",
    "CDK2": "CHEMBL301",
    "ERK2": "CHEMBL4040",
    "EGFR": "CHEMBL203",
    "HER2": "CHEMBL1824",
    "CDK5": "CHEMBL4036",
    "MAOB": "CHEMBL2039",
    "COX1": "CHEMBL221",
    "COX2": "CHEMBL230",
}

_CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
_PAGE_LIMIT = 1000
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF = 5


def fetch_actives(
    target_id: str,
    *,
    min_pchembl: float = 5.0,
    cache_dir: Path | None = None,
) -> list[str]:
    """Fetch active compound SMILES from ChEMBL for a given target.

    Parameters
    ----------
    target_id:
        Internal target ID (e.g. "BACE1").
    min_pchembl:
        Minimum pChEMBL value (5.0 = IC50 ≤ 10 μM).
    cache_dir:
        If set, cache results as ``{cache_dir}/{target_id}_actives.json``.

    Returns:
    -------
    list[str]
        Canonical SMILES of active compounds (deduplicated).
    """
    chembl_id = TARGET_CHEMBL_MAP.get(target_id)
    if chembl_id is None:
        raise ValueError(f"Unknown target_id '{target_id}'; not in TARGET_CHEMBL_MAP")

    if cache_dir is not None:
        cache_path = cache_dir / f"{target_id}_actives.json"
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
            logger.info("Loaded {} actives for {} from cache", len(data), target_id)
            return data

    smiles_set: set[str] = set()
    offset = 0

    while True:
        url = (
            f"{_CHEMBL_API}/activity.json"
            f"?target_chembl_id={chembl_id}"
            f"&pchembl_value__gte={min_pchembl}"
            f"&limit={_PAGE_LIMIT}&offset={offset}"
        )

        resp = _get_with_retry(url)
        payload = resp.json()
        activities = payload.get("activities", [])

        for act in activities:
            smi = act.get("canonical_smiles")
            if smi:
                smiles_set.add(smi)

        meta = payload.get("page_meta", {})
        if meta.get("next") is None or not activities:
            break
        offset += _PAGE_LIMIT

    result = sorted(smiles_set)
    logger.info("Fetched {} unique actives for {} (ChEMBL {})", len(result), target_id, chembl_id)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{target_id}_actives.json"
        cache_path.write_text(json.dumps(result))

    return result


def find_dual_actives(actives_a: list[str], actives_b: list[str]) -> list[str]:
    """Find compounds active on both targets (canonical SMILES intersection)."""
    return sorted(set(actives_a) & set(actives_b))


def _smiles_to_fp(smi: str) -> DataStructs.ExplicitBitVect | None:
    """Convert SMILES to Morgan fingerprint (radius=2, 2048 bits)."""
    mol = MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def compute_nearest_tanimoto(
    query_smiles: list[str],
    reference_smiles: list[str],
) -> NDArray:
    """Compute nearest-neighbor Tanimoto similarity for each query molecule.

    Parameters
    ----------
    query_smiles:
        Generated molecules to evaluate.
    reference_smiles:
        Known active molecules (from ChEMBL).

    Returns:
    -------
    NDArray
        Array of shape (len(query_smiles),) with max Tanimoto similarity
        to any reference molecule. NaN for invalid SMILES.
    """
    ref_fps = [_smiles_to_fp(smi) for smi in reference_smiles]
    ref_fps = [fp for fp in ref_fps if fp is not None]

    if not ref_fps:
        return np.full(len(query_smiles), np.nan)

    result = np.full(len(query_smiles), np.nan)
    for i, smi in enumerate(query_smiles):
        fp = _smiles_to_fp(smi)
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        result[i] = max(sims)

    return result


def enrichment_factor(
    similarities: NDArray,
    k: int,
    threshold: float = 0.4,
) -> float:
    """Compute enrichment factor at top-K.

    EF@K = (hits_in_top_K / K) / (total_hits / N)

    A hit is defined as a molecule with nearest-neighbor Tanimoto ≥ threshold.

    Parameters
    ----------
    similarities:
        Nearest-neighbor Tanimoto similarities (from compute_nearest_tanimoto).
    k:
        Number of top molecules to consider.
    threshold:
        Tanimoto threshold for a "hit".

    Returns:
    -------
    float
        Enrichment factor. Returns 0.0 if no hits exist in the full set.
    """
    valid = similarities[~np.isnan(similarities)]
    if len(valid) == 0 or k <= 0:
        return 0.0

    total_hits = np.sum(valid >= threshold)
    if total_hits == 0:
        return 0.0

    n = len(valid)
    # Sort descending by similarity, take top-K
    top_k = np.sort(valid)[::-1][:k]
    hits_in_top_k = np.sum(top_k >= threshold)

    baseline_rate = total_hits / n
    top_k_rate = hits_in_top_k / min(k, n)

    return float(top_k_rate / baseline_rate)


def _get_with_retry(url: str) -> requests.Response:
    """GET with retry and backoff."""
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp
        except (requests.RequestException, requests.HTTPError) as exc:
            if attempt == _RETRY_ATTEMPTS:
                raise
            logger.warning("ChEMBL API attempt {}/{} failed: {}", attempt, _RETRY_ATTEMPTS, exc)
            time.sleep(_RETRY_BACKOFF * attempt)
    raise RuntimeError("Unreachable")  # pragma: no cover
