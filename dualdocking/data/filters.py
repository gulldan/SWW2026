"""Drug-likeness filters: Lipinski, QED, PAINS, SA score.

Post-scoring analytical filters that do NOT affect RL reward.
Applied after scoring to flag molecules that pass standard drug-likeness criteria.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, FilterCatalog
from rdkit.Contrib.SA_Score import sascorer

# Module-level PAINS catalog (singleton)
_pains_params = FilterCatalog.FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS_CATALOG = FilterCatalog.FilterCatalog(_pains_params)


@dataclass(frozen=True, slots=True)
class FilterResult:
    """Result of applying all drug-likeness filters to a single molecule."""

    smiles: str
    passes_lipinski: bool
    passes_qed: bool
    passes_pains: bool
    passes_sa: bool
    passes_veber: bool
    passes_all: bool
    mw: float
    logp: float
    hbd: int
    hba: int
    qed_score: float
    sa_score: float
    pains_matches: int
    rotatable_bonds: float
    tpsa: float


@dataclass
class FilterStats:
    """Aggregate statistics from filtering a batch of molecules."""

    n_total: int
    n_valid: int
    n_pass_lipinski: int
    n_pass_qed: int
    n_pass_pains: int
    n_pass_sa: int
    n_pass_veber: int
    n_pass_all: int

    def to_dict(self) -> dict[str, Any]:
        """Return filter stats as a plain dictionary."""
        return {
            "n_total": self.n_total,
            "n_valid": self.n_valid,
            "n_pass_lipinski": self.n_pass_lipinski,
            "n_pass_qed": self.n_pass_qed,
            "n_pass_pains": self.n_pass_pains,
            "n_pass_sa": self.n_pass_sa,
            "n_pass_veber": self.n_pass_veber,
            "n_pass_all": self.n_pass_all,
        }


def load_filter_config(path: Path) -> dict:
    """Load filters.yaml and return the ``filters`` dict.

    Args:
        path: Path to filters.yaml.

    Returns:
        Parsed filter configuration dictionary.
    """
    with path.open() as f:
        raw = yaml.safe_load(f)
    return raw.get("filters", {})


def lipinski(mol: Chem.Mol, cfg: dict) -> bool:
    """Check Lipinski Rule of Five.

    Args:
        mol: RDKit molecule object.
        cfg: Lipinski config section with mw_max, logp_max, hbd_max, hba_max.

    Returns:
        True if molecule passes all four Lipinski criteria.
    """
    if not cfg.get("enabled", True):
        return True
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    return (
        mw <= cfg.get("mw_max", 500.0)
        and logp <= cfg.get("logp_max", 5.0)
        and hbd <= cfg.get("hbd_max", 5)
        and hba <= cfg.get("hba_max", 10)
    )


def qed_check(mol: Chem.Mol, cfg: dict) -> bool:
    """Check QED (quantitative estimate of drug-likeness) threshold.

    Args:
        mol: RDKit molecule object.
        cfg: QED config section with threshold_min.

    Returns:
        True if QED score >= threshold_min.
    """
    if not cfg.get("enabled", True):
        return True
    score = QED.qed(mol)
    return score >= cfg.get("threshold_min", 0.5)


def pains_check(mol: Chem.Mol) -> bool:
    """Check for PAINS (pan-assay interference compounds) substructures.

    Args:
        mol: RDKit molecule object.

    Returns:
        True if molecule has NO PAINS matches (i.e. passes).
    """
    return not _PAINS_CATALOG.HasMatch(mol)


def sa_check(mol: Chem.Mol, cfg: dict) -> bool:
    """Check synthetic accessibility score threshold.

    SA score ranges from 1 (easy) to 10 (hard).

    Args:
        mol: RDKit molecule object.
        cfg: SA score config section with threshold_max.

    Returns:
        True if SA score <= threshold_max.
    """
    if not cfg.get("enabled", True):
        return True
    score = sascorer.calculateScore(mol)
    return score <= cfg.get("threshold_max", 5.0)


def veber_check(mol: Chem.Mol, cfg: dict) -> bool:
    """Check Veber rules (oral bioavailability).

    Rotatable bonds <= 10 and TPSA <= 140 A^2.

    Args:
        mol: RDKit molecule object.
        cfg: Veber config section with rotatable_bonds_max, tpsa_max.

    Returns:
        True if molecule passes both Veber criteria.
    """
    if not cfg.get("enabled", True):
        return True
    rotatable = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    return rotatable <= cfg.get("rotatable_bonds_max", 10) and tpsa <= cfg.get("tpsa_max", 140.0)


def filter_one(smiles: str, cfg: dict) -> FilterResult:
    """Apply all drug-likeness filters to a single SMILES string.

    Args:
        smiles: SMILES string.
        cfg: Full filter configuration dict (from load_filter_config).

    Returns:
        FilterResult with per-filter pass/fail and computed properties.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return FilterResult(
            smiles=smiles,
            passes_lipinski=False,
            passes_qed=False,
            passes_pains=False,
            passes_sa=False,
            passes_veber=False,
            passes_all=False,
            mw=float("nan"),
            logp=float("nan"),
            hbd=0,
            hba=0,
            qed_score=float("nan"),
            sa_score=float("nan"),
            pains_matches=0,
            rotatable_bonds=float("nan"),
            tpsa=float("nan"),
        )

    lip_cfg = cfg.get("lipinski", {})
    qed_cfg = cfg.get("qed", {})
    pains_cfg = cfg.get("pains", {})
    sa_cfg = cfg.get("sa_score", {})
    veber_cfg = cfg.get("veber", {})

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    qed_score = QED.qed(mol)
    sa_score_val = sascorer.calculateScore(mol)
    pains_entries = _PAINS_CATALOG.GetMatches(mol)
    pains_matches = len(pains_entries)
    rotatable_bonds = float(Descriptors.NumRotatableBonds(mol))
    tpsa = Descriptors.TPSA(mol)

    pass_lip = lipinski(mol, lip_cfg)
    pass_qed = qed_check(mol, qed_cfg)
    pass_pains = pains_check(mol) if pains_cfg.get("enabled", True) else True
    pass_sa = sa_check(mol, sa_cfg)
    pass_veber = veber_check(mol, veber_cfg)
    pass_all = pass_lip and pass_qed and pass_pains and pass_sa and pass_veber

    return FilterResult(
        smiles=smiles,
        passes_lipinski=pass_lip,
        passes_qed=pass_qed,
        passes_pains=pass_pains,
        passes_sa=pass_sa,
        passes_veber=pass_veber,
        passes_all=pass_all,
        mw=mw,
        logp=logp,
        hbd=hbd,
        hba=hba,
        qed_score=qed_score,
        sa_score=sa_score_val,
        pains_matches=pains_matches,
        rotatable_bonds=rotatable_bonds,
        tpsa=tpsa,
    )


def compute_filters(smiles_list: list[str], cfg: dict) -> tuple[list[FilterResult], FilterStats]:
    """Apply drug-likeness filters to a batch of SMILES.

    Args:
        smiles_list: List of SMILES strings.
        cfg: Full filter configuration dict (from load_filter_config).

    Returns:
        Tuple of (list of FilterResult, FilterStats).
    """
    results: list[FilterResult] = []
    n_valid = 0
    n_lip = 0
    n_qed = 0
    n_pains = 0
    n_sa = 0
    n_veber = 0
    n_all = 0

    for smi in smiles_list:
        r = filter_one(smi, cfg)
        results.append(r)
        if Chem.MolFromSmiles(smi) is not None:
            n_valid += 1
        if r.passes_lipinski:
            n_lip += 1
        if r.passes_qed:
            n_qed += 1
        if r.passes_pains:
            n_pains += 1
        if r.passes_sa:
            n_sa += 1
        if r.passes_veber:
            n_veber += 1
        if r.passes_all:
            n_all += 1

    stats = FilterStats(
        n_total=len(smiles_list),
        n_valid=n_valid,
        n_pass_lipinski=n_lip,
        n_pass_qed=n_qed,
        n_pass_pains=n_pains,
        n_pass_sa=n_sa,
        n_pass_veber=n_veber,
        n_pass_all=n_all,
    )
    logger.info(
        "Filters: {}/{} valid, {}/{} pass all filters",
        n_valid,
        len(smiles_list),
        n_all,
        len(smiles_list),
    )
    return results, stats
