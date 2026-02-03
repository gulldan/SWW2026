"""Fairness checks for generation budget and duplication."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

from loguru import logger


def _fallback_canonicalize(smi: str) -> str | None:
    smi = smi.strip()
    return smi or None


def _load_canonicalize() -> Callable[[str], str | None]:
    try:
        module = importlib.import_module("dualdocking.chem_sanitize")
        return module.canonicalize
    except Exception:  # pragma: no cover - fallback when RDKit is unavailable
        return _fallback_canonicalize


canonicalize_fn: Callable[[str], str | None] = _load_canonicalize()


def _deduplicate(smiles: list[str]) -> tuple[int, int]:
    """Return (n_total, n_unique) with optional canonicalization."""
    n_total = len(smiles)
    if n_total == 0:
        return 0, 0

    seen: set[str] = set()
    for smi in smiles:
        canon = canonicalize_fn(smi.strip())
        if canon is None:
            continue
        seen.add(canon)
    return n_total, len(seen)


def validate_fairness(results_dir: str | Path) -> dict[str, Any]:
    """Validate generation fairness for a single run directory.

    Checks:
    - n_molecules in [990, 1010]
    - duplication_rate < 0.20
    """
    results_path = Path(results_dir)
    issues: list[str] = []

    generated_csv = results_path / "generated.csv"
    if not generated_csv.exists():
        issues.append(f"generated.csv не найден: {generated_csv}")
        return {
            "budget_consistent": False,
            "issues": issues,
            "n_molecules": 0,
            "n_molecules_unique": 0,
            "duplication_rate": 1.0,
        }

    # Read smiles
    lines = generated_csv.read_text().strip().splitlines()
    smiles = [line.split(",")[0] for line in lines[1:] if line.strip()]

    n_total, n_unique = _deduplicate(smiles)
    duplication_rate = 0.0 if n_total == 0 else 1.0 - (n_unique / n_total)

    if not (990 <= n_total <= 1010):
        issues.append(f"n_molecules вне диапазона [990, 1010]: {n_total}")
    if duplication_rate >= 0.20:
        issues.append(f"duplication_rate >= 0.20: {duplication_rate:.3f}")

    if issues:
        logger.warning("Fairness issues in {}: {}", results_path, "; ".join(issues))

    return {
        "budget_consistent": len(issues) == 0,
        "issues": issues,
        "n_molecules": n_total,
        "n_molecules_unique": n_unique,
        "duplication_rate": round(duplication_rate, 4),
    }
