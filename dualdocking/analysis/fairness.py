"""Fairness checks for generation budget and duplication.

Validates that each RL run produced a consistent molecule count and that
the duplication rate stays within acceptable bounds. Also checks reward
quality metadata (invalid/timeout fractions).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable

from loguru import logger


def _fallback_canonicalize(smi: str) -> str | None:
    """Strip-only canonicalization used when RDKit is unavailable.

    Args:
        smi: Raw SMILES string.

    Returns:
        Stripped SMILES, or ``None`` if blank.
    """
    smi = smi.strip()
    return smi or None


def _load_canonicalize() -> Callable[[str], str | None]:
    """Load the canonical SMILES function, falling back to strip-only.

    Returns:
        A callable ``(str) -> str | None`` for SMILES canonicalization.
    """
    try:
        module = importlib.import_module("dualdocking.data.chem_sanitize")
        return module.canonicalize
    except Exception:  # pragma: no cover - fallback when RDKit is unavailable
        return _fallback_canonicalize


canonicalize_fn: Callable[[str], str | None] = _load_canonicalize()


def _deduplicate(smiles: list[str]) -> tuple[int, int]:
    """Count total and unique SMILES with optional canonicalization.

    Args:
        smiles: Raw SMILES strings.

    Returns:
        Tuple of ``(n_total, n_unique)``.
    """
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

    Checks that the molecule count is within [990, 1010] and that the
    duplication rate is below 20 %.

    Args:
        results_dir: Path to a single run's results directory containing
            ``generated.parquet`` (or ``generated.csv`` fallback).

    Returns:
        Dict with keys ``budget_consistent``, ``issues``, ``n_molecules``,
        ``n_molecules_unique``, and ``duplication_rate``.
    """
    results_path = Path(results_dir)
    issues: list[str] = []

    from dualdocking.data.io_utils import read_df, stem_exists

    generated_stem = results_path / "generated"
    if not stem_exists(generated_stem):
        issues.append(f"generated.parquet/csv not found in {results_path}")
        return {
            "budget_consistent": False,
            "issues": issues,
            "n_molecules": 0,
            "n_molecules_unique": 0,
            "duplication_rate": 1.0,
        }

    # Read smiles
    try:
        gen_df = read_df(generated_stem)
        smiles = gen_df["smiles"].tolist()
    except Exception as exc:
        issues.append(f"Error reading generated: {exc}")
        return {
            "budget_consistent": False,
            "issues": issues,
            "n_molecules": 0,
            "n_molecules_unique": 0,
            "duplication_rate": 1.0,
        }

    n_total, n_unique = _deduplicate(smiles)
    duplication_rate = 0.0 if n_total == 0 else 1.0 - (n_unique / n_total)

    if not (990 <= n_total <= 1010):
        issues.append(f"n_molecules out of range [990, 1010]: {n_total}")
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


def validate_reward_quality(
    results_dir: str | Path,
    max_invalid_fraction: float = 0.10,
    max_timeout_fraction: float = 0.05,
) -> dict[str, Any]:
    """Validate reward quality for a single run directory.

    Reads ``freedpp_workspace/reward_metadata.json`` and checks that
    invalid/timeout fractions are within acceptable thresholds.

    Args:
        results_dir: Path to a single run's results directory.
        max_invalid_fraction: Maximum allowed fraction of invalid rewards.
        max_timeout_fraction: Maximum allowed fraction of timeouts.

    Returns:
        Dict with keys ``run_valid``, ``issues``, ``total_calls``, and
        per-target invalid/timeout fractions.
    """
    results_path = Path(results_dir)
    metadata_path = results_path / "freedpp_workspace" / "reward_metadata.json"
    issues: list[str] = []

    if not metadata_path.exists():
        return {
            "run_valid": False,
            "issues": ["reward_metadata.json not found"],
            "total_calls": 0,
            "invalid_fraction_a": 0.0,
            "invalid_fraction_b": 0.0,
            "timeout_fraction_a": 0.0,
            "timeout_fraction_b": 0.0,
        }

    try:
        raw = json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return {
            "run_valid": False,
            "issues": [f"malformed reward_metadata.json: {exc}"],
            "total_calls": 0,
            "invalid_fraction_a": 0.0,
            "invalid_fraction_b": 0.0,
            "timeout_fraction_a": 0.0,
            "timeout_fraction_b": 0.0,
        }

    total_calls = int(raw.get("total_calls", 0))

    frac_keys = {
        "invalid_fraction_a": "invalid_reward_fraction_a",
        "invalid_fraction_b": "invalid_reward_fraction_b",
        "timeout_fraction_a": "timeout_fraction_a",
        "timeout_fraction_b": "timeout_fraction_b",
    }
    fracs: dict[str, float] = {}
    for out_key, raw_key in frac_keys.items():
        val = raw.get(raw_key)
        if val is None:
            issues.append(f"missing key: {raw_key}")
            fracs[out_key] = 0.0
        else:
            fracs[out_key] = float(val)

    if fracs["invalid_fraction_a"] > max_invalid_fraction:
        issues.append(f"invalid_reward_fraction_a={fracs['invalid_fraction_a']:.3f} > {max_invalid_fraction}")
    if fracs["invalid_fraction_b"] > max_invalid_fraction:
        issues.append(f"invalid_reward_fraction_b={fracs['invalid_fraction_b']:.3f} > {max_invalid_fraction}")
    if fracs["timeout_fraction_a"] > max_timeout_fraction:
        issues.append(f"timeout_fraction_a={fracs['timeout_fraction_a']:.3f} > {max_timeout_fraction}")
    if fracs["timeout_fraction_b"] > max_timeout_fraction:
        issues.append(f"timeout_fraction_b={fracs['timeout_fraction_b']:.3f} > {max_timeout_fraction}")

    return {
        "run_valid": len(issues) == 0,
        "issues": issues,
        "total_calls": total_calls,
        **fracs,
    }
