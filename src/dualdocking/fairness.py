"""Fairness checks for generation budget and duplication."""

from __future__ import annotations

import importlib
import json
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


def validate_reward_quality(
    results_dir: str | Path,
    max_invalid_fraction: float = 0.10,
    max_timeout_fraction: float = 0.05,
) -> dict[str, Any]:
    """Validate reward quality for a single run directory.

    Reads ``freedpp_workspace/reward_metadata.json`` and checks that
    invalid/timeout fractions are within acceptable thresholds.
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
