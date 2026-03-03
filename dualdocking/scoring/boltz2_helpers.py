"""Reusable Boltz-2 helper functions extracted from ``boltz2.py``."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from loguru import logger


class _WarnLogger(Protocol):
    """Minimal logger protocol used by helper functions."""

    def warning(self, message: str, *args: object) -> None:
        """Log warning message.

        Args:
            message: Log message template/text.
            *args: Optional format arguments.
        """

    def error(self, message: str, *args: object) -> None:
        """Log error message.

        Args:
            message: Log message template/text.
            *args: Optional format arguments.
        """


def msa_content_hash(msa_path: Path | None) -> str:
    """Compute a short hash for MSA content.

    Args:
        msa_path: MSA CSV path or ``None``.

    Returns:
        First 16 hex chars of SHA-256(file bytes), or ``"no_msa"``.
    """
    if msa_path is None:
        return "no_msa"
    return hashlib.sha256(msa_path.read_bytes()).hexdigest()[:16]


def sequence_content_hash(sequence: str) -> str:
    """Compute a short hash for protein sequence.

    Args:
        sequence: Amino-acid sequence.

    Returns:
        First 16 hex chars of SHA-256(sequence).
    """
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]


def cache_key(
    target_id: str,
    smiles: str,
    *,
    ctx_hash: str = "",
    seq_hash: str = "",
    msa_hash: str = "",
) -> str:
    """Build deterministic SHA-256 cache key for ``(target, smiles)``.

    Args:
        target_id: Target identifier.
        smiles: Input SMILES.
        ctx_hash: Context hash segment.
        seq_hash: Sequence hash segment.
        msa_hash: MSA hash segment.

    Returns:
        Full SHA-256 hex digest.
    """
    blob = f"{target_id}||{smiles}||{ctx_hash}||{seq_hash}||{msa_hash}".encode()
    return hashlib.sha256(blob).hexdigest()


def require_msa_path(msa_cache_dir: Path, target_id: str) -> Path:
    """Return required pre-computed MSA path for a target.

    Args:
        msa_cache_dir: Directory with precomputed MSA CSV files.
        target_id: Target identifier.

    Returns:
        Existing MSA path for ``target_id``.

    Raises:
        FileNotFoundError: If ``{msa_cache_dir}/{target_id}.csv`` is absent.
    """
    msa_path = msa_cache_dir / f"{target_id}.csv"
    if msa_path.exists():
        return msa_path
    msg = (
        f"Pre-computed MSA not found for {target_id}. "
        "Run: uv run python cli/s1_prepare/prepare_targets.py --pair <pair> --prepare-all"
    )
    raise FileNotFoundError(msg)


def collect_valid_smiles(
    smiles_list: list[str],
    *,
    validate_smiles: Callable[[str], bool],
    neutralize_smiles: Callable[[str], str],
    max_ligand_atoms: int,
    warning_template: str,
    log: _WarnLogger,
) -> list[tuple[int, str]]:
    """Collect valid neutralized SMILES while preserving indices.

    Args:
        smiles_list: Raw input SMILES.
        validate_smiles: Validation callback.
        neutralize_smiles: Neutralization callback.
        max_ligand_atoms: Max atom count used in warning formatting.
        warning_template: Warning template with ``{}`` placeholders.
        log: Logger-like instance.

    Returns:
        List of ``(index, neutralized_smiles)``.
    """
    valid_pairs: list[tuple[int, str]] = []
    for idx, smiles in enumerate(smiles_list):
        if validate_smiles(smiles):
            valid_pairs.append((idx, neutralize_smiles(smiles)))
        else:
            log.warning(warning_template, max_ligand_atoms, smiles)
    return valid_pairs


def write_boltz_yaml(
    sequence: str,
    smiles: str,
    path: Path,
    *,
    msa_path: Path | None = None,
) -> None:
    """Write Boltz-2 single-complex YAML input.

    Args:
        sequence: Protein sequence.
        smiles: Ligand SMILES.
        path: Destination YAML path.
        msa_path: Optional MSA CSV path.
    """
    import yaml

    protein_entry: dict[str, str] = {
        "id": "A",
        "sequence": sequence,
    }
    if msa_path is not None:
        protein_entry["msa"] = str(msa_path.resolve())

    data = {
        "version": 1,
        "sequences": [
            {"protein": protein_entry},
            {"ligand": {"id": "B", "smiles": smiles}},
        ],
        "properties": [
            {"affinity": {"binder": "B"}},
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


def parse_affinity_file(affinity_path: Path) -> float:
    """Parse ``affinity_pred_value`` from affinity JSON file.

    Args:
        affinity_path: Path to affinity JSON file.

    Returns:
        Parsed affinity value.

    Raises:
        FileNotFoundError: If file does not exist.
        KeyError: If ``affinity_pred_value`` is missing.
    """
    if not affinity_path.exists():
        msg = f"Affinity JSON not found: {affinity_path}"
        raise FileNotFoundError(msg)
    data = json.loads(affinity_path.read_text())
    return float(data["affinity_pred_value"])


def parse_affinity(out_dir: Path, stem: str) -> float:
    """Parse affinity for a given input stem from Boltz output directory.

    Args:
        out_dir: Boltz output root directory.
        stem: Input stem name.

    Returns:
        Parsed affinity value.
    """
    affinity_path = out_dir / f"boltz_results_{stem}" / "predictions" / stem / f"affinity_{stem}.json"
    return parse_affinity_file(affinity_path)


def nim_payload(
    sequence: str,
    smiles: str,
    *,
    predict_params: dict[str, int | float | str],
    msa_content: str | None = None,
) -> dict[str, Any]:
    """Build NIM request payload for one protein-ligand pair.

    Args:
        sequence: Protein sequence.
        smiles: Ligand SMILES.
        predict_params: Backend request parameters.
        msa_content: Optional raw CSV content from ColabFold MSA file.

    Returns:
        Request payload dictionary.
    """
    protein_entry: dict[str, Any] = {"id": "A", "molecule_type": "protein", "sequence": sequence}
    if msa_content is not None:
        protein_entry["msa"] = {
            "colabfold": {"csv": {"format": "csv", "alignment": msa_content}},
        }
    return {
        "polymers": [protein_entry],
        "ligands": [{"id": "B", "smiles": smiles, "predict_affinity": True}],
        **predict_params,
    }


def parse_nim_affinity(
    result: dict[str, Any],
    smiles: str,
    *,
    log: _WarnLogger | None = None,
) -> float:
    """Extract averaged NIM affinity from response payload.

    Args:
        result: Decoded NIM response payload.
        smiles: SMILES used for context in log messages.
        log: Logger-like instance.

    Returns:
        Mean affinity score, or NaN on parse failures.
    """
    active_log = log if log is not None else cast("_WarnLogger", logger)
    try:
        affinities = result["affinities"]["B"]
        pred_values = affinities["affinity_pred_value"]
        return float(np.mean(pred_values))
    except (KeyError, TypeError, IndexError) as exc:
        active_log.error(f"Failed to parse NIM affinity for SMILES {smiles}: {exc}")
        return float("nan")
