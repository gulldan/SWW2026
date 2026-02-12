"""Configuration loading utilities for the DualDocking pipeline.

Provides functions to load and validate YAML configuration files
(pairs.yaml, metrics.yaml, experiment.yaml) used throughout the project.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML contents.
    """
    with Path(path).open() as f:
        return yaml.safe_load(f)


def load_pairs(config_dir: str | Path = "configs") -> dict[str, Any]:
    """Load pairs.yaml with schema validation.

    Validates that each pair has ``target_a`` and ``target_b`` entries with
    ``id`` and ``organism`` fields. Non-blocked pairs must use Homo sapiens
    targets unless ``DUALDOCKING_ALLOW_NON_HUMAN`` is set.

    Args:
        config_dir: Directory containing pairs.yaml, or direct path to the
            YAML file.

    Returns:
        Parsed pairs configuration dict with a ``"pairs"`` key.

    Raises:
        ValueError: If the YAML is missing required keys or contains
            non-human targets without the override env var.
    """
    path = Path(config_dir)
    if path.is_dir():
        path = path / "pairs.yaml"
    data = load_yaml(path)
    if "pairs" not in data:
        raise ValueError("pairs.yaml must contain 'pairs' key")
    allow_non_human = os.getenv("DUALDOCKING_ALLOW_NON_HUMAN", "").lower() in ("true", "1")
    if allow_non_human:
        logger.warning("DUALDOCKING_ALLOW_NON_HUMAN=true — non-human targets allowed (exploratory mode)")

    for pair_id, pair_cfg in data["pairs"].items():
        for role in ("target_a", "target_b"):
            if role not in pair_cfg:
                raise ValueError(f"Pair '{pair_id}' missing '{role}'")
            if "id" not in pair_cfg[role]:
                raise ValueError(f"Pair '{pair_id}'.{role} missing 'id'")
            if "organism" not in pair_cfg[role]:
                raise ValueError(f"Pair '{pair_id}'.{role} missing 'organism'")
            # Skip organism check for blocked pairs — they are excluded from analysis.
            if pair_cfg.get("set") == "blocked":
                continue
            organism = pair_cfg[role]["organism"].strip()
            if organism.casefold() != "homo sapiens" and not allow_non_human:
                target_id = pair_cfg[role]["id"]
                raise ValueError(
                    f"Pair '{pair_id}'.{role} target '{target_id}' has organism='{organism}' "
                    f"(non-Homo sapiens). Only human targets allowed. "
                    f"Set DUALDOCKING_ALLOW_NON_HUMAN=true to bypass for exploratory runs."
                )
    return data


def load_metrics(config_dir: str | Path = "configs") -> dict[str, Any]:
    """Load metrics.yaml configuration.

    Args:
        config_dir: Directory containing metrics.yaml, or direct path.

    Returns:
        Parsed metrics configuration dict.
    """
    path = Path(config_dir)
    if path.is_dir():
        path = path / "metrics.yaml"
    return load_yaml(path)


def collect_blocked_target_ids(pairs: dict[str, Any]) -> set[str]:
    """Return target IDs belonging to blocked pairs.

    Args:
        pairs: The ``"pairs"`` dict from :func:`load_pairs`.

    Returns:
        Set of target IDs from blocked pairs.
    """
    blocked: set[str] = set()
    for pair_cfg in pairs.values():
        if pair_cfg.get("set") == "blocked":
            for role in ("target_a", "target_b"):
                blocked.add(pair_cfg[role]["id"])
    return blocked


def collect_all_target_ids(pairs: dict[str, Any]) -> list[str]:
    """Collect unique target IDs from all pairs.

    Args:
        pairs: The ``"pairs"`` dict from :func:`load_pairs`.

    Returns:
        Sorted list of unique target IDs.
    """
    ids: list[str] = []
    for pair_cfg in pairs.values():
        ids.append(pair_cfg["target_a"]["id"])
        ids.append(pair_cfg["target_b"]["id"])
    return sorted(set(ids))


def collect_target_pdb_map(pairs: dict[str, Any]) -> dict[str, str]:
    """Build a target_id → pdb_id mapping from pairs config.

    Args:
        pairs: The ``"pairs"`` dict from :func:`load_pairs`.

    Returns:
        Mapping from target ID to PDB accession code.
    """
    mapping: dict[str, str] = {}
    for pair_cfg in pairs.values():
        for role in ("target_a", "target_b"):
            t = pair_cfg[role]
            mapping[t["id"]] = t["pdb_id"]
    return mapping


def load_experiment(config_dir: str | Path = "configs") -> dict[str, Any]:
    """Load experiment.yaml configuration.

    Args:
        config_dir: Directory containing experiment.yaml, or direct path.

    Returns:
        Parsed experiment configuration dict.
    """
    path = Path(config_dir)
    if path.is_dir():
        path = path / "experiment.yaml"
    return load_yaml(path)
