"""Утилиты загрузки конфигурации."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Загрузить YAML-файл и вернуть содержимое как dict."""
    with Path(path).open() as f:
        return yaml.safe_load(f)


def load_pairs(config_dir: str | Path = "configs") -> dict[str, Any]:
    """Загрузить pairs.yaml с валидацией структуры."""
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
    """Загрузить metrics.yaml."""
    path = Path(config_dir)
    if path.is_dir():
        path = path / "metrics.yaml"
    return load_yaml(path)


def load_experiment(config_dir: str | Path = "configs") -> dict[str, Any]:
    """Загрузить experiment.yaml."""
    path = Path(config_dir)
    if path.is_dir():
        path = path / "experiment.yaml"
    return load_yaml(path)
