"""Утилиты загрузки конфигурации."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Загрузить YAML-файл и вернуть содержимое как dict."""
    with Path(path).open() as f:
        return yaml.safe_load(f)


def load_pairs(config_dir: str | Path = "configs") -> dict[str, Any]:
    """Загрузить pairs.yaml."""
    return load_yaml(Path(config_dir) / "pairs.yaml")


def load_metrics(config_dir: str | Path = "configs") -> dict[str, Any]:
    """Загрузить metrics.yaml."""
    return load_yaml(Path(config_dir) / "metrics.yaml")


def load_experiment(config_dir: str | Path = "configs") -> dict[str, Any]:
    """Загрузить experiment.yaml."""
    return load_yaml(Path(config_dir) / "experiment.yaml")
