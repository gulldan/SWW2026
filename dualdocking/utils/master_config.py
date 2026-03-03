"""Helpers for reading values from ``configs/master_config.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def read_master_config_field(
    project_root: Path,
    path: str,
    *,
    default: str | None = None,
) -> str:
    """Read a dotted key path from ``configs/master_config.yaml``.

    Args:
        project_root: Repository/project root directory.
        path: Dot-separated key path (e.g. ``"software.boltz_version"``).
        default: Value returned when file/key is missing. If ``None``,
            missing entries raise.

    Returns:
        String value for *path* (or *default* when provided).

    Raises:
        FileNotFoundError: If config file is missing and *default* is ``None``.
        KeyError: If key path is missing and *default* is ``None``.
    """
    master_path = project_root / "configs" / "master_config.yaml"
    if not master_path.exists():
        if default is not None:
            return default
        msg = f"master_config.yaml not found: {master_path}"
        raise FileNotFoundError(msg)

    data = yaml.safe_load(master_path.read_text()) or {}
    value: Any = data
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = None
            break

    if value in (None, ""):
        if default is not None:
            return default
        msg = f"Field '{path}' not found in master_config.yaml"
        raise KeyError(msg)

    return str(value)
