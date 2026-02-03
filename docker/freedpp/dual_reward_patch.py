"""Dual-target reward patch for FREED++.

When the environment variable ``DUAL_RECEPTOR_B`` is set, this module
monkey-patches ``DockingVina.__call__`` so that each molecule is scored
against **two** receptors and the combined reward uses ``max(score_a, score_b)``
for docking energies (lower is better).

The patch is activated automatically on import when ``DUAL_RECEPTOR_B``
is present in the environment (see ``entrypoint.sh``).

Box parameters for the second receptor are read from
``/workspace/dual_config.json``::

    {"receptor_b": "/workspace/receptor_b.pdbqt", "center_b": "12.3,4.5,6.7", "size_b": "20,20,20"}
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_normalization(path: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("PyYAML required to load normalization.yaml") from exc

    with Path(path).open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError("normalization.yaml должен быть словарём")
    return data


def _apply_dual_patch() -> None:
    """Monkey-patch DockingVina to support single_a/single_b/double z-scored rewards."""
    reward_mode = os.environ.get("REWARD_MODE", "single_a")
    normalization_path = os.environ.get("NORMALIZATION_PATH", "/workspace/normalization.yaml")
    target_id_a = os.environ.get("TARGET_ID_A") or ""
    target_id_b = os.environ.get("TARGET_ID_B") or ""

    normalization = _load_normalization(normalization_path)
    targets = normalization.get("targets", {})

    if target_id_a not in targets:
        raise KeyError(f"Target A '{target_id_a}' не найден в normalization.yaml")

    mean_a = float(targets[target_id_a]["mean"])
    std_a = float(targets[target_id_a]["std"])

    dual_receptor_b = os.environ.get("DUAL_RECEPTOR_B")
    dual_cfg = None
    if reward_mode in {"single_b", "double"}:
        if not dual_receptor_b:
            raise RuntimeError("DUAL_RECEPTOR_B обязателен для single_b/double режимов")
        if target_id_b not in targets:
            raise KeyError(f"Target B '{target_id_b}' не найден в normalization.yaml")

        config_path = Path(os.environ.get("DUAL_CONFIG", "/workspace/dual_config.json"))
        with config_path.open() as f:
            dual_cfg = json.load(f)

    # Import after FREED++ is on sys.path
    try:
        from ffreed.envs.docking import DockingVina
    except ImportError:
        sys.path.insert(0, "/opt/freed")
        from ffreed.envs.docking import DockingVina

    _original_call = DockingVina.__call__

    def _zscore(value: float, mean: float, std: float) -> float:
        if std == 0:
            raise ZeroDivisionError("std=0 в normalization.yaml")
        return (value - mean) / std

    def _dual_call(self: Any, *args: Any, **kwargs: Any) -> float:
        """Score molecule against receptor A (and B when requested)."""
        # Score A — original receptor
        score_a = _original_call(self, *args, **kwargs)
        z_a = _zscore(score_a, mean_a, std_a)
        if abs(z_a) > 10:
            logger.warning("dual_reward_patch: |z_a| > 10 (%.2f)", z_a)

        if reward_mode == "single_a":
            return -z_a

        if dual_cfg is None:
            raise RuntimeError("dual_cfg отсутствует при режиме single_b/double")

        receptor_b = dual_cfg["receptor_b"]
        center_b = dual_cfg["center_b"]
        size_b = dual_cfg["size_b"]

        mean_b = float(targets[target_id_b]["mean"])
        std_b = float(targets[target_id_b]["std"])

        # Save original receptor state
        orig_receptor = self.receptor
        orig_center = self.box_center
        orig_size = self.box_size

        # Score B — second receptor
        try:
            self.receptor = receptor_b
            self.box_center = [float(x) for x in center_b.split(",")]
            self.box_size = [float(x) for x in size_b.split(",")]
            score_b = _original_call(self, *args, **kwargs)
        finally:
            # Restore original state
            self.receptor = orig_receptor
            self.box_center = orig_center
            self.box_size = orig_size

        z_b = _zscore(score_b, mean_b, std_b)
        if abs(z_b) > 10:
            logger.warning("dual_reward_patch: |z_b| > 10 (%.2f)", z_b)

        if reward_mode == "single_b":
            return -z_b

        # Combined reward (worst-case in z-space)
        return -max(z_a, z_b)

    DockingVina.__call__ = _dual_call
    logger.info("dual_reward_patch: patched DockingVina (reward_mode=%s)", reward_mode)


# Auto-apply on import
_apply_dual_patch()
