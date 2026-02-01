"""Dual-target reward patch for FREED++.

When the environment variable ``DUAL_RECEPTOR_B`` is set, this module
monkey-patches ``DockingVina.__call__`` so that each molecule is scored
against **two** receptors and the combined reward is ``min(score_a, score_b)``.

The patch is activated automatically on import when ``DUAL_RECEPTOR_B``
is present in the environment (see ``entrypoint.sh``).

Box parameters for the second receptor are read from
``/workspace/dual_config.json``::

    {"receptor_b": "/workspace/receptor_b.pdbqt", "center_b": "12.3,4.5,6.7", "size_b": "20,20,20"}
"""

from __future__ import annotations

import json
import os
import sys


def _apply_dual_patch() -> None:
    """Monkey-patch DockingVina to score against two receptors."""
    dual_receptor_b = os.environ.get("DUAL_RECEPTOR_B")
    if not dual_receptor_b:
        return

    config_path = os.environ.get("DUAL_CONFIG", "/workspace/dual_config.json")
    with open(config_path) as f:
        dual_cfg = json.load(f)

    receptor_b = dual_cfg["receptor_b"]
    center_b = dual_cfg["center_b"]
    size_b = dual_cfg["size_b"]

    # Import after FREED++ is on sys.path
    try:
        from ffreed.envs.docking import DockingVina
    except ImportError:
        sys.path.insert(0, "/opt/freed")
        from ffreed.envs.docking import DockingVina

    _original_call = DockingVina.__call__

    def _dual_call(self, *args, **kwargs):
        """Score molecule against receptor A (original) then receptor B."""
        # Score A — original receptor
        score_a = _original_call(self, *args, **kwargs)

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

        # Combined reward: worst-case for docking energies (lower = better)
        # Для минимизации по энергии это max(score_a, score_b)
        return max(score_a, score_b)

    DockingVina.__call__ = _dual_call
    print(f"[dual_reward_patch] Patched DockingVina for dual-target: {receptor_b}")


# Auto-apply on import
_apply_dual_patch()
