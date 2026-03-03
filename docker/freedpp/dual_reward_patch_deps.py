"""Compatibility imports for the FREED++ dual reward patch.

This module centralizes fallback imports so patch helpers can run both:
- inside the project package (``dualdocking.*`` imports), and
- inside the Docker image where modules are copied to ``/opt``.
"""

from __future__ import annotations

import sys

# mol_prep: try installed package first, fall back to /opt (Docker COPY).
try:
    from dualdocking.data.mol_prep import prepare_ligand_pdbqt
except ImportError:
    if "/opt" not in sys.path:
        sys.path.insert(0, "/opt")
    from mol_prep import prepare_ligand_pdbqt  # type: ignore[no-redef]

# Shared reward scalarization: try project package first, then /opt copy.
try:
    from dualdocking.reward.selectivity import (
        ResolvedRewardMode,
        get_reward_modes_config_source,
        resolve_reward_mode,
    )
except ImportError:
    if "/opt" not in sys.path:
        sys.path.insert(0, "/opt")
    from reward_selectivity import (  # type: ignore[no-redef]
        ResolvedRewardMode,
        get_reward_modes_config_source,
        resolve_reward_mode,
    )

__all__ = [
    "ResolvedRewardMode",
    "get_reward_modes_config_source",
    "prepare_ligand_pdbqt",
    "resolve_reward_mode",
]
