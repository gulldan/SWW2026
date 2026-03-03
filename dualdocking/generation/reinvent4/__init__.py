"""REINVENT4 SMILES-based RL generation backend.

Re-exports public API from submodules for backward compatibility.
"""

from dualdocking.generation.reinvent4.core import (
    CACHE_DIR,
    DATA_DIR,
    PROJECT_ROOT,
    Reinvent4GenerationConfig,
    _cache_key,
    _load_normalization,
    _parse_output_smiles,
    generate,
)
from dualdocking.generation.reinvent4.scoring import RewardStats, _reward_stats

__all__ = [
    "CACHE_DIR",
    "DATA_DIR",
    "PROJECT_ROOT",
    "Reinvent4GenerationConfig",
    "RewardStats",
    "_cache_key",
    "_load_normalization",
    "_parse_output_smiles",
    "_reward_stats",
    "generate",
]
