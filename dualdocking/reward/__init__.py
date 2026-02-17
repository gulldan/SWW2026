"""Reward utilities and pluggable selectivity scalarization modes."""

from dualdocking.reward.selectivity import (
    ResolvedRewardMode,
    RewardModeDefinition,
    evaluate_reward_mode,
    get_reward_mode,
    get_reward_modes_config_source,
    list_reward_modes,
    load_reward_modes_from_yaml,
    load_reward_plugins,
    register_reward_mode,
    resolve_reward_mode,
    reward_mode_requires_dual_docking,
    reward_mode_requires_secondary_target,
)

__all__ = [
    "ResolvedRewardMode",
    "RewardModeDefinition",
    "evaluate_reward_mode",
    "get_reward_mode",
    "get_reward_modes_config_source",
    "list_reward_modes",
    "load_reward_modes_from_yaml",
    "load_reward_plugins",
    "register_reward_mode",
    "resolve_reward_mode",
    "reward_mode_requires_dual_docking",
    "reward_mode_requires_secondary_target",
]
