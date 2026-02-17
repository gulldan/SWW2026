"""Shared selectivity reward functions with plugin-style mode registry.

This module centralizes reward scalarization logic so multiple generators
(RL and non-RL) can reuse the same definitions.
"""

from __future__ import annotations

import importlib
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_EPS = 1e-12
_PLUGIN_ENV_VAR = "DUALDOCKING_REWARD_PLUGINS"
_PLUGIN_ENTRYPOINT = "register_reward_modes"
_REWARD_MODES_PATH_ENV_VAR = "DUALDOCKING_REWARD_MODES_PATH"
_REWARD_MODES_FILENAME = "reward_modes.yaml"


RewardEvaluator = Callable[[float | None, float | None, Mapping[str, float]], float]


@dataclass(frozen=True)
class RewardModeDefinition:
    """Definition of one reward mode in the registry.

    Attributes:
        name: Unique mode name.
        requires_secondary_target: Whether target-B normalization is required.
        requires_dual_docking: Whether both z_a and z_b must be scored per call.
        default_params: Default numeric parameters for the mode.
        evaluator: Callable that maps z-scores to a scalar reward.
    """

    name: str
    requires_secondary_target: bool
    requires_dual_docking: bool
    default_params: Mapping[str, float]
    evaluator: RewardEvaluator


@dataclass(frozen=True)
class ResolvedRewardMode:
    """Resolved reward mode with concrete parameter values.

    Attributes:
        definition: Base mode definition from registry.
        params: Effective numeric parameters (defaults merged with overrides).
    """

    definition: RewardModeDefinition
    params: Mapping[str, float]

    @property
    def name(self) -> str:
        """Return mode name."""
        return self.definition.name

    @property
    def requires_secondary_target(self) -> bool:
        """Whether target-B normalization is required."""
        return self.definition.requires_secondary_target

    @property
    def requires_dual_docking(self) -> bool:
        """Whether both z_a and z_b are required per reward call."""
        return self.definition.requires_dual_docking

    def evaluate(self, *, z_a: float | None = None, z_b: float | None = None) -> float:
        """Evaluate reward from z-scores.

        Args:
            z_a: z-score for target A.
            z_b: z-score for target B.

        Returns:
            Scalar reward value.
        """
        return self.definition.evaluator(z_a, z_b, self.params)


_REGISTRY: dict[str, RewardModeDefinition] = {}
_PLUGIN_SPECS_LOADED: set[str] = set()
_REWARD_MODES_CONFIG_SOURCE: str | None = None


def _require_value(value: float | None, *, name: str) -> float:
    """Require a score value and validate finiteness."""
    if value is None:
        msg = f"{name} is required for this reward mode"
        raise ValueError(msg)
    if not math.isfinite(value):
        msg = f"{name} must be finite, got {value}"
        raise ValueError(msg)
    return value


def _param(params: Mapping[str, float], key: str, default: float) -> float:
    """Read a finite float parameter with fallback."""
    value = params.get(key, default)
    if not math.isfinite(value):
        msg = f"Parameter '{key}' must be finite, got {value}"
        raise ValueError(msg)
    return float(value)


def _single_a_eval(z_a: float | None, z_b: float | None, params: Mapping[str, float]) -> float:  # noqa: ARG001
    return _require_value(z_a, name="z_a")


def _single_b_eval(z_a: float | None, z_b: float | None, params: Mapping[str, float]) -> float:  # noqa: ARG001
    return _require_value(z_b, name="z_b")


def _double_eval(z_a: float | None, z_b: float | None, params: Mapping[str, float]) -> float:  # noqa: ARG001
    za = _require_value(z_a, name="z_a")
    zb = _require_value(z_b, name="z_b")
    return max(za, zb)


def _linear_equal_eval(z_a: float | None, z_b: float | None, params: Mapping[str, float]) -> float:  # noqa: ARG001
    za = _require_value(z_a, name="z_a")
    zb = _require_value(z_b, name="z_b")
    return za + zb


def _biased_a_eval(z_a: float | None, z_b: float | None, params: Mapping[str, float]) -> float:
    za = _require_value(z_a, name="z_a")
    zb = _require_value(z_b, name="z_b")
    weight_b = _param(params, "weight_b", 0.3)
    return za + weight_b * zb


def _selective_a_eval(z_a: float | None, z_b: float | None, params: Mapping[str, float]) -> float:  # noqa: ARG001
    za = _require_value(z_a, name="z_a")
    zb = _require_value(z_b, name="z_b")
    return za - zb


def _selective_weighted_eval(z_a: float | None, z_b: float | None, params: Mapping[str, float]) -> float:
    """Evaluate z_on - lambda_off * z_off."""
    za = _require_value(z_a, name="z_a")
    zb = _require_value(z_b, name="z_b")
    lambda_off = _param(params, "lambda_off", 1.0)
    if lambda_off <= _EPS:
        msg = f"lambda_off must be > 0, got {lambda_off}"
        raise ValueError(msg)
    return za - lambda_off * zb


_EVALUATOR_BY_KIND: dict[str, RewardEvaluator] = {
    "single_a": _single_a_eval,
    "single_b": _single_b_eval,
    "double": _double_eval,
    "linear_equal": _linear_equal_eval,
    "biased_a": _biased_a_eval,
    "selective_a": _selective_a_eval,
    "selective_weighted": _selective_weighted_eval,
}


def _normalize_params(params: Mapping[str, Any] | None) -> dict[str, float]:
    """Convert user params to validated float dict."""
    if params is None:
        return {}
    normalized: dict[str, float] = {}
    for key, value in params.items():
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            msg = f"Reward parameter '{key}' is not a number: {value!r}"
            raise ValueError(msg) from exc
        if not math.isfinite(number):
            msg = f"Reward parameter '{key}' must be finite, got {value!r}"
            raise ValueError(msg)
        normalized[str(key)] = number
    return normalized


def register_reward_mode(
    *,
    name: str,
    evaluator: RewardEvaluator,
    requires_secondary_target: bool,
    requires_dual_docking: bool,
    default_params: Mapping[str, float] | None = None,
    allow_override: bool = False,
) -> None:
    """Register a reward mode.

    Args:
        name: Unique mode name.
        evaluator: Callable combining z-scores into one reward.
        requires_secondary_target: Whether target-B normalization is required.
        requires_dual_docking: Whether both z_a and z_b must be scored.
        default_params: Optional default numeric params.
        allow_override: Whether to replace existing registration.
    """
    mode_name = name.strip()
    if not mode_name:
        raise ValueError("Reward mode name must be non-empty")
    if mode_name in _REGISTRY and not allow_override:
        msg = f"Reward mode '{mode_name}' is already registered"
        raise ValueError(msg)

    params = _normalize_params(default_params)
    _REGISTRY[mode_name] = RewardModeDefinition(
        name=mode_name,
        requires_secondary_target=requires_secondary_target,
        requires_dual_docking=requires_dual_docking,
        default_params=params,
        evaluator=evaluator,
    )


def _as_bool(value: Any, *, key: str, mode_name: str) -> bool:
    """Validate strict bool values from YAML."""
    if not isinstance(value, bool):
        msg = f"Mode '{mode_name}' key '{key}' must be boolean, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _reward_modes_candidates() -> list[Path]:
    """Return candidate config paths in priority order."""
    candidates: list[Path] = []

    env_path = os.environ.get(_REWARD_MODES_PATH_ENV_VAR, "").strip()
    if env_path:
        candidates.append(Path(env_path))

    module_dir = Path(__file__).resolve().parent
    candidates.append(module_dir / _REWARD_MODES_FILENAME)
    candidates.append(module_dir.parent.parent / "configs" / _REWARD_MODES_FILENAME)
    candidates.append(Path.cwd() / "configs" / _REWARD_MODES_FILENAME)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _resolve_reward_modes_path(path: str | Path | None = None) -> Path | None:
    """Resolve reward-modes YAML path.

    Args:
        path: Explicit YAML path. If omitted, use default search candidates.

    Returns:
        Existing path or ``None`` when not found.

    Raises:
        FileNotFoundError: Explicit *path* does not exist.
    """
    if path is not None:
        explicit = Path(path)
        if not explicit.exists():
            raise FileNotFoundError(f"Reward modes config not found: {explicit}")
        return explicit

    for candidate in _reward_modes_candidates():
        if candidate.exists():
            return candidate
    return None


def get_reward_modes_config_source() -> str | None:
    """Return the source path currently used to register built-in modes."""
    return _REWARD_MODES_CONFIG_SOURCE


def load_reward_modes_from_yaml(
    path: str | Path | None = None,
    *,
    allow_override: bool = True,
) -> str | None:
    """Load reward-mode definitions from YAML and register them.

    YAML schema:
        version: 1
        modes:
          mode_name:
            kind: selective_weighted
            requires_secondary_target: true
            requires_dual_docking: true
            default_params:
              lambda_off: 1.0

    Args:
        path: Optional explicit path to ``reward_modes.yaml``.
        allow_override: Whether YAML modes may overwrite existing entries.

    Returns:
        Path string used to load modes, or ``None`` if no config was found.
    """
    global _REWARD_MODES_CONFIG_SOURCE

    config_path = _resolve_reward_modes_path(path)
    if config_path is None:
        return None

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("PyYAML is required to load reward modes config") from exc

    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Reward modes config must be a mapping: {config_path}")

    raw_modes = payload.get("modes")
    if not isinstance(raw_modes, Mapping):
        raise TypeError(f"Reward modes config must contain mapping key 'modes': {config_path}")

    for raw_name, raw_cfg in raw_modes.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"Reward mode name must be a non-empty string: {raw_name!r}")
        if not isinstance(raw_cfg, Mapping):
            raise TypeError(f"Mode '{raw_name}' config must be a mapping")

        kind_obj = raw_cfg.get("kind", raw_name)
        if not isinstance(kind_obj, str) or not kind_obj.strip():
            raise ValueError(f"Mode '{raw_name}' has invalid kind: {kind_obj!r}")
        kind = kind_obj.strip()
        evaluator = _EVALUATOR_BY_KIND.get(kind)
        if evaluator is None:
            available = ", ".join(sorted(_EVALUATOR_BY_KIND))
            raise ValueError(f"Mode '{raw_name}' uses unknown kind '{kind}'. Available kinds: {available}")

        needs_secondary = _as_bool(
            raw_cfg.get("requires_secondary_target"),
            key="requires_secondary_target",
            mode_name=raw_name,
        )
        needs_dual = _as_bool(
            raw_cfg.get("requires_dual_docking"),
            key="requires_dual_docking",
            mode_name=raw_name,
        )
        if needs_dual and not needs_secondary:
            raise ValueError(
                f"Mode '{raw_name}' cannot set requires_dual_docking=true with requires_secondary_target=false",
            )

        raw_params = raw_cfg.get("default_params", {})
        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, Mapping):
            raise TypeError(f"Mode '{raw_name}' key 'default_params' must be a mapping")

        register_reward_mode(
            name=raw_name,
            evaluator=evaluator,
            requires_secondary_target=needs_secondary,
            requires_dual_docking=needs_dual,
            default_params=raw_params,
            allow_override=allow_override,
        )

    _REWARD_MODES_CONFIG_SOURCE = str(config_path)
    return _REWARD_MODES_CONFIG_SOURCE


def load_reward_plugins(specs: str | None = None) -> list[str]:
    """Load reward plugins from module specs.

    Each spec is either ``"package.module"`` (import side effects or
    ``register_reward_modes()`` entrypoint) or
    ``"package.module:callable_name"``.

    Args:
        specs: Comma-separated plugin specs. When ``None``, reads
            ``DUALDOCKING_REWARD_PLUGINS`` from environment.

    Returns:
        Plugin specs activated in this call (already-loaded specs omitted).
    """
    raw_specs = specs if specs is not None else os.environ.get(_PLUGIN_ENV_VAR, "")
    loaded_now: list[str] = []
    for raw_spec in raw_specs.split(","):
        spec = raw_spec.strip()
        if not spec or spec in _PLUGIN_SPECS_LOADED:
            continue

        module_name, sep, entrypoint_name = spec.partition(":")
        if not module_name:
            raise ValueError(f"Invalid reward plugin spec: {spec!r}")

        module = importlib.import_module(module_name)
        entrypoint = entrypoint_name if sep else _PLUGIN_ENTRYPOINT
        callback = getattr(module, entrypoint, None)

        if callback is None and sep:
            msg = f"Reward plugin '{spec}' is missing callable '{entrypoint}'"
            raise AttributeError(msg)

        if callback is not None:
            if not callable(callback):
                msg = f"Reward plugin '{spec}' attribute '{entrypoint}' is not callable"
                raise TypeError(msg)
            callback()

        _PLUGIN_SPECS_LOADED.add(spec)
        loaded_now.append(spec)

    return loaded_now


def _ensure_env_plugins_loaded() -> None:
    """Load environment-configured plugins once per process."""
    load_reward_plugins()


def list_reward_modes() -> list[str]:
    """List registered reward mode names."""
    _ensure_env_plugins_loaded()
    return sorted(_REGISTRY)


def get_reward_mode(name: str) -> RewardModeDefinition:
    """Get reward mode definition by name.

    Raises:
        KeyError: If mode is unknown.
    """
    _ensure_env_plugins_loaded()
    mode = _REGISTRY.get(name)
    if mode is None:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown reward mode '{name}'. Available: {available}")
    return mode


def resolve_reward_mode(name: str, params: Mapping[str, Any] | None = None) -> ResolvedRewardMode:
    """Resolve mode name and parameter overrides into executable mode."""
    definition = get_reward_mode(name)
    merged = dict(definition.default_params)
    merged.update(_normalize_params(params))
    return ResolvedRewardMode(definition=definition, params=merged)


def reward_mode_requires_secondary_target(name: str) -> bool:
    """Return whether mode requires target-B normalization."""
    return get_reward_mode(name).requires_secondary_target


def reward_mode_requires_dual_docking(name: str) -> bool:
    """Return whether mode requires scoring both targets in one call."""
    return get_reward_mode(name).requires_dual_docking


def evaluate_reward_mode(
    name: str,
    *,
    z_a: float | None = None,
    z_b: float | None = None,
    params: Mapping[str, Any] | None = None,
) -> float:
    """Convenience API: resolve mode and evaluate reward in one call."""
    mode = resolve_reward_mode(name, params)
    return mode.evaluate(z_a=z_a, z_b=z_b)


def _register_builtin_modes_fallback() -> None:
    """Register hard-coded built-in modes when YAML config is unavailable."""
    register_reward_mode(
        name="single_a",
        evaluator=_single_a_eval,
        requires_secondary_target=False,
        requires_dual_docking=False,
    )
    register_reward_mode(
        name="single_b",
        evaluator=_single_b_eval,
        requires_secondary_target=True,
        requires_dual_docking=False,
    )
    register_reward_mode(
        name="double",
        evaluator=_double_eval,
        requires_secondary_target=True,
        requires_dual_docking=True,
    )
    register_reward_mode(
        name="linear_equal",
        evaluator=_linear_equal_eval,
        requires_secondary_target=True,
        requires_dual_docking=True,
    )
    register_reward_mode(
        name="biased_a",
        evaluator=_biased_a_eval,
        requires_secondary_target=True,
        requires_dual_docking=True,
        default_params={"weight_b": 0.3},
    )
    register_reward_mode(
        name="selective_a",
        evaluator=_selective_a_eval,
        requires_secondary_target=True,
        requires_dual_docking=True,
    )
    register_reward_mode(
        name="selective_weighted",
        evaluator=_selective_weighted_eval,
        requires_secondary_target=True,
        requires_dual_docking=True,
        default_params={"lambda_off": 1.0},
    )


def _initialize_builtin_registry() -> None:
    """Initialize built-in modes from YAML config with safe fallback."""
    global _REWARD_MODES_CONFIG_SOURCE
    if load_reward_modes_from_yaml() is None:
        _register_builtin_modes_fallback()
        _REWARD_MODES_CONFIG_SOURCE = "<hardcoded-fallback>"


_initialize_builtin_registry()
