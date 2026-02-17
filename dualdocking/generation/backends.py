"""Generation backend registry and adapters.

This module decouples pipeline scripts from a hard dependency on FREED++ by
introducing a stable backend interface and config-driven backend resolution.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from dualdocking.reward import reward_mode_requires_secondary_target
from dualdocking.utils.naming import canonicalize_name

GenerationMode = Literal["single_a", "single_b", "double", "linear_equal", "biased_a", "selective_a", "selective_weighted"]


@dataclass(frozen=True)
class GenerationRequest:
    """Backend-agnostic generation request.

    Attributes:
        pair_id: Pair identifier from ``configs/pairs.yaml``.
        pair_config: Pair configuration mapping.
        mode: Generation mode.
        seed: Random seed for the run.
        output_dir: Run output directory.
        n_molecules: Molecule budget for this run.
        epochs_override: Optional epoch override from CLI.
        steps_per_epoch_override: Optional steps-per-epoch override from CLI.
        generator_config: ``generator`` section from ``experiment.yaml``.
        quality_gate: ``quality_gate`` section from ``experiment.yaml``.
    """

    pair_id: str
    pair_config: Mapping[str, Any]
    mode: GenerationMode
    seed: int
    output_dir: Path
    n_molecules: int
    epochs_override: int | None
    steps_per_epoch_override: int | None
    generator_config: Mapping[str, Any]
    quality_gate: Mapping[str, Any]


GenerationBackend = Callable[[GenerationRequest], list[str]]

_BUILTIN_BACKENDS: dict[str, GenerationBackend] = {}


def _as_mapping(value: object) -> Mapping[str, Any]:
    """Return *value* as a mapping or an empty mapping when invalid."""
    if isinstance(value, Mapping):
        return cast("Mapping[str, Any]", value)
    return {}


def _canonical_backend_name(name: str) -> str:
    """Normalize backend names to canonical built-in keys."""
    return canonicalize_name(name, {"freed": "freedpp", "freedpp": "freedpp"})


def _target_id(pair_config: Mapping[str, Any], role: str) -> str:
    """Extract ``target_a.id`` or ``target_b.id`` from pair config."""
    target_cfg = pair_config.get(role)
    if not isinstance(target_cfg, Mapping):
        msg = f"Invalid pair config: missing mapping for {role}"
        raise TypeError(msg)
    target_id = target_cfg.get("id")
    if not isinstance(target_id, str) or not target_id:
        msg = f"Invalid pair config: missing {role}.id"
        raise ValueError(msg)
    return target_id


def _cfg_int(cfg: Mapping[str, Any], key: str, default: int) -> int:
    """Read an integer value from mapping with fallback."""
    try:
        return int(cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def _cfg_float(cfg: Mapping[str, Any], key: str, default: float) -> float:
    """Read a float value from mapping with fallback."""
    try:
        return float(cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def _cfg_bool(cfg: Mapping[str, Any], key: str, default: bool) -> bool:
    """Read a boolean value from mapping with fallback."""
    value = cfg.get(key, default)
    if isinstance(value, bool):
        return value
    return bool(value)


def _cfg_str(cfg: Mapping[str, Any], key: str, default: str) -> str:
    """Read a string value from mapping with fallback."""
    value = cfg.get(key, default)
    if isinstance(value, str) and value:
        return value
    return default


def _cfg_str_maybe_empty(cfg: Mapping[str, Any], key: str, default: str) -> str:
    """Read string value preserving empty/whitespace-stripped content."""
    value = cfg.get(key, default)
    if isinstance(value, str):
        return value.strip()
    return default


def _freedpp_backend(request: GenerationRequest) -> list[str]:
    """Built-in FREED++ generation backend adapter."""
    from dualdocking.generation.freedpp import GenerationConfig, generate

    freedpp_cfg = _as_mapping(request.generator_config.get("freedpp"))
    use_model_cache_default = True
    reward_modes_passthrough_default = True

    second_target_id: str | None = None
    if reward_mode_requires_secondary_target(request.mode):
        second_target_id = _target_id(request.pair_config, "target_b")

    gen_config = GenerationConfig(
        target_id=_target_id(request.pair_config, "target_a"),
        n_molecules=request.n_molecules,
        seed=request.seed,
        output_dir=request.output_dir,
        reward_mode=request.mode,
        second_target_id=second_target_id,
        epochs=request.epochs_override if request.epochs_override is not None else _cfg_int(freedpp_cfg, "epochs", 200),
        num_sub_proc=_cfg_int(freedpp_cfg, "num_sub_proc", 12),
        starting_smile=_cfg_str(freedpp_cfg, "starting_smile", "c1([*:1])c([*:2])ccc([*:3])c1"),
        use_cache=_cfg_bool(freedpp_cfg, "use_model_cache", use_model_cache_default),
        docker_image=_cfg_str(freedpp_cfg, "docker_image", "dualdocking/freedpp:latest"),
        training_timeout=_cfg_int(freedpp_cfg, "training_timeout", 86400),
        box_padding=_cfg_float(freedpp_cfg, "box_padding", 10.0),
        vina_program=_cfg_str(freedpp_cfg, "vina_program", "unidock_gpu"),
        steps_per_epoch=(
            request.steps_per_epoch_override
            if request.steps_per_epoch_override is not None
            else _cfg_int(freedpp_cfg, "steps_per_epoch", 256)
        ),
        qg_max_invalid_fraction=_cfg_float(request.quality_gate, "max_invalid_fraction", 0.10),
        qg_max_timeout_fraction=_cfg_float(request.quality_gate, "max_timeout_fraction", 0.05),
        pass_reward_modes_to_container=_cfg_bool(
            freedpp_cfg,
            "pass_reward_modes_to_container",
            reward_modes_passthrough_default,
        ),
        reward_modes_path_in_container=_cfg_str_maybe_empty(
            freedpp_cfg,
            "reward_modes_path_in_container",
            "/opt/reward_modes.yaml",
        ),
    )
    return generate(gen_config)


def register_generation_backend(name: str, backend: GenerationBackend) -> None:
    """Register a generation backend under *name*."""
    canonical = _canonical_backend_name(name)
    _BUILTIN_BACKENDS[canonical] = backend


def list_generation_backends() -> list[str]:
    """List available built-in generation backends."""
    return sorted(_BUILTIN_BACKENDS)


def _load_external_backend(spec: str) -> GenerationBackend:
    """Load external backend from ``module`` or ``module:callable`` spec."""
    module_path, _, attr = spec.partition(":")
    module = importlib.import_module(module_path)
    target_name = attr or "generate"
    candidate = getattr(module, target_name, None)
    if candidate is None:
        msg = f"Generation backend '{spec}' must expose callable '{target_name}'"
        raise ValueError(msg)
    if not callable(candidate):
        msg = f"Generation backend '{spec}' attribute '{target_name}' is not callable"
        raise TypeError(msg)
    return cast("GenerationBackend", candidate)


def resolve_generation_backend(generator_cfg: Mapping[str, Any] | None) -> tuple[str, GenerationBackend]:
    """Resolve generation backend from ``experiment.yaml`` generator section.

    Resolution order:
    1. ``generator.backend``
    2. ``generator.name`` (legacy)
    3. built-in ``freedpp``

    Non built-in values are treated as external backend specs:
    ``package.module`` (expects ``generate`` callable) or
    ``package.module:callable``.
    """
    cfg = generator_cfg or {}
    raw_spec = cfg.get("backend", cfg.get("name", "freedpp"))
    spec = str(raw_spec).strip() if raw_spec is not None else "freedpp"
    canonical = _canonical_backend_name(spec)

    if canonical in _BUILTIN_BACKENDS:
        return canonical, _BUILTIN_BACKENDS[canonical]
    return spec, _load_external_backend(spec)


register_generation_backend("freedpp", _freedpp_backend)
