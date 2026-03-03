"""State and reward-evaluation helpers for the FREED++ dual reward patch."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dual_reward_patch_deps import ResolvedRewardMode, get_reward_modes_config_source, resolve_reward_mode

logger = logging.getLogger(__name__)


def _configure_patch_logging() -> None:
    """Configure logger level/handlers for container runtime visibility.

    Uses ``DUAL_REWARD_PATCH_LOG_LEVEL`` (default: ``INFO``). If the root logger
    has no handlers, installs a basic stream handler so ``logger.info`` messages
    are visible in container stdout and captured run logs.
    """
    level_name = os.environ.get("DUAL_REWARD_PATCH_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        )
    logger.setLevel(level)


_configure_patch_logging()

_MAX_Z_WARNINGS = 5
_z_warn_count = {"a": 0, "b": 0}
_err_warn_count = {"a": 0, "b": 0}
_RESULT_PATTERN = re.compile(r"RESULT[:\s]+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE)
_CORRUPT_SCORE_LOG_LIMIT = int(os.environ.get("CORRUPT_SCORE_LOG_LIMIT", "20"))
_corrupt_score_warn_count = 0

# Ligand complexity gates — molecules exceeding these limits are rejected
# at preparation time because large / highly flexible molecules cause
# UniDock GPU numerical overflow (scores in millions) and batch timeouts.
_MAX_HEAVY_ATOMS = int(os.environ.get("MAX_HEAVY_ATOMS", "70"))
_MAX_ROTATABLE_BONDS = int(os.environ.get("MAX_ROTATABLE_BONDS", "12"))
_CALL_DEBUG_LOG_LIMIT = int(os.environ.get("REWARD_CALL_DEBUG_LOG_LIMIT", "25"))

# ---------------------------------------------------------------------------
# Invalid-docking exclusion (ENG-01 / KVA-255)
# ---------------------------------------------------------------------------

_INVALID_Z_PENALTY = float(os.environ.get("INVALID_Z_PENALTY", "4.0"))
if not (0 < _INVALID_Z_PENALTY < float("inf")):
    raise ValueError(f"INVALID_Z_PENALTY must be a positive finite number, got {_INVALID_Z_PENALTY}")

_MAX_Z_ABS = float(os.environ.get("MAX_Z_ABS", "10.0"))
if not (0 < _MAX_Z_ABS < float("inf")):
    raise ValueError(f"MAX_Z_ABS must be a positive finite number, got {_MAX_Z_ABS}")

_call_stats: dict[str, int] = {
    "total": 0,
    "invalid_a": 0,
    "invalid_b": 0,
    "timeout_a": 0,
    "timeout_b": 0,
}

# Thread-local side-channel: _rdkit_docking sets timeout status for the current
# thread, _dual_call consumes and resets it after each _original_call.
_timeout_state = threading.local()

_METADATA_PATH = Path(os.environ.get("REWARD_METADATA_PATH", "/workspace/reward_metadata.json"))

_SCORE_MIN, _SCORE_MAX = -20.0, 10.0


@dataclass(frozen=True)
class _TargetStats:
    """Normalization parameters for a single target."""

    mean: float
    std: float


@dataclass(frozen=True)
class _DualReceptorConfig:
    """Secondary receptor settings used in dual-target modes."""

    receptor_b: str
    center_b: Any
    size_b: Any


@dataclass(frozen=True)
class _PatchContext:
    """Immutable runtime context for reward patch callbacks."""

    reward_mode: ResolvedRewardMode
    normalization_path: str
    target_id_a: str
    target_id_b: str
    stats_a: _TargetStats
    stats_b: _TargetStats | None
    dual_cfg: _DualReceptorConfig | None


def _load_reward_params_from_env() -> dict[str, float]:
    """Load reward-mode numeric parameters from environment variables."""
    params: dict[str, float] = {}

    raw_json = os.environ.get("REWARD_PARAMS_JSON", "").strip()
    if raw_json:
        parsed = json.loads(raw_json)
        if not isinstance(parsed, dict):
            raise TypeError("REWARD_PARAMS_JSON must be a JSON object")
        params = {str(key): float(value) for key, value in parsed.items()}

    lambda_off = os.environ.get("REWARD_LAMBDA_OFF")
    if lambda_off is not None and "lambda_off" not in params:
        params["lambda_off"] = float(lambda_off)

    weight_b = os.environ.get("REWARD_WEIGHT_B")
    if weight_b is not None and "weight_b" not in params:
        params["weight_b"] = float(weight_b)

    return params


def _receptor_label(receptor: str) -> str:
    """Return a short receptor label for logs."""
    stem = Path(receptor).stem
    return stem or receptor


def _short_value(value: str, max_len: int = 80) -> str:
    """Return value truncated for compact logs."""
    if len(value) <= max_len:
        return value
    return f"{value[: max_len - 3]}..."


def _log_call_debug(message: str, *values: Any) -> None:
    """Emit bounded debug logs for per-molecule reward calls."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    if _call_stats["total"] > _CALL_DEBUG_LOG_LIMIT:
        return
    logger.debug(message, *values)


def _warn_large_z(label: Literal["a", "b"], z_value: float) -> None:
    """Warn on extreme z-scores, rate-limited per target label."""
    if abs(z_value) <= 10:
        return
    if _z_warn_count[label] >= _MAX_Z_WARNINGS:
        return
    _z_warn_count[label] += 1
    logger.warning("dual_reward_patch: |z_%s| > 10 (%.2f)", label, z_value)


def _warn_error_value(label: Literal["a", "b"], value: float, error_val: float) -> None:
    """Warn when docking returned the configured error sentinel."""
    if value != error_val:
        return
    if _err_warn_count[label] >= _MAX_Z_WARNINGS:
        return
    _err_warn_count[label] += 1
    logger.warning("dual_reward_patch: docking returned error_val for target %s (%s)", label, value)


def _load_target_stats(targets: dict[str, Any], target_id: str, label: Literal["A", "B"]) -> _TargetStats:
    """Load mean/std for one target from normalization config."""
    if target_id not in targets:
        raise KeyError(f"Target {label} '{target_id}' not found in normalization.yaml")
    return _TargetStats(mean=float(targets[target_id]["mean"]), std=float(targets[target_id]["std"]))


def _load_dual_receptor_config(mode_name: str) -> _DualReceptorConfig:
    """Load secondary receptor config from ``DUAL_CONFIG``."""
    dual_receptor_b = os.environ.get("DUAL_RECEPTOR_B")
    if not dual_receptor_b:
        raise RuntimeError(f"DUAL_RECEPTOR_B is required for mode '{mode_name}'")

    config_path = Path(os.environ.get("DUAL_CONFIG", "/workspace/dual_config.json"))
    logger.info("dual_reward_patch: stage=dual_config loading path=%s", config_path)
    with config_path.open() as f:
        raw_cfg = json.load(f)

    dual_cfg = _DualReceptorConfig(
        receptor_b=str(raw_cfg["receptor_b"]),
        center_b=raw_cfg["center_b"],
        size_b=raw_cfg["size_b"],
    )
    logger.info(
        "dual_reward_patch: stage=dual_config loaded receptor_b=%s center_b=%s size_b=%s",
        dual_cfg.receptor_b,
        dual_cfg.center_b,
        dual_cfg.size_b,
    )
    return dual_cfg


def _load_patch_context(*, gpu_batch_enabled: bool) -> _PatchContext:
    """Load and validate environment-derived runtime patch context.

    Args:
        gpu_batch_enabled: Whether GPU batch docking precompute is active.

    Returns:
        Immutable context for reward callbacks and docking hooks.
    """
    reward_mode_name = os.environ.get("REWARD_MODE", "single_a")
    reward_params = _load_reward_params_from_env()
    reward_mode = resolve_reward_mode(reward_mode_name, reward_params)
    normalization_path = os.environ.get("NORMALIZATION_PATH", "/workspace/normalization.yaml")
    target_id_a = os.environ.get("TARGET_ID_A") or ""
    target_id_b = os.environ.get("TARGET_ID_B") or ""

    logger.info(
        "dual_reward_patch: stage=init start reward_mode=%s target_a=%s target_b=%s normalization=%s gpu_batch=%s",
        reward_mode.name,
        target_id_a or "<unset>",
        target_id_b or "<unset>",
        normalization_path,
        gpu_batch_enabled,
    )
    logger.info(
        "dual_reward_patch: stage=init limits max_heavy=%d max_rotatable=%d invalid_z_penalty=%.2f max_z_abs=%.2f",
        _MAX_HEAVY_ATOMS,
        _MAX_ROTATABLE_BONDS,
        _INVALID_Z_PENALTY,
        _MAX_Z_ABS,
    )

    normalization = _load_normalization(normalization_path)
    targets = normalization.get("targets", {})
    logger.info(
        "dual_reward_patch: stage=normalization loaded targets=%d path=%s",
        len(targets) if isinstance(targets, dict) else -1,
        normalization_path,
    )
    logger.info(
        "dual_reward_patch: stage=reward_mode resolved mode=%s params=%s requires_secondary=%s requires_dual=%s",
        reward_mode.name,
        dict(reward_mode.params),
        reward_mode.requires_secondary_target,
        reward_mode.requires_dual_docking,
    )
    logger.info(
        "dual_reward_patch: stage=reward_mode source=%s",
        get_reward_modes_config_source() or "<unknown>",
    )

    stats_a = _load_target_stats(targets, target_id_a, "A")
    logger.info(
        "dual_reward_patch: stage=normalization target_a=%s mean=%.4f std=%.4f",
        target_id_a,
        stats_a.mean,
        stats_a.std,
    )

    stats_b: _TargetStats | None = None
    if reward_mode.requires_secondary_target:
        stats_b = _load_target_stats(targets, target_id_b, "B")
        logger.info(
            "dual_reward_patch: stage=normalization target_b=%s mean=%.4f std=%.4f",
            target_id_b,
            stats_b.mean,
            stats_b.std,
        )

    dual_cfg = _load_dual_receptor_config(reward_mode.name) if reward_mode.requires_dual_docking else None
    return _PatchContext(
        reward_mode=reward_mode,
        normalization_path=normalization_path,
        target_id_a=target_id_a,
        target_id_b=target_id_b,
        stats_a=stats_a,
        stats_b=stats_b,
        dual_cfg=dual_cfg,
    )


def _evaluate_target_score(
    *,
    label: Literal["a", "b"],
    score: float,
    timed_out: bool,
    error_val: float,
    stats: _TargetStats,
) -> tuple[float, bool]:
    """Compute z-score and update invalid/timeout counters for one target."""
    z_value, invalid = _capped_zscore(score, error_val, stats.mean, stats.std, _INVALID_Z_PENALTY)
    if invalid:
        _call_stats[f"invalid_{label}"] += 1
        if timed_out:
            _call_stats[f"timeout_{label}"] += 1
        _warn_error_value(label, score, error_val)
    else:
        _warn_large_z(label, z_value)
    return z_value, invalid


@contextlib.contextmanager
def _temporary_receptor_config(config: dict[str, Any], dual_cfg: _DualReceptorConfig) -> Iterator[None]:
    """Temporarily replace receptor and box parameters with target-B values."""
    original_receptor = config["receptor"]
    original_center = config["box_center"]
    original_size = config["box_size"]

    try:
        config["receptor"] = dual_cfg.receptor_b
        config["box_center"] = _coords(dual_cfg.center_b)
        config["box_size"] = _coords(dual_cfg.size_b)
        yield
    finally:
        config["receptor"] = original_receptor
        config["box_center"] = original_center
        config["box_size"] = original_size


def _score_secondary_target(
    dock_obj: Any,
    original_call: Callable[..., float],
    dual_cfg: _DualReceptorConfig,
    *args: Any,
    **kwargs: Any,
) -> float:
    """Score molecule on secondary receptor by temporary config swap."""
    with _temporary_receptor_config(dock_obj.config, dual_cfg):
        return original_call(dock_obj, *args, **kwargs)


def _set_last_docking_timed_out(value: bool) -> None:
    """Set timeout side-channel for current thread."""
    _timeout_state.last_docking_timed_out = value


def _consume_last_docking_timed_out() -> bool:
    """Consume and reset timeout side-channel for current thread."""
    value = bool(getattr(_timeout_state, "last_docking_timed_out", False))
    _timeout_state.last_docking_timed_out = False
    return value


def _get_last_docking_timed_out() -> bool:
    """Read timeout side-channel without reset."""
    return bool(getattr(_timeout_state, "last_docking_timed_out", False))


def _capped_zscore(
    score: float,
    error_val: float,
    mean: float,
    std: float,
    penalty_cap: float,
) -> tuple[float, bool]:
    """Compute z-score, capping at *penalty_cap* when *score* equals *error_val*.

    Args:
        score: Docking score value.
        error_val: Error sentinel used by docking backend.
        mean: Target normalization mean.
        std: Target normalization std.
        penalty_cap: Replacement z-score for invalid scores.

    Returns:
        Tuple of ``(z_value, is_invalid)``.
    """
    if score == error_val:
        return penalty_cap, True
    if std == 0:
        raise ZeroDivisionError("std=0 in normalization.yaml")
    z = (score - mean) / std
    z = max(-_MAX_Z_ABS, min(_MAX_Z_ABS, z))
    return z, False


def _write_reward_metadata(*, path: Path | None = None) -> None:
    """Write invalid-reward statistics to JSON.

    Args:
        path: Explicit output path.  When ``None`` (default / production),
              uses the module-level ``_METADATA_PATH`` derived from the
              ``REWARD_METADATA_PATH`` environment variable.
    """
    metadata_path = path if path is not None else _METADATA_PATH

    total = _call_stats["total"]
    if total == 0:
        logger.debug("dual_reward_patch: stage=metadata skipped reason=no_calls")
        return

    reward_mode_name = os.environ.get("REWARD_MODE", "single_a")
    target_id_a = os.environ.get("TARGET_ID_A", "")
    target_id_b = os.environ.get("TARGET_ID_B", "")
    reward_params: dict[str, float] = {}

    try:
        reward_params = _load_reward_params_from_env()
        resolved_mode = resolve_reward_mode(reward_mode_name, reward_params)
    except Exception as exc:
        logger.warning("dual_reward_patch: stage=metadata reward-mode resolve failed: %s", exc)
        resolved_mode = None

    if resolved_mode is None:
        primary_target = target_id_b if reward_mode_name == "single_b" else target_id_a
    elif resolved_mode.requires_dual_docking:
        primary_target = "both"
    elif resolved_mode.requires_secondary_target:
        primary_target = target_id_b
    else:
        primary_target = target_id_a

    metadata = {
        "total_calls": total,
        "invalid_count_a": _call_stats["invalid_a"],
        "invalid_count_b": _call_stats["invalid_b"],
        "invalid_reward_fraction_a": _call_stats["invalid_a"] / total,
        "invalid_reward_fraction_b": _call_stats["invalid_b"] / total,
        "timeout_count_a": _call_stats["timeout_a"],
        "timeout_count_b": _call_stats["timeout_b"],
        "timeout_fraction_a": _call_stats["timeout_a"] / total,
        "timeout_fraction_b": _call_stats["timeout_b"] / total,
        "invalid_z_penalty": _INVALID_Z_PENALTY,
        "max_z_abs": _MAX_Z_ABS,
        "primary_target": primary_target,
        "reward_mode": reward_mode_name,
        "reward_params": reward_params,
    }
    max_inv = float(os.environ.get("QG_MAX_INVALID_FRACTION", "0.10"))
    max_to = float(os.environ.get("QG_MAX_TIMEOUT_FRACTION", "0.05"))
    inv_ok = all(metadata[f"invalid_reward_fraction_{label}"] <= max_inv for label in ("a", "b"))
    to_ok = all(metadata[f"timeout_fraction_{label}"] <= max_to for label in ("a", "b"))
    metadata["run_valid"] = inv_ok and to_ok

    logger.info(
        "dual_reward_patch: stage=metadata summary total=%d "
        "invalid_a=%.4f invalid_b=%.4f timeout_a=%.4f timeout_b=%.4f run_valid=%s",
        total,
        metadata["invalid_reward_fraction_a"],
        metadata["invalid_reward_fraction_b"],
        metadata["timeout_fraction_a"],
        metadata["timeout_fraction_b"],
        metadata["run_valid"],
    )

    try:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("dual_reward_patch: reward metadata written to %s", metadata_path)
    except OSError as exc:
        logger.warning("dual_reward_patch: failed to write reward metadata: %s", exc)


def _load_normalization(path: str) -> dict[str, Any]:
    """Load normalization YAML as dictionary."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("PyYAML required to load normalization.yaml") from exc

    with Path(path).open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError("normalization.yaml must be a dictionary")
    return data


def _coords(value: Any) -> list[float]:
    """Parse 3D coordinate triplet from CSV string or iterable."""
    parts = [float(x) for x in value.split(",")] if isinstance(value, str) else [float(x) for x in value]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 coordinates, got: {value}")
    return parts


def _parse_result_from_pdbqt(path: Path) -> float | None:
    """Parse UniDock ``RESULT`` score from a PDBQT output file."""
    global _corrupt_score_warn_count
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if "RESULT" not in line.upper():
                    continue
                match = _RESULT_PATTERN.search(line)
                if match:
                    score = float(match.group(1))
                    if not (_SCORE_MIN <= score <= _SCORE_MAX):
                        if _corrupt_score_warn_count < _CORRUPT_SCORE_LOG_LIMIT:
                            logger.warning(
                                "corrupt docking score %.3f outside [%.0f, %.0f]: %s",
                                score,
                                _SCORE_MIN,
                                _SCORE_MAX,
                                path,
                            )
                        elif _corrupt_score_warn_count == _CORRUPT_SCORE_LOG_LIMIT:
                            logger.warning(
                                "corrupt docking score warnings suppressed after %d entries",
                                _CORRUPT_SCORE_LOG_LIMIT,
                            )
                        _corrupt_score_warn_count += 1
                        return None
                    logger.debug("dual_reward_patch: stage=parse_result score=%.3f source=%s", score, path)
                    return score
    except OSError as exc:
        logger.debug("dual_reward_patch: stage=parse_result failed_to_read path=%s error=%s", path, exc)
        return None
    logger.debug("dual_reward_patch: stage=parse_result no_result_line path=%s", path)
    return None


__all__ = [
    "_CALL_DEBUG_LOG_LIMIT",
    "_INVALID_Z_PENALTY",
    "_MAX_HEAVY_ATOMS",
    "_MAX_ROTATABLE_BONDS",
    "_MAX_Z_ABS",
    "_METADATA_PATH",
    "_DualReceptorConfig",
    "_PatchContext",
    "_TargetStats",
    "_call_stats",
    "_capped_zscore",
    "_consume_last_docking_timed_out",
    "_coords",
    "_evaluate_target_score",
    "_get_last_docking_timed_out",
    "_load_patch_context",
    "_log_call_debug",
    "_parse_result_from_pdbqt",
    "_receptor_label",
    "_score_secondary_target",
    "_set_last_docking_timed_out",
    "_short_value",
    "_write_reward_metadata",
    "logger",
]
