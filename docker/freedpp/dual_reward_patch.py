"""Dual-target reward patch for FREED++.

This patch does three things:
1) replaces ligand preparation backend (OpenBabel) with RDKit+Meeko,
2) applies z-score based rewards via shared reward-mode registry,
3) uses UniDock GPU batch docking (``--gpu_batch``) instead of per-ligand CPU calls.

Box parameters for the second receptor are read from ``/workspace/dual_config.json``::

    {"receptor_b": "/workspace/receptor_b.pdbqt", "center_b": "12.3,4.5,6.7", "size_b": "20,20,20"}
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import re
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from multiprocessing import Pool as MPPool
from pathlib import Path
from subprocess import TimeoutExpired, run
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Literal

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

logger = logging.getLogger(__name__)

_MAX_Z_WARNINGS = 5
_z_warn_count = {"a": 0, "b": 0}
_err_warn_count = {"a": 0, "b": 0}
_RESULT_PATTERN = re.compile(r"RESULT[:\s]+(-?\d+(?:\.\d+)?)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# GPU batch docking (Phase 2 optimisation)
# ---------------------------------------------------------------------------
# When enabled, CacheAndPool for DockingScore pre-computes all docking scores
# via a single ``unidock --gpu_batch`` call instead of N×num_sub_proc CPU calls.
_GPU_BATCH_ENABLED = os.environ.get("GPU_BATCH_DOCK", "1") != "0"
_gpu_dock_cache: dict[tuple[str, str], float] = {}  # (receptor_path, smiles) → score
_gpu_batch_log_count = 0

# Ligand complexity gates — molecules exceeding these limits are rejected
# at preparation time because large / highly flexible molecules cause
# UniDock GPU numerical overflow (scores in millions) and batch timeouts.
_MAX_HEAVY_ATOMS = int(os.environ.get("MAX_HEAVY_ATOMS", "70"))
_MAX_ROTATABLE_BONDS = int(os.environ.get("MAX_ROTATABLE_BONDS", "12"))
_CALL_DEBUG_LOG_LIMIT = int(os.environ.get("REWARD_CALL_DEBUG_LOG_LIMIT", "25"))


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
    return stem if stem else receptor


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


def _load_patch_context() -> _PatchContext:
    """Load and validate environment-derived runtime patch context."""
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
        _GPU_BATCH_ENABLED,
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


def _cleanup_gpu_cache_entries(primary_receptor: str, dual_cfg: _DualReceptorConfig | None, smiles: list[str]) -> None:
    """Remove precomputed entries consumed by current CacheAndPool call."""
    for smi in smiles:
        _gpu_dock_cache.pop((primary_receptor, smi), None)
        if dual_cfg is not None:
            _gpu_dock_cache.pop((dual_cfg.receptor_b, smi), None)


def _prepare_pdbqt_worker(args: tuple[str, int, str]) -> bool:
    """Worker for parallel ligand preparation via multiprocessing.Pool.map."""
    smiles, seed, outpath = args
    return prepare_ligand_pdbqt(
        smiles, outpath, seed=seed,
        max_heavy=_MAX_HEAVY_ATOMS, max_rotatable=_MAX_ROTATABLE_BONDS,
    )


def _run_gpu_batch(
    vina_program: str,
    receptor: str,
    center: list[float],
    size: list[float],
    pdbqt_paths: list[str],
    output_dir: str,
    exhaustiveness: int,
    num_modes: int,
    seed: int,
    timeout: int = 600,
) -> bool:
    """Run UniDock --gpu_batch for a list of prepared PDBQT files.

    Returns True on success, False on failure.
    """
    receptor_name = _receptor_label(receptor)
    logger.info(
        "gpu_batch_dock: stage=run start receptor=%s ligands=%d exhaustiveness=%d num_modes=%d seed=%d timeout=%ds",
        receptor_name,
        len(pdbqt_paths),
        exhaustiveness,
        num_modes,
        seed,
        timeout,
    )
    logger.debug(
        "gpu_batch_dock: stage=run box receptor=%s center=(%.3f, %.3f, %.3f) size=(%.3f, %.3f, %.3f)",
        receptor_name,
        center[0],
        center[1],
        center[2],
        size[0],
        size[1],
        size[2],
    )

    cmd = [
        vina_program,
        "--receptor",
        receptor,
        "--gpu_batch",
        *pdbqt_paths,
        "--dir",
        output_dir,
        "--center_x",
        str(center[0]),
        "--center_y",
        str(center[1]),
        "--center_z",
        str(center[2]),
        "--size_x",
        str(size[0]),
        "--size_y",
        str(size[1]),
        "--size_z",
        str(size[2]),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        str(num_modes),
        "--seed",
        str(seed),
    ]
    try:
        result = run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("gpu_batch_dock: unidock rc=%s: %s", result.returncode, (result.stderr or "")[-500:])
            return False
        logger.info("gpu_batch_dock: stage=run done receptor=%s", receptor_name)
        return True
    except TimeoutExpired:
        logger.warning("gpu_batch_dock: stage=run timeout receptor=%s after %ss", receptor_name, timeout)
        return False


def _precompute_gpu_batch_for_receptor(
    unseen_smiles: list[str],
    config: dict[str, Any],
    receptor: str,
    center: Any,
    size: Any,
) -> None:
    """Pre-compute docking scores for *unseen_smiles* against one receptor.

    Results are stored in ``_gpu_dock_cache[(receptor, smi)]``.
    """
    global _gpu_batch_log_count
    error_val = config["error_val"]
    seed = config["seed"]
    receptor_name = _receptor_label(receptor)
    unseen_count = len(unseen_smiles)
    logger.info(
        "gpu_batch_dock: stage=precompute start receptor=%s unseen=%d",
        receptor_name,
        unseen_count,
    )

    if not unseen_smiles:
        logger.debug("gpu_batch_dock: stage=precompute skipped receptor=%s (no unseen ligands)", receptor_name)
        return

    center_f = _coords(center) if isinstance(center, str) else [float(x) for x in center]
    size_f = _coords(size) if isinstance(size, str) else [float(x) for x in size]

    with TemporaryDirectory(prefix="gpu_batch_") as tmpdir:
        lig_dir = Path(tmpdir) / "ligands"
        out_dir = Path(tmpdir) / "output"
        lig_dir.mkdir()
        out_dir.mkdir()

        # 1 conformer per SMILES (n_conf uses identical seed → identical result)
        tasks = []
        for i, smi in enumerate(unseen_smiles):
            fname = f"lig_{i:06d}.pdbqt"
            tasks.append((smi, seed, str(lig_dir / fname)))

        # CPU-parallel ligand preparation (RDKit + Meeko)
        cpu_count = os.cpu_count() or 1
        n_workers = min(len(tasks), cpu_count)
        logger.info(
            "gpu_batch_dock: stage=prepare receptor=%s workers=%d tasks=%d cpu_count=%d",
            receptor_name,
            n_workers,
            len(tasks),
            cpu_count,
        )
        with MPPool(processes=n_workers) as pool:
            prep_ok = pool.map(_prepare_pdbqt_worker, tasks)

        valid_items = [(i, tasks[i]) for i in range(len(tasks)) if prep_ok[i]]
        valid_paths = [t[2] for _, t in valid_items]
        prep_failed = len(tasks) - len(valid_items)
        logger.info(
            "gpu_batch_dock: stage=prepare done receptor=%s prepared=%d failed=%d",
            receptor_name,
            len(valid_items),
            prep_failed,
        )

        # Mark failed preps as error_val
        for i in range(len(tasks)):
            if not prep_ok[i]:
                _gpu_dock_cache[(receptor, tasks[i][0])] = error_val

        if not valid_paths:
            logger.warning(
                "gpu_batch_dock: stage=precompute no_valid_ligands receptor=%s all=%d",
                receptor_name,
                len(tasks),
            )
            return

        ok = _run_gpu_batch(
            config["vina_program"],
            receptor,
            center_f,
            size_f,
            valid_paths,
            str(out_dir),
            config["exhaustiveness"],
            config["num_modes"],
            seed,
        )

        if ok:
            n_cached = 0
            for _, (smi, _, fpath) in valid_items:
                stem = Path(fpath).stem
                out_path = out_dir / f"{stem}_out.pdbqt"
                parsed = _parse_result_from_pdbqt(out_path)
                if parsed is not None:
                    _gpu_dock_cache[(receptor, smi)] = parsed
                    n_cached += 1
                # Not cached: _rdkit_docking falls through to CPU
            n_corrupt = len(valid_items) - n_cached
            if _gpu_batch_log_count < 5:
                _gpu_batch_log_count += 1
                logger.info(
                    "gpu_batch_dock: %d/%d ligands scored (GPU), %d corrupt → CPU fallback, receptor %s",
                    n_cached,
                    len(tasks),
                    n_corrupt,
                    receptor_name,
                )
            if n_corrupt > 0:
                logger.warning(
                    "gpu_batch_dock: %d/%d corrupt scores for %s",
                    n_corrupt,
                    len(valid_items),
                    receptor_name,
                )
        else:
            logger.warning("gpu_batch_dock: batch failed for %s, falling back to per-ligand CPU", receptor_name)


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


def _set_last_docking_timed_out(value: bool) -> None:
    _timeout_state.last_docking_timed_out = value


def _consume_last_docking_timed_out() -> bool:
    value = bool(getattr(_timeout_state, "last_docking_timed_out", False))
    _timeout_state.last_docking_timed_out = False
    return value


def _get_last_docking_timed_out() -> bool:
    return bool(getattr(_timeout_state, "last_docking_timed_out", False))


_METADATA_PATH = Path(os.environ.get("REWARD_METADATA_PATH", "/workspace/reward_metadata.json"))


def _capped_zscore(
    score: float,
    error_val: float,
    mean: float,
    std: float,
    penalty_cap: float,
) -> tuple[float, bool]:
    """Compute z-score, capping at *penalty_cap* when *score* equals *error_val*.

    Returns (z_value, is_invalid).
    """
    if score == error_val:
        return penalty_cap, True
    if std == 0:
        raise ZeroDivisionError("std=0 in normalization.yaml")
    z = (score - mean) / std
    z = max(-_MAX_Z_ABS, min(_MAX_Z_ABS, z))
    return z, False


def _write_reward_metadata() -> None:
    """Write invalid-reward statistics to JSON (registered via atexit)."""
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
        _METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _METADATA_PATH.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("dual_reward_patch: reward metadata written to %s", _METADATA_PATH)
    except OSError as exc:
        logger.warning("dual_reward_patch: failed to write reward metadata: %s", exc)


def _load_normalization(path: str) -> dict[str, Any]:
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
    parts = [float(x) for x in value.split(",")] if isinstance(value, str) else [float(x) for x in value]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 coordinates, got: {value}")
    return parts


_SCORE_MIN, _SCORE_MAX = -20.0, 10.0


def _parse_result_from_pdbqt(path: Path) -> float | None:
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if "RESULT" not in line.upper():
                    continue
                match = _RESULT_PATTERN.search(line)
                if match:
                    score = float(match.group(1))
                    if not (_SCORE_MIN <= score <= _SCORE_MAX):
                        logger.warning(
                            "corrupt docking score %.3f outside [%.0f, %.0f]: %s",
                            score,
                            _SCORE_MIN,
                            _SCORE_MAX,
                            path,
                        )
                        return None
                    logger.debug("dual_reward_patch: stage=parse_result score=%.3f source=%s", score, path)
                    return score
    except OSError as exc:
        logger.debug("dual_reward_patch: stage=parse_result failed_to_read path=%s error=%s", path, exc)
        return None
    logger.debug("dual_reward_patch: stage=parse_result no_result_line path=%s", path)
    return None


def _build_rdkit_docking(docking_vina_cls: Any) -> Callable[..., float]:
    """Build DockingVina.docking replacement using RDKit+Meeko prep."""

    def _rdkit_docking(  # noqa: PLR0913
        smile: str,
        *,
        vina_program: str,
        receptor: str,
        box_center: Any,
        box_size: Any,
        error_val: float,
        seed: int,
        num_modes: int,
        exhaustiveness: int,
        timeout_dock: int,
        timeout_gen3d: int,  # noqa: ARG001
        **kwargs: Any,  # noqa: ARG001
    ) -> float:
        _set_last_docking_timed_out(value=False)
        receptor_name = _receptor_label(receptor)
        short_smile = _short_value(smile)

        cache_key = (receptor, smile)
        if cache_key in _gpu_dock_cache:
            logger.debug(
                "dual_reward_patch: stage=rdkit_docking cache_hit receptor=%s smile=%s",
                receptor_name,
                short_smile,
            )
            return _gpu_dock_cache[cache_key]
        logger.debug(
            "dual_reward_patch: stage=rdkit_docking cache_miss receptor=%s smile=%s",
            receptor_name,
            short_smile,
        )

        center = _coords(box_center)
        size = _coords(box_size)

        with (
            NamedTemporaryFile(mode="w", suffix=".pdbqt") as ligand,
            NamedTemporaryFile(mode="r+t", suffix=".pdbqt") as docking_file,
        ):
            if not prepare_ligand_pdbqt(
                smile, ligand.name, seed=seed,
                max_heavy=_MAX_HEAVY_ATOMS, max_rotatable=_MAX_ROTATABLE_BONDS,
            ):
                logger.warning(
                    "dual_reward_patch: stage=rdkit_docking ligand_prep_failed receptor=%s smile=%s",
                    receptor_name,
                    short_smile,
                )
                return error_val

            cmd = [
                vina_program,
                "--receptor",
                receptor,
                "--ligand",
                ligand.name,
                "--out",
                docking_file.name,
                "--center_x",
                str(center[0]),
                "--center_y",
                str(center[1]),
                "--center_z",
                str(center[2]),
                "--size_x",
                str(size[0]),
                "--size_y",
                str(size[1]),
                "--size_z",
                str(size[2]),
                "--num_modes",
                str(num_modes),
                "--exhaustiveness",
                str(exhaustiveness),
                "--seed",
                str(seed),
            ]

            try:
                result = run(  # noqa: S603
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_dock,
                    check=False,
                )
            except TimeoutExpired:
                _set_last_docking_timed_out(value=True)
                logger.warning(
                    "dual_reward_patch: stage=rdkit_docking timeout receptor=%s timeout=%ss smile=%s",
                    receptor_name,
                    timeout_dock,
                    short_smile,
                )
                return error_val

            if result.returncode != 0:
                tail = (result.stderr or result.stdout or "").strip()[-500:]
                if tail:
                    logger.debug("dual_reward_patch: unidock rc=%s: %s", result.returncode, tail)
                logger.warning(
                    "dual_reward_patch: stage=rdkit_docking failed rc=%d receptor=%s smile=%s",
                    result.returncode,
                    receptor_name,
                    short_smile,
                )
                return error_val

            parsed = docking_vina_cls.parse_output(result.stdout, error_val)
            if parsed != error_val:
                if not (_SCORE_MIN <= parsed <= _SCORE_MAX):
                    logger.warning(
                        "corrupt docking score %.3f from CPU path: %s",
                        parsed,
                        smile,
                    )
                    return error_val
                logger.debug(
                    "dual_reward_patch: stage=rdkit_docking parsed_stdout receptor=%s score=%.3f smile=%s",
                    receptor_name,
                    parsed,
                    short_smile,
                )
                return parsed

            parsed_from_file = _parse_result_from_pdbqt(Path(docking_file.name))
            if parsed_from_file is not None:
                logger.debug(
                    "dual_reward_patch: stage=rdkit_docking parsed_file receptor=%s score=%.3f smile=%s",
                    receptor_name,
                    parsed_from_file,
                    short_smile,
                )
                return parsed_from_file
            logger.warning(
                "dual_reward_patch: stage=rdkit_docking parse_failed receptor=%s smile=%s",
                receptor_name,
                short_smile,
            )
            return error_val

    return _rdkit_docking


def _build_dual_call(
    *,
    original_call: Callable[..., float],
    context: _PatchContext,
) -> Callable[..., float]:
    """Build DockingVina.__call__ replacement for configured reward mode."""

    def _dual_call(self: Any, *args: Any, **kwargs: Any) -> float:
        _call_stats["total"] += 1
        call_idx = _call_stats["total"]
        mode = context.reward_mode
        _log_call_debug(
            "dual_reward_patch: stage=reward_call start idx=%d mode=%s",
            call_idx,
            mode.name,
        )

        if mode.requires_secondary_target and not mode.requires_dual_docking:
            if context.stats_b is None:
                raise RuntimeError(f"stats_b missing for mode '{mode.name}'")
            score_b = original_call(self, *args, **kwargs)
            timed_out_b = _consume_last_docking_timed_out()
            error_val_b = float(self.config.get("error_val", 100.0))
            z_b, invalid_b = _evaluate_target_score(
                label="b",
                score=score_b,
                timed_out=timed_out_b,
                error_val=error_val_b,
                stats=context.stats_b,
            )
            _log_call_debug(
                "dual_reward_patch: stage=reward_call done idx=%d mode=single_b score_b=%.3f z_b=%.3f invalid_b=%s timeout_b=%s",
                call_idx,
                score_b,
                z_b,
                invalid_b,
                timed_out_b,
            )
            return mode.evaluate(z_b=z_b)

        score_a = original_call(self, *args, **kwargs)
        timed_out_a = _consume_last_docking_timed_out()
        error_val_a = float(self.config.get("error_val", 100.0))
        z_a, invalid_a = _evaluate_target_score(
            label="a",
            score=score_a,
            timed_out=timed_out_a,
            error_val=error_val_a,
            stats=context.stats_a,
        )

        if not mode.requires_secondary_target:
            reward_single = mode.evaluate(z_a=z_a)
            _log_call_debug(
                "dual_reward_patch: stage=reward_call done idx=%d mode=%s "
                "score_a=%.3f z_a=%.3f reward=%.3f invalid_a=%s timeout_a=%s",
                call_idx,
                mode.name,
                score_a,
                z_a,
                reward_single,
                invalid_a,
                timed_out_a,
            )
            return reward_single

        if context.dual_cfg is None:
            raise RuntimeError(f"dual_cfg missing for mode '{mode.name}'")
        if context.stats_b is None:
            raise RuntimeError(f"stats_b missing for mode '{mode.name}'")

        score_b = _score_secondary_target(self, original_call, context.dual_cfg, *args, **kwargs)
        timed_out_b = _consume_last_docking_timed_out()
        error_val_b = float(self.config.get("error_val", 100.0))
        z_b, invalid_b = _evaluate_target_score(
            label="b",
            score=score_b,
            timed_out=timed_out_b,
            error_val=error_val_b,
            stats=context.stats_b,
        )

        reward = mode.evaluate(z_a=z_a, z_b=z_b)
        _log_call_debug(
            "dual_reward_patch: stage=reward_call done idx=%d mode=%s "
            "z_a=%.3f z_b=%.3f reward=%.3f invalid_a=%s invalid_b=%s timeout_a=%s timeout_b=%s",
            call_idx,
            mode.name,
            z_a,
            z_b,
            reward,
            invalid_a,
            invalid_b,
            timed_out_a,
            timed_out_b,
        )
        return reward

    return _dual_call


def _patch_cache_and_pool(docking_vina_cls: Any, context: _PatchContext) -> None:
    """Patch CacheAndPool to precompute docking with UniDock GPU batch."""
    if not _GPU_BATCH_ENABLED:
        logger.info("dual_reward_patch: GPU batch docking DISABLED (GPU_BATCH_DOCK=0)")
        return

    try:
        from ffreed.utils import CacheAndPool
    except (ImportError, AttributeError):
        logger.debug("dual_reward_patch: ffreed.utils.CacheAndPool not available, GPU batch disabled")
        CacheAndPool = None  # type: ignore[assignment,misc]  # noqa: N806

    if CacheAndPool is None:
        logger.info("dual_reward_patch: GPU batch docking DISABLED (CacheAndPool unavailable)")
        return

    original_cap_call = CacheAndPool.__call__

    def _gpu_aware_cap_call(self: Any, args: list[str]) -> list[Any]:
        unseen = list(set(args).difference(self.results))
        mode = context.reward_mode
        logger.info(
            "dual_reward_patch: stage=gpu_cache_pool start requested=%d unseen=%d cached=%d",
            len(args),
            len(unseen),
            len(self.results),
        )
        if unseen and hasattr(self.func, "property") and isinstance(self.func.property, docking_vina_cls):
            config = self.func.property.config
            logger.info(
                "dual_reward_patch: stage=gpu_cache_pool precompute primary=%s mode=%s",
                _receptor_label(config["receptor"]),
                mode.name,
            )
            _precompute_gpu_batch_for_receptor(
                unseen,
                config,
                config["receptor"],
                config["box_center"],
                config["box_size"],
            )
            if mode.requires_dual_docking and context.dual_cfg is not None:
                logger.info(
                    "dual_reward_patch: stage=gpu_cache_pool precompute secondary=%s",
                    _receptor_label(context.dual_cfg.receptor_b),
                )
                _precompute_gpu_batch_for_receptor(
                    unseen,
                    config,
                    context.dual_cfg.receptor_b,
                    context.dual_cfg.center_b,
                    context.dual_cfg.size_b,
                )

            values = list(map(self.func, unseen))
            new_results = dict(zip(unseen, values))
            self.results = {**self.results, **new_results}
            logger.info(
                "dual_reward_patch: stage=gpu_cache_pool scored unseen=%d total_cached=%d",
                len(unseen),
                len(self.results),
            )

            _cleanup_gpu_cache_entries(config["receptor"], context.dual_cfg, unseen)
            logger.debug(
                "dual_reward_patch: stage=gpu_cache_pool cleaned_cache_entries smiles=%d",
                len(unseen),
            )
            return [self.results[arg] for arg in args]

        logger.debug("dual_reward_patch: stage=gpu_cache_pool fallback=original_cache_and_pool")
        return original_cap_call(self, args)

    CacheAndPool.__call__ = _gpu_aware_cap_call
    logger.info("dual_reward_patch: GPU batch docking ENABLED (CacheAndPool patched)")


def _patch_docking_vina(docking_vina_cls: Any, context: _PatchContext) -> None:
    """Apply DockingVina monkey patches for docking and reward modes."""
    original_call = docking_vina_cls.__call__

    docking_vina_cls.docking = staticmethod(_build_rdkit_docking(docking_vina_cls))
    logger.info("dual_reward_patch: stage=patching replaced DockingVina.docking backend=rdkit_meeko")

    docking_vina_cls.__call__ = _build_dual_call(original_call=original_call, context=context)
    atexit.register(_write_reward_metadata)
    logger.info("dual_reward_patch: stage=patching replaced DockingVina.__call__ and registered metadata writer")

    _patch_cache_and_pool(docking_vina_cls, context)


def _apply_dual_patch() -> None:
    """Monkey-patch DockingVina for dual-target reward calculation."""
    context = _load_patch_context()

    # ffreed must already be on sys.path (set by run_with_patch.py)
    from ffreed.env.docking import DockingVina

    logger.info("dual_reward_patch: stage=patching hook_target=DockingVina")
    _patch_docking_vina(DockingVina, context)
    logger.info(
        "dual_reward_patch: patched DockingVina"
        " (reward_mode=%s, ligand_backend=rdkit_meeko,"
        " invalid_z_penalty=%.1f, max_z_abs=%.1f)",
        context.reward_mode.name,
        _INVALID_Z_PENALTY,
        _MAX_Z_ABS,
    )


# Auto-apply on import
_apply_dual_patch()
