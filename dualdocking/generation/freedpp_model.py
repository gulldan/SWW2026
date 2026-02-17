"""Model/cache/argument helpers for FREED++ generation."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol


class _LoggerLike(Protocol):
    """Minimal logger protocol for FREED++ model helpers."""

    def debug(self, message: str, *args: object) -> None:
        """Log debug message.

        Args:
            message: Message template or text.
            *args: Optional formatting arguments.
        """

    def info(self, message: str, *args: object) -> None:
        """Log info message.

        Args:
            message: Message template or text.
            *args: Optional formatting arguments.
        """


class _GenerationConfigLike(Protocol):
    """Protocol for config fields used by model helpers."""

    target_id: str
    second_target_id: str | None
    reward_mode: str
    epochs: int
    steps_per_epoch: int
    num_sub_proc: int
    seed: int
    vina_program: str
    starting_smile: str
    n_molecules: int


def auto_num_sub_proc(
    *,
    env_num_sub_proc: str | None,
    detect_vram_gb: Callable[[], float | None],
    log: _LoggerLike,
) -> int:
    """Pick ``num_sub_proc`` based on GPU VRAM with env override.

    Args:
        env_num_sub_proc: Optional override from ``FREEDPP_NUM_SUB_PROC``.
        detect_vram_gb: Callback that returns primary GPU VRAM in GB.
        log: Logger-like instance.

    Returns:
        Number of concurrent docking subprocesses.
    """
    if env_num_sub_proc is not None:
        return int(env_num_sub_proc)

    vram_gb = detect_vram_gb()

    if vram_gb is None:
        log.debug("GPU VRAM detection failed for num_sub_proc, using default=8")
        return 8

    log.debug(f"GPU VRAM detected: {vram_gb:.1f} GB")
    if vram_gb >= 76:
        return 24
    if vram_gb >= 38:
        return 12
    return 8


def cache_key(config: _GenerationConfigLike, *, key_version: str) -> str:
    """Compute cache key from generation settings.

    Args:
        config: Generation configuration object.
        key_version: Manual cache-key schema version token.

    Returns:
        First 16 hex chars of SHA-256 digest.
    """
    parts = [
        key_version,
        config.target_id,
        config.second_target_id or "",
        config.reward_mode,
        str(config.epochs),
        str(config.steps_per_epoch),
        str(config.num_sub_proc),
        str(config.seed),
        config.vina_program,
        config.starting_smile,
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def find_cached_model(key: str, *, cache_dir: Path, log: _LoggerLike) -> Path | None:
    """Find cached trained model directory by cache key.

    Args:
        key: Cache key value.
        cache_dir: Root cache directory.
        log: Logger-like instance.

    Returns:
        Model directory path, or ``None`` if not found.
    """
    model_dir = cache_dir / key
    if model_dir.exists() and any(model_dir.iterdir()):
        log.info(f"Found cached model: {model_dir}")
        return model_dir
    return None


def resolve_vina_path(program: str, *, vina_paths: Mapping[str, str]) -> str:
    """Map ``vina_program`` token to container binary path.

    Args:
        program: Program token (e.g. ``"unidock_gpu"``) or explicit path.
        vina_paths: Mapping of supported tokens to container paths.

    Returns:
        Resolved path to executable inside container.
    """
    if program in vina_paths:
        return vina_paths[program]
    return program


def build_freedpp_args(
    config: _GenerationConfigLike,
    pdbqt_a: Path,
    center_a: str,
    size_a: str,
    *,
    commands: str,
    auto_num_sub_proc: Callable[[], int],
    resolve_vina_path: Callable[[str], str],
    log: _LoggerLike,
) -> list[str]:
    """Build CLI argument list for FREED++ ``main.py``.

    Args:
        config: Generation configuration.
        pdbqt_a: Path to primary receptor PDBQT.
        center_a: Binding box center for primary receptor.
        size_a: Binding box size for primary receptor.
        commands: FREED++ command set (``"train,sample"`` or ``"sample"``).
        auto_num_sub_proc: Callback for auto process-count selection.
        resolve_vina_path: Callback mapping ``vina_program`` to binary path.
        log: Logger-like instance.

    Returns:
        Argument vector for the FREED++ container command.
    """
    num_sub_proc = config.num_sub_proc if config.num_sub_proc > 0 else auto_num_sub_proc()
    log.debug(
        f"FREED++ num_sub_proc={num_sub_proc} (config={config.num_sub_proc}, auto={config.num_sub_proc == 0})",
    )

    return [
        "--commands",
        commands,
        "--receptor",
        f"/workspace/{pdbqt_a.name}",
        "--box_center",
        center_a,
        "--box_size",
        size_a,
        "--fragments",
        "/opt/freed/data/zinc_crem.json",
        "--alert_collections",
        "/opt/freed/data/alert_collections.csv",
        "--vina_program",
        resolve_vina_path(config.vina_program),
        "--starting_smile",
        config.starting_smile,
        "--exp_root",
        "/workspace/output",
        "--name",
        "run",
        "--epochs",
        str(config.epochs),
        "--num_sub_proc",
        str(num_sub_proc),
        "--seed",
        str(config.seed),
        "--num_mols",
        str(config.n_molecules),
        "--steps_per_epoch",
        str(config.steps_per_epoch),
        "--reward_version",
        "soft",
        "--num_modes",
        "1",
    ]
