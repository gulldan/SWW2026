"""Core pipeline primitives for command execution and logging.

This module provides low-level utilities used by all pipeline stages:
logging, subprocess execution, and environment management.
"""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from dualdocking.utils.nim import nim_url_ready

if TYPE_CHECKING:
    from dualdocking.pipeline.config import ScorerConfig


def utc_timestamp() -> str:
    """Return current UTC time as ISO-8601 string with ``Z`` suffix.

    Returns:
        Timestamp string like ``"2026-02-14T12:34:56Z"``.
    """
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def log_to_file(log_path: Path, message: str) -> None:
    """Append a message line to a log file.

    Creates parent directories if needed.

    Args:
        log_path: Path to the log file.
        message: Line to append (newline added automatically).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def detect_gpu_count() -> int:
    """Detect the number of available CUDA GPUs.

    Checks in order:
    1. ``CUDA_VISIBLE_DEVICES`` environment variable
    2. ``nvidia-smi`` output
    3. ``torch.cuda.device_count()``

    Returns:
        Number of detected CUDA devices, or 1 if detection fails.
    """
    # 1. Honor CUDA_VISIBLE_DEVICES if explicitly set
    devices = _parse_cuda_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if devices is not None:
        if devices:
            logger.debug("GPU count via CUDA_VISIBLE_DEVICES: {}", len(devices))
            return len(devices)
        # CUDA_VISIBLE_DEVICES masks all devices; keep pipeline conservative.
        logger.debug("CUDA_VISIBLE_DEVICES masks all GPUs, assuming 1 GPU")
        return 1

    # 2. nvidia-smi (shows physical GPUs, unaffected by env vars)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
            if lines:
                logger.debug("GPU count via nvidia-smi: {}", len(lines))
                return len(lines)
    except Exception:  # noqa: S110
        pass

    # 3. Fallback: torch
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            if n >= 1:
                logger.debug("GPU count via torch: {}", n)
                return n
    except Exception:  # noqa: S110
        pass

    logger.debug("GPU detection failed, assuming 1 GPU")
    return 1


def _parse_cuda_visible_devices(raw: str | None) -> list[str] | None:
    """Parse ``CUDA_VISIBLE_DEVICES`` into ordered device tokens.

    Args:
        raw: Raw environment value, or ``None`` when unset.

    Returns:
        Parsed list of device tokens when variable is set, empty list when it
        masks all GPUs, or ``None`` when unset/``all`` (meaning unrestricted).
    """
    if raw is None:
        return None

    cvd_value = raw.strip()
    if not cvd_value:
        return []
    cvd_value_l = cvd_value.lower()
    if cvd_value_l == "all":
        return None
    if cvd_value == "-1" or cvd_value_l in {"none", "void"}:
        return []
    return [item.strip() for item in cvd_value.split(",") if item.strip()]


def _resolve_gpu_token(gpu_id: int, env: dict[str, str]) -> str:
    """Resolve logical GPU slot to the actual CUDA device token.

    If ``CUDA_VISIBLE_DEVICES`` is set to a subset (for example ``"1,3"``),
    slot ``0`` maps to ``"1"``, slot ``1`` maps to ``"3"``.

    Args:
        gpu_id: Logical GPU slot from scheduler.
        env: Environment used for subprocess launch.

    Returns:
        Device token suitable for Docker ``--gpus device=...`` and CUDA env vars.
    """
    visible = _parse_cuda_visible_devices(env.get("CUDA_VISIBLE_DEVICES"))
    if visible is None:
        return str(gpu_id)
    if 0 <= gpu_id < len(visible):
        return visible[gpu_id]

    logger.warning(
        "GPU slot {} is out of range for CUDA_VISIBLE_DEVICES={!r}; using slot index directly",
        gpu_id,
        env.get("CUDA_VISIBLE_DEVICES", ""),
    )
    return str(gpu_id)


def build_env_for_task(
    scorer_config: ScorerConfig,
    gpu_id: int | None = None,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build environment variables for a pipeline task.

    Sets GPU device IDs and scorer-specific configuration.

    Args:
        scorer_config: Scorer backend configuration.
        gpu_id: CUDA device index, or ``None`` for default.
        base_env: Starting environment (defaults to ``os.environ``).

    Returns:
        Complete environment dict suitable for ``subprocess.run(env=...)``.
    """
    env = dict(base_env or os.environ)

    if gpu_id is not None:
        gpu_token = _resolve_gpu_token(gpu_id, env)
        env["FREEDPP_CUDA_DEVICE"] = gpu_token
        env["SCORER_CUDA_DEVICE"] = gpu_token

    env["DUALDOCKING_SCORER_ENGINE"] = scorer_config.engine

    if scorer_config.engine == "boltz2":
        if gpu_id is not None:
            env["BOLTZ_CUDA_DEVICE"] = env["SCORER_CUDA_DEVICE"]
        env["BOLTZ_BACKEND"] = scorer_config.backend
        if scorer_config.backend == "nim":
            # Pass ALL NIM URLs for round-robin across GPU instances
            nim_urls = scorer_config.nim_urls
            if nim_urls:
                env["BOLTZ_NIM_URLS"] = ",".join(nim_urls)
                env["BOLTZ_NIM_URL"] = nim_urls[0]  # Fallback for single-URL code
        else:
            env.pop("BOLTZ_NIM_URLS", None)
            env.pop("BOLTZ_NIM_URL", None)
    else:
        # Clear Boltz-specific vars for non-Boltz scorers
        env.pop("BOLTZ_BACKEND", None)
        env.pop("BOLTZ_NIM_URLS", None)
        env.pop("BOLTZ_NIM_URL", None)
        env.pop("BOLTZ_CUDA_DEVICE", None)

    return env


def run_command(
    cmd: list[str],
    log_path: Path,
    project_root: Path,
    env: dict[str, str] | None = None,
    *,
    allow_inprocess: bool = True,
) -> None:
    """Execute a subprocess command with logging.

    Logs the command and output to a file, raises on non-zero exit.

    Args:
        cmd: Command tokens to execute.
        log_path: Path to the log file.
        project_root: Project root directory (for in-process dispatch).
        env: Environment variables (defaults to current ``os.environ``).
        allow_inprocess: Whether to allow in-process dispatch (must be
            ``False`` for concurrent execution due to non-thread-safe globals).

    Raises:
        RuntimeError: If the subprocess exits with a non-zero return code.
    """
    from dualdocking.command_runner import run_project_command

    gpu_tag = ""
    if env and "SCORER_CUDA_DEVICE" in env:
        gpu_tag = f" [GPU {env['SCORER_CUDA_DEVICE']}]"

    log_to_file(log_path, f"[{utc_timestamp()}]{gpu_tag} RUN: {' '.join(cmd)}")
    logger.info("Running{}: {}", gpu_tag, " ".join(cmd))

    result = run_project_command(
        cmd,
        project_root=project_root,
        env=env or dict(os.environ),
        stdout_pipe=True,
        stderr_to_parent=True,
        allow_inprocess=allow_inprocess,
    )

    if result.returncode != 0:
        error_msg = (result.stdout or "")[-4000:]
        log_to_file(log_path, f"[{utc_timestamp()}] ERROR (rc={result.returncode}):\n{error_msg}")
        raise RuntimeError(f"Command failed (rc={result.returncode}): {' '.join(cmd)}")

    if result.stdout:
        log_to_file(log_path, f"[{utc_timestamp()}] OUT: {result.stdout[-2000:]}")


def check_nim_health(urls: list[str], timeout: float = 5.0) -> list[str]:
    """Probe NIM URLs and return only healthy instances.

    Args:
        urls: List of NIM base URLs to check.
        timeout: HTTP request timeout in seconds.

    Returns:
        Subset of *urls* that responded as healthy.
    """
    healthy: list[str] = []
    for url in urls:
        if nim_url_ready(url, timeout=timeout):
            healthy.append(url)
            logger.info("NIM health OK: {}", url)
        else:
            logger.warning("NIM not reachable or not ready: {}", url)
    return healthy
