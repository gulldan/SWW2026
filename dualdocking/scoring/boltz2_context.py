"""Context/provenance helpers for Boltz-2 scoring cache segmentation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from typing import Protocol


class _LoggerLike(Protocol):
    """Minimal logger protocol for context helpers."""

    def info(self, message: str, *args: object) -> None:
        """Log informational message.

        Args:
            message: Log message template/text.
            *args: Optional format arguments.
        """

    def warning(self, message: str, *args: object) -> None:
        """Log warning message.

        Args:
            message: Log message template/text.
            *args: Optional format arguments.
        """


def run_cmd(cmd: list[str]) -> str:
    """Run command and return stripped output.

    Args:
        cmd: Command with arguments.

    Returns:
        Command output with surrounding whitespace stripped. Returns empty
        string on command failures.
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    except FileNotFoundError:
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout or result.stderr or "").strip()


def docker_digest(
    image: str,
    *,
    master_config_field: Callable[[str], str],
    log: _LoggerLike,
) -> str:
    """Resolve Docker digest/ID with master-config fallback.

    Args:
        image: Docker image reference.
        master_config_field: Callback for reading master config dotted path.
        log: Logger-like instance.

    Returns:
        Docker digest/ID string, or ``"unknown"`` when unavailable.
    """
    out = run_cmd(["docker", "image", "inspect", image, "--format", "{{index .RepoDigests 0}}"])
    if "@" in out:
        return out.split("@")[-1]
    if out:
        return out

    image_id = run_cmd(["docker", "image", "inspect", image, "--format", "{{.Id}}"])
    if image_id:
        return image_id

    try:
        return master_config_field("software.docker_image_digest")
    except Exception:
        log.warning("Cannot determine Docker digest; using 'unknown'")
        return "unknown"


def context_hash(
    *,
    backend: str,
    nim_predict_params: dict[str, int | float | str],
    nim_urls: list[str],
    boltz_version: str,
    cli_args_base: list[str],
    accelerator: str,
    docker_digest: str,
    log: _LoggerLike,
) -> str:
    """Compute cache context hash for current scoring configuration.

    Args:
        backend: Active backend (``"cli"`` or ``"nim"``).
        nim_predict_params: NIM request parameter map.
        nim_urls: NIM URL list (used for logging only).
        boltz_version: Boltz package version.
        cli_args_base: CLI argument vector.
        accelerator: Requested accelerator.
        docker_digest: Docker digest/ID string.
        log: Logger-like instance.

    Returns:
        16-character context hash prefix.
    """
    if backend == "nim":
        nim_params_str = json.dumps(nim_predict_params, sort_keys=True)
        blob = f"nim||{nim_params_str}".encode()
        context_hash_value = hashlib.sha256(blob).hexdigest()[:16]
        log.info(
            f"Scoring context (NIM): urls={nim_urls}, "
            f"params_hash={hashlib.sha256(nim_params_str.encode()).hexdigest()[:8]}... -> ctx={context_hash_value}",
        )
        return context_hash_value

    cli_args = "|".join(cli_args_base)
    blob = f"{boltz_version}||{cli_args}||{accelerator}||{docker_digest}".encode()
    context_hash_value = hashlib.sha256(blob).hexdigest()[:16]
    log.info(
        f"Scoring context: boltz={boltz_version}, "
        f"cli_hash={hashlib.sha256(cli_args.encode()).hexdigest()[:8]}..., "
        f"accel={accelerator}, docker={docker_digest[:16]}... -> ctx={context_hash_value}",
    )
    return context_hash_value


def context_info(
    *,
    backend: str,
    cache_context_hash: str,
    nim_predict_params: dict[str, int | float | str],
    nim_urls: list[str],
    nim_concurrency: int,
    boltz_version: str,
    cli_args_base: list[str],
    accelerator: str,
    docker_digest: str,
) -> dict[str, str | int | list[str]]:
    """Build context metadata payload for manifests.

    Args:
        backend: Active backend (``"cli"`` or ``"nim"``).
        cache_context_hash: Precomputed context hash.
        nim_predict_params: NIM request parameter map.
        nim_urls: NIM URL list.
        nim_concurrency: NIM concurrency level.
        boltz_version: Boltz package version.
        cli_args_base: CLI argument vector.
        accelerator: Requested accelerator.
        docker_digest: Docker digest/ID string.

    Returns:
        Dictionary with backend-specific provenance fields.
    """
    if backend == "nim":
        nim_params_str = json.dumps(nim_predict_params, sort_keys=True)
        return {
            "cache_context_hash": cache_context_hash,
            "backend": "nim",
            "nim_urls": nim_urls,
            "nim_concurrency": nim_concurrency,
            "nim_params_hash": hashlib.sha256(nim_params_str.encode()).hexdigest()[:16],
        }

    cli_args = "|".join(cli_args_base)
    return {
        "cache_context_hash": cache_context_hash,
        "backend": "cli",
        "boltz_version": boltz_version,
        "docker_digest": docker_digest,
        "accelerator": accelerator,
        "cli_args_hash": hashlib.sha256(cli_args.encode()).hexdigest()[:16],
    }
