"""Subprocess and retry helpers for Boltz-2 CLI scoring."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol


class _LoggerLike(Protocol):
    """Minimal logger protocol for subprocess helpers."""

    def debug(self, message: str, *args: object) -> None:
        """Log debug message.

        Args:
            message: Message template/text.
            *args: Format arguments.
        """

    def info(self, message: str, *args: object) -> None:
        """Log info message.

        Args:
            message: Message template/text.
            *args: Format arguments.
        """


class _WriteYamlFn(Protocol):
    """Protocol for YAML writer callback."""

    def __call__(
        self,
        sequence: str,
        smiles: str,
        path: Path,
        *,
        msa_path: Path | None,
    ) -> None:
        """Write Boltz input YAML.

        Args:
            sequence: Target protein sequence.
            smiles: Ligand SMILES.
            path: Destination YAML path.
            msa_path: Optional precomputed MSA CSV path.
        """


class _RunPredictFn(Protocol):
    """Protocol for single-input ``boltz predict`` callback."""

    def __call__(
        self,
        yaml_path: Path,
        out_dir: Path,
        *,
        device: str | None,
    ) -> subprocess.CompletedProcess[str]:
        """Run single-input Boltz prediction command.

        Args:
            yaml_path: Input YAML path.
            out_dir: Output directory.
            device: Optional CUDA device token.

        Returns:
            Completed subprocess result.
        """


class _ParseAffinityFn(Protocol):
    """Protocol for affinity parser callback."""

    def __call__(self, out_dir: Path, stem: str) -> float:
        """Parse affinity value for a single stem.

        Args:
            out_dir: Boltz output root directory.
            stem: Input stem.

        Returns:
            Parsed affinity value.
        """


def build_boltz_env(
    device: str | None,
    *,
    fallback_device: str | None,
    log: _LoggerLike,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build environment for Boltz subprocess invocation.

    Args:
        device: Explicit CUDA device token.
        fallback_device: Fallback device token from environment.
        log: Logger-like instance.
        base_env: Base environment mapping. Defaults to ``os.environ``.

    Returns:
        Environment dictionary for subprocess call.
    """
    env = dict(os.environ if base_env is None else base_env)
    cuda_device = device if device is not None else fallback_device
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_device
        log.debug(f"Boltz GPU pinned to CUDA_VISIBLE_DEVICES={cuda_device}")
    return env


def run_boltz_predict(
    yaml_path: Path,
    out_dir: Path,
    *,
    project_root: Path,
    accelerator: str,
    cli_args_base: Sequence[str],
    timeout: int,
    env: dict[str, str],
    log: _LoggerLike,
) -> subprocess.CompletedProcess[str]:
    """Run ``boltz predict`` for a single YAML input.

    Args:
        yaml_path: Input YAML path.
        out_dir: Output directory.
        project_root: Working directory for subprocess.
        accelerator: Boltz accelerator token (``gpu`` or ``cpu``).
        cli_args_base: Shared Boltz CLI arguments.
        timeout: Timeout in seconds.
        env: Subprocess environment dictionary.
        log: Logger-like instance.

    Returns:
        Completed subprocess result.
    """
    cmd = [
        "uv",
        "run",
        "boltz",
        "predict",
        str(yaml_path),
        "--out_dir",
        str(out_dir),
        "--accelerator",
        accelerator,
        *cli_args_base,
    ]
    log.debug(f"Boltz CMD: {' '.join(cmd)}")

    return subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(project_root),
        env=env,
        check=False,
    )


def _log_boltz_stderr(stderr: str, *, log: _LoggerLike) -> None:
    """Log relevant Boltz stderr lines and drop warning noise.

    Args:
        stderr: Raw subprocess stderr output.
        log: Logger-like instance.
    """
    noise_patterns = ("DeprecationWarning", "SyntaxWarning", "FutureWarning", "UserWarning", "warnings.warn")
    keep_keywords = ("error", "running", "predict", "loaded", "sampling", "progress")

    for line in stderr.splitlines():
        if any(pattern in line for pattern in noise_patterns):
            continue
        lower_line = line.lower()
        if any(keyword in lower_line for keyword in keep_keywords):
            log.info(f"[Boltz-2] {line.rstrip()}")


def run_boltz_predict_dir(
    input_dir: Path,
    out_dir: Path,
    *,
    project_root: Path,
    accelerator: str,
    cli_args_base: Sequence[str],
    timeout: int,
    env: dict[str, str],
    log: _LoggerLike,
) -> subprocess.CompletedProcess[str]:
    """Run ``boltz predict`` for a directory of YAML inputs.

    Args:
        input_dir: Directory with YAML files.
        out_dir: Output directory.
        project_root: Working directory for subprocess.
        accelerator: Boltz accelerator token.
        cli_args_base: Shared Boltz CLI arguments.
        timeout: Timeout in seconds.
        env: Subprocess environment dictionary.
        log: Logger-like instance.

    Returns:
        Completed subprocess result.
    """
    cmd = [
        "uv",
        "run",
        "boltz",
        "predict",
        str(input_dir),
        "--out_dir",
        str(out_dir),
        "--accelerator",
        accelerator,
        *cli_args_base,
    ]
    log.debug(f"Boltz batch CMD: {' '.join(cmd)}")

    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(project_root),
        env=env,
        check=False,
    )

    if result.stderr:
        _log_boltz_stderr(result.stderr, log=log)
    return result


def score_single_internal(
    target_id: str,
    smiles: str,
    sequence: str,
    msa_path: Path | None,
    *,
    device: str | None,
    write_boltz_yaml: _WriteYamlFn,
    run_boltz_predict: _RunPredictFn,
    parse_affinity: _ParseAffinityFn,
    log: _LoggerLike,
) -> float:
    """Score one molecule in an isolated temporary workspace.

    This helper is used as a fallback path after failed batch runs.

    Args:
        target_id: Target identifier.
        smiles: Neutralized ligand SMILES.
        sequence: Target protein sequence.
        msa_path: Optional MSA CSV path.
        device: Optional CUDA device token.
        write_boltz_yaml: YAML writer callback.
        run_boltz_predict: CLI predictor callback.
        parse_affinity: Affinity parser callback.
        log: Logger-like instance.

    Returns:
        Affinity score or ``NaN`` on failures.
    """
    with tempfile.TemporaryDirectory(prefix=f"boltz_retry_{target_id}_") as tmpdir:
        tmp_path = Path(tmpdir)
        yaml_stem = "input"
        yaml_path = tmp_path / f"{yaml_stem}.yaml"
        out_dir = tmp_path / "output"

        write_boltz_yaml(sequence, smiles, yaml_path, msa_path=msa_path)

        try:
            result = run_boltz_predict(yaml_path, out_dir, device=device)
        except subprocess.TimeoutExpired:
            log.debug(f"Boltz retry timeout for SMILES: {smiles}")
            return float("nan")

        if result.returncode != 0:
            log.debug(f"Boltz retry failed (rc={result.returncode}) for SMILES: {smiles}")
            return float("nan")

        try:
            return parse_affinity(out_dir, yaml_stem)
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            return float("nan")
