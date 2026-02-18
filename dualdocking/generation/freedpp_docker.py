"""Docker runtime and training-log helpers for FREED++ generation."""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from dualdocking.reward import reward_mode_requires_dual_docking, reward_mode_requires_secondary_target


class _LoggerLike(Protocol):
    """Minimal logger protocol for FREED++ Docker helpers."""

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

    def warning(self, message: str, *args: object) -> None:
        """Log warning message.

        Args:
            message: Message template or text.
            *args: Optional formatting arguments.
        """

    def error(self, message: str, *args: object) -> None:
        """Log error message.

        Args:
            message: Message template or text.
            *args: Optional formatting arguments.
        """


class _GenerationConfigLike(Protocol):
    """Protocol for config fields used by Docker helpers."""

    reward_mode: str
    target_id: str
    second_target_id: str | None
    qg_max_invalid_fraction: float
    qg_max_timeout_fraction: float
    docker_image: str
    seed: int
    epochs: int
    training_timeout: int
    pass_reward_modes_to_container: bool
    reward_modes_path_in_container: str


class _PopenLike(Protocol):
    """Protocol for subprocess-like objects returned by ``Popen``."""

    returncode: int | None
    stdout: Iterable[str] | None

    def wait(self, timeout: int | None = None) -> object:
        """Wait for process completion.

        Args:
            timeout: Timeout in seconds.

        Returns:
            Implementation-specific result.
        """


class _ProgressLike(Protocol):
    """Protocol for tqdm-like progress bars."""

    n: int

    def refresh(self) -> None:
        """Refresh displayed progress."""

    def set_postfix_str(self, s: str) -> None:
        """Set postfix string.

        Args:
            s: Postfix message.
        """

    def close(self) -> None:
        """Close progress bar."""


_RunSubprocessFn = Callable[..., object]
_RunProjectCommandFn = Callable[..., object]
_BuildArgsFn = Callable[..., list[str]]
_PopenFactory = Callable[..., object]
_TqdmFactory = Callable[..., _ProgressLike | None]
_GetEnvFn = Callable[[str, str | None], str | None]

# Matches patterns like: loss=0.123, reward=-7.5, score: -6.2
_METRIC_RE = re.compile(
    r"(?P<key>loss|reward|score|entropy|lr|kl)\s*[=:]\s*(?P<value>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)

# Epoch/iteration number extraction
_EPOCH_RE = re.compile(
    r"[Ee](?:poch|pisode)\s*[:\[#]?\s*(\d+)"
    r"|#(\d+)"
    r"|\[(\d+)/\d+\]"
    r"|[Ii]ter(?:ation)?\s*[:\[#]?\s*(\d+)"
)


def _resolve_docker_gpu_flag(getenv: _GetEnvFn) -> str:
    """Resolve Docker ``--gpus`` argument from environment variables.

    Priority:
    1. ``FREEDPP_DOCKER_GPUS`` explicit Docker value (for example ``all``,
       ``device=0`` or ``device=1,3``).
    2. ``FREEDPP_CUDA_DEVICE`` mapped to ``device=<token>``.
    3. Fallback ``all``.

    Args:
        getenv: Environment lookup callback.

    Returns:
        Value for Docker ``--gpus`` flag.
    """
    explicit = (getenv("FREEDPP_DOCKER_GPUS", None) or "").strip()
    if explicit:
        return explicit

    gpu_device = (getenv("FREEDPP_CUDA_DEVICE", None) or "").strip()
    if gpu_device:
        return f"device={gpu_device}"
    return "all"


def _result_returncode(result: object) -> int:
    """Safely extract integer return code from subprocess-like result.

    Args:
        result: Subprocess result object.

    Returns:
        Return code, or 1 when unavailable.
    """
    value = getattr(result, "returncode", 1)
    return value if isinstance(value, int) else 1


def _result_text(result: object, field: str) -> str:
    """Safely extract stdout/stderr text from subprocess-like result.

    Args:
        result: Subprocess result object.
        field: Attribute name (``"stdout"`` or ``"stderr"``).

    Returns:
        Text value (empty string when absent).
    """
    value = getattr(result, field, "")
    return value if isinstance(value, str) else ""


def ensure_docker_image(
    image: str,
    *,
    docker_bin: str,
    project_root: Path,
    freed_data_files: tuple[Path, ...],
    run_subprocess: _RunSubprocessFn,
    run_project_command: _RunProjectCommandFn,
    log: _LoggerLike,
) -> None:
    """Ensure FREED++ Docker image exists, building it if absent.

    Args:
        image: Docker image tag.
        docker_bin: Docker binary path.
        project_root: Project root directory.
        freed_data_files: Required FREED data files.
        run_subprocess: Subprocess runner callback.
        run_project_command: Project command runner callback.
        log: Logger-like instance.

    Raises:
        RuntimeError: Docker inspection/build or data fetch failed.
        FileNotFoundError: Dockerfile is missing.
    """
    result = run_subprocess(
        [docker_bin, "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if _result_returncode(result) == 0:
        log.debug(f"Docker image {image} found")
        return

    stderr = _result_text(result, "stderr")
    if "No such image" not in stderr and "not found" not in stderr.lower():
        raise RuntimeError(f"Docker inspect failed (rc={_result_returncode(result)}): {stderr.strip()}")

    log.info(f"Docker image {image} not found, building...")
    dockerfile_dir = project_root / "docker" / "freedpp"
    dockerfile_path = dockerfile_dir / "Dockerfile"
    if not dockerfile_path.exists():
        raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")

    missing_files = [path for path in freed_data_files if not path.exists()]
    if missing_files:
        log.info(f"Missing FREED data files, downloading: {', '.join(str(path) for path in missing_files)}")
        fetch_result = run_project_command(
            ["python", "cli/s0_fetch/fetch_freed_fragments.py"],
            project_root=project_root,
            capture_output=True,
            allow_inprocess=True,
        )
        if _result_returncode(fetch_result) != 0:
            tail = (_result_text(fetch_result, "stdout") + _result_text(fetch_result, "stderr"))[-2000:]
            raise RuntimeError(f"Failed to download FREED data files:\n{tail}")

    run_subprocess(
        [
            docker_bin,
            "build",
            "-t",
            image,
            "-f",
            str(dockerfile_path),
            str(project_root),
        ],
        check=True,
    )
    log.info(f"Docker image {image} built")


def run_freedpp(  # noqa: PLR0913
    config: _GenerationConfigLike,
    workspace: Path,
    pdbqt_a: Path,
    center_a: str,
    size_a: str,
    *,
    pdbqt_b: Path | None,
    center_b: str | None,
    size_b: str | None,
    commands: str,
    docker_bin: str,
    build_freedpp_args: _BuildArgsFn,
    popen: _PopenFactory,
    pipe: int,
    getenv: _GetEnvFn,
    tqdm_factory: _TqdmFactory,
    log: _LoggerLike,
    extra_docker_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run FREED++ inside Docker and stream logs with progress extraction.

    Args:
        config: Generation configuration.
        workspace: Host workspace mounted to ``/workspace``.
        pdbqt_a: Primary receptor path.
        center_a: Primary binding-box center.
        size_a: Primary binding-box size.
        pdbqt_b: Optional secondary receptor path.
        center_b: Optional secondary binding-box center.
        size_b: Optional secondary binding-box size.
        commands: FREED++ command string.
        docker_bin: Docker binary path.
        build_freedpp_args: Callback for CLI argument construction.
        popen: Process-spawn callback.
        pipe: Sentinel equivalent to ``subprocess.PIPE``.
        getenv: Environment lookup callback.
        tqdm_factory: Progress-bar factory callback.
        log: Logger-like instance.

    Returns:
        Completed process object with combined stdout.

    Raises:
        ValueError: ``single_b`` mode is missing receptor-B inputs.
        subprocess.CalledProcessError: Docker process exits non-zero.
    """
    mode_requires_dual = reward_mode_requires_dual_docking(config.reward_mode)
    mode_requires_secondary = reward_mode_requires_secondary_target(config.reward_mode)

    if mode_requires_secondary and not mode_requires_dual:
        if pdbqt_b is None:
            raise ValueError(f"{config.reward_mode} requires receptor B (second_target_id must be set)")
        if center_b is None or size_b is None:
            raise ValueError(f"{config.reward_mode} requires center_b and size_b")
        args = build_freedpp_args(config, pdbqt_b, center_b, size_b, commands)
    else:
        args = build_freedpp_args(config, pdbqt_a, center_a, size_a, commands)

    gpu_flag = _resolve_docker_gpu_flag(getenv)
    patch_log_level = getenv("DUAL_REWARD_PATCH_LOG_LEVEL", "INFO") or "INFO"
    prep_mp_start_method = getenv("FREEDPP_PREP_MP_START_METHOD", "spawn") or "spawn"
    prep_max_workers = getenv("FREEDPP_PREP_MAX_WORKERS", "8") or "8"
    force_spawn = getenv("FREEDPP_FORCE_SPAWN", "1") or "1"
    default_python_warnings = "ignore:invalid escape sequence:SyntaxWarning"
    python_warnings = getenv("PYTHONWARNINGS", default_python_warnings) or default_python_warnings

    docker_cmd: list[str] = [
        docker_bin,
        "run",
        "--rm",
        "--gpus",
        gpu_flag,
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "DGLBACKEND=pytorch",
        "-e",
        f"PYTHONWARNINGS={python_warnings}",
        "-e",
        f"DUAL_REWARD_PATCH_LOG_LEVEL={patch_log_level}",
        "-e",
        f"FREEDPP_PREP_MP_START_METHOD={prep_mp_start_method}",
        "-e",
        f"FREEDPP_PREP_MAX_WORKERS={prep_max_workers}",
        "-e",
        f"FREEDPP_FORCE_SPAWN={force_spawn}",
        "-v",
        f"{workspace}:/workspace",
        "-e",
        f"REWARD_MODE={config.reward_mode}",
        "-e",
        f"TARGET_ID_A={config.target_id}",
        "-e",
        f"TARGET_ID_B={config.second_target_id or ''}",
        "-e",
        "NORMALIZATION_PATH=/workspace/normalization.yaml",
        "-e",
        f"QG_MAX_INVALID_FRACTION={config.qg_max_invalid_fraction}",
        "-e",
        f"QG_MAX_TIMEOUT_FRACTION={config.qg_max_timeout_fraction}",
        "-e",
        "GPU_BATCH_DOCK=1",
    ]

    if config.pass_reward_modes_to_container:
        reward_modes_path = config.reward_modes_path_in_container.strip()
        if not reward_modes_path:
            raise ValueError("pass_reward_modes_to_container=true requires reward_modes_path_in_container")
        docker_cmd.extend(
            [
                "-e",
                f"DUALDOCKING_REWARD_MODES_PATH={reward_modes_path}",
            ]
        )

    if mode_requires_dual and pdbqt_b is not None:
        dual_cfg = {
            "receptor_b": f"/workspace/{pdbqt_b.name}",
            "center_b": center_b,
            "size_b": size_b,
        }
        dual_config_path = workspace / "dual_config.json"
        dual_config_path.write_text(json.dumps(dual_cfg, indent=2))

        docker_cmd.extend(
            [
                "-e",
                f"DUAL_RECEPTOR_B=/workspace/{pdbqt_b.name}",
                "-e",
                "DUAL_CONFIG=/workspace/dual_config.json",
            ]
        )

    if extra_docker_args:
        docker_cmd.extend(extra_docker_args)

    docker_cmd.append(config.docker_image)
    docker_cmd.extend(args)

    def _ts() -> str:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")

    run_id = getenv("EXPERIMENT_RUN_ID", "adhoc") or "adhoc"
    log_ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = workspace / f"freedpp_{config.reward_mode}_seed_{config.seed}_{run_id}_{log_ts}.log"

    # Build a short tag for log lines: [GPU 0 | single_a/seed_42]
    tag = f"GPU {gpu_flag.removeprefix('device=')} | {config.reward_mode}/seed_{config.seed}"

    log.info(f"[{tag}] Starting FREED++: {' '.join(docker_cmd)}")
    log.info(f"[{tag}] Log -> {log_path}")

    epoch_bar: _ProgressLike | None = None
    log_file = None
    captured_lines: list[str] = []
    process: _PopenLike | None = None

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        log_file.write(f"[{_ts()}] RUN_ID={run_id}\n")
        log_file.write(f"[{_ts()}] CMD: {' '.join(docker_cmd)}\n")
        log_file.flush()

        process = cast(
            "_PopenLike",
            popen(
                docker_cmd,
                stdout=pipe,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ),
        )

        if "train" in commands:
            epoch_bar = tqdm_factory(total=config.epochs, desc=f"FREED++ [{tag}]", unit="epoch")

        train_start = time.monotonic()
        last_epoch_logged = -1
        line_count = 0
        noise_patterns = ("DeprecationWarning", "SyntaxWarning", "FutureWarning", "UserWarning", "warnings.warn")

        stdout_iter = process.stdout if process.stdout is not None else []
        for line in stdout_iter:
            stripped = line.rstrip()
            log_file.write(f"[{_ts()}] {stripped}\n")
            log_file.flush()
            line_count += 1

            if any(pattern in stripped for pattern in noise_patterns):
                captured_lines.append(line)
                continue

            if epoch_bar is not None:
                match = _EPOCH_RE.search(stripped)
                if match:
                    epoch_num = int(next(group for group in match.groups() if group is not None))
                    epoch_bar.n = min(epoch_num, config.epochs)
                    epoch_bar.refresh()
                    log_interval = max(1, config.epochs // 10)
                    if epoch_num > 0 and epoch_num % log_interval == 0 and epoch_num != last_epoch_logged:
                        last_epoch_logged = epoch_num
                        elapsed = time.monotonic() - train_start
                        sec_per_epoch = elapsed / epoch_num
                        remaining = (config.epochs - epoch_num) * sec_per_epoch
                        log.info(
                            f"[{tag}] Epoch {epoch_num}/{config.epochs} — "
                            f"{elapsed:.0f}s elapsed, ~{remaining / 60:.0f}m remaining ({sec_per_epoch:.1f}s/epoch)",
                        )
                elif line_count % 500 == 0:
                    elapsed = time.monotonic() - train_start
                    epoch_bar.set_postfix_str(f"{line_count} lines, {elapsed:.0f}s")

            low = stripped.lower()
            if any(keyword in low for keyword in ("error", "sample", "saving", "loss", "score", "result")):
                log.info(f"[{tag}] {stripped}")
            captured_lines.append(line)

        process.wait(timeout=config.training_timeout)
    finally:
        if epoch_bar is not None:
            epoch_bar.close()
        if log_file is not None:
            log_file.close()

    stdout_text = "".join(captured_lines)

    if process is None or process.returncode != 0:
        return_code = process.returncode if process is not None and process.returncode is not None else -1
        log.error(f"[{tag}] FREED++ failed (rc={return_code})")
        log.error(f"[{tag}] output: {stdout_text[-2000:]}")
        log.error(f"[{tag}] log -> {log_path}")
        raise subprocess.CalledProcessError(return_code, docker_cmd, output=stdout_text)

    return subprocess.CompletedProcess(docker_cmd, process.returncode, stdout=stdout_text, stderr="")


def parse_training_log(log_path: Path) -> list[dict[str, int | float]]:
    """Parse a FREED++ runtime log into per-epoch metric records.

    Args:
        log_path: Path to ``freedpp_*.log`` file.

    Returns:
        List of per-epoch records with ``epoch`` and parsed metric fields.
    """
    if not log_path.exists():
        return []

    records: list[dict[str, int | float]] = []
    seen_epochs: set[int] = set()

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    for line in text.splitlines():
        epoch_match = _EPOCH_RE.search(line)
        if not epoch_match:
            continue

        epoch_num = int(next(group for group in epoch_match.groups() if group is not None))
        metrics = {match.group("key").lower(): float(match.group("value")) for match in _METRIC_RE.finditer(line)}
        if not metrics:
            continue

        if epoch_num in seen_epochs:
            for record in records:
                if record["epoch"] == epoch_num:
                    record.update(metrics)
                    break
            continue

        seen_epochs.add(epoch_num)
        records.append({"epoch": epoch_num, **metrics})

    return records
