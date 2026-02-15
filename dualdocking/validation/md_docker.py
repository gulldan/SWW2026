"""Docker runner for the MD validation stage.

Builds and runs the MD Docker image (GROMACS + AmberTools + acpype +
OpenBabel) so the host system does not need these tools installed.
Follows the same patterns as :mod:`dualdocking.data.fetch.docker` and
:mod:`dualdocking.generation.freedpp`.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from loguru import logger

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DOCKER_BIN: str = shutil.which("docker") or "docker"
DEFAULT_IMAGE: str = "dualdocking/md:latest"


# ── Image management ────────────────────────────────────────────────


def docker_image_exists(image: str = DEFAULT_IMAGE) -> bool:
    """Check whether a Docker image exists locally."""
    result = subprocess.run(  # noqa: S603
        [DOCKER_BIN, "image", "inspect", image],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def build_md_image(
    project_root: Path | None = None,
    *,
    image: str = DEFAULT_IMAGE,
    cuda_arch: str = "89",
    force: bool = False,
    dry_run: bool = False,
    timeout: int = 7200,
) -> bool:
    """Build the MD Docker image if it does not exist.

    Args:
        project_root: Repository root (Docker build context).
        image: Docker image tag.
        cuda_arch: CUDA architecture for GROMACS (e.g. "89" for RTX 4090).
        force: Rebuild even if the image already exists.
        dry_run: Log the command without executing.
        timeout: Maximum seconds to wait for the build.

    Returns:
        True if the image exists or was built successfully.
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    if not force and docker_image_exists(image):
        logger.info("[md-docker] Image already exists: {}", image)
        return True

    dockerfile = project_root / "docker" / "md" / "Dockerfile"
    if not dockerfile.exists():
        raise FileNotFoundError(f"Dockerfile not found: {dockerfile}")

    cmd = [
        DOCKER_BIN,
        "build",
        "-f", str(dockerfile),
        "-t", image,
        "--build-arg", f"CUDA_ARCHITECTURES={cuda_arch}",
        str(project_root),
    ]

    logger.info("[md-docker] RUN: {}", " ".join(cmd))
    if dry_run:
        return True

    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.error("[md-docker] Build TIMEOUT after {}s", timeout)
        return False

    if result.returncode != 0:
        tail = ((result.stdout or "") + (result.stderr or ""))[-3000:]
        logger.error("[md-docker] Build FAILED (rc={}): {}", result.returncode, tail)
        return False

    logger.info("[md-docker] Image built: {}", image)
    return True


# ── Run MD inside Docker ────────────────────────────────────────────


def run_md_in_docker(  # noqa: PLR0913
    *,
    pair: str,
    mode: str | None = None,
    top_n: int = 3,
    config: str = "configs/pairs.yaml",
    experiment_config: str = "configs/experiment.yaml",
    mmpbsa: bool = False,
    simulation_time: float | None = None,
    no_gpu: bool = False,
    image: str = DEFAULT_IMAGE,
    project_root: Path | None = None,
    dry_run: bool = False,
    timeout: int = 604800,
) -> subprocess.CompletedProcess[str]:
    """Run the MD validation CLI inside a Docker container.

    Volume mounts mirror the host project layout at ``/workspace``.

    Args:
        pair: Pair ID from pairs.yaml.
        mode: Generation mode (None = all configured modes).
        top_n: Top-N molecules per mode.
        config: Path to pairs.yaml relative to project root.
        experiment_config: Path to experiment.yaml relative to project root.
        mmpbsa: Run MM-GBSA calculation.
        simulation_time: Override simulation time in ns.
        no_gpu: Disable GPU acceleration.
        image: Docker image tag.
        project_root: Repository root for bind mounts.
        dry_run: Log command without executing.
        timeout: Maximum seconds for the Docker run.

    Returns:
        CompletedProcess from ``docker run``.
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    gpu_flags = [] if no_gpu else ["--gpus", "all"]

    volumes = [
        "-v", f"{project_root / 'output'}:/workspace/output",
        "-v", f"{project_root / 'configs'}:/workspace/configs:ro",
        "-v", f"{project_root / 'dualdocking'}:/workspace/dualdocking:ro",
        "-v", f"{project_root / 'cli'}:/workspace/cli:ro",
        "-v", f"{project_root / 'pyproject.toml'}:/workspace/pyproject.toml:ro",
    ]

    # Build CLI args to forward into the container entrypoint
    cli_args = ["--pair", pair, "--top-n", str(top_n)]
    cli_args += ["--config", config, "--experiment-config", experiment_config]

    if mode is not None:
        cli_args += ["--mode", mode]
    if mmpbsa:
        cli_args.append("--mmpbsa")
    if simulation_time is not None:
        cli_args += ["--simulation-time", str(simulation_time)]
    if no_gpu:
        cli_args.append("--no-gpu")

    cmd = [
        DOCKER_BIN, "run", "--rm",
        *gpu_flags,
        *volumes,
        image,
        *cli_args,
    ]

    logger.info("[md-docker] RUN: {}", " ".join(cmd))
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="[dry-run]", stderr="")

    result = subprocess.run(  # noqa: S603
        cmd,
        cwd=str(project_root),
        stdout=None,
        stderr=None,
        timeout=timeout,
        check=False,
    )

    if result.returncode != 0:
        tail = ((result.stdout or "") + (result.stderr or ""))[-3000:]
        logger.error("[md-docker] Container FAILED (rc={}): {}", result.returncode, tail)
    else:
        logger.info("[md-docker] Container finished successfully")

    return result
