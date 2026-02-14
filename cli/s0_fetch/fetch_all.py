"""Stage 0: Fetch all artifacts required for offline operation.

After successful execution of this script, subsequent stages
(calibration, generation, scoring, analysis) require no internet access.

Operations in order:
  1. uv sync              -- Python dependencies into .venv/
  2. fetch_freed_data     -- docker/freedpp/data/{zinc_crem.json, alert_collections.csv}
  3. docker build         -- Docker image FREED++ (CUDA, PyTorch, DGL, UniDock, FFREED)
  4. fetch_calibration    -- data/calibration/*.smi (HuggingFace/URL)
  5. fetch_baseline       -- data/baseline/*.tar.xz (optional, URL)
  6. prepare_targets      -- PDB (RCSB), FASTA (UniProt), MSA (ColabFold), PDBQT

Usage::

    uv run python cli/s0_fetch/fetch_all.py
    uv run python cli/s0_fetch/fetch_all.py --skip-baseline --skip-docker-build
    uv run python cli/s0_fetch/fetch_all.py --dry-run

Thin CLI wrapper -- core logic lives in :mod:`dualdocking.pipeline.stages.fetch`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import click
from loguru import logger

from dualdocking.command_runner import run_project_command
from dualdocking.pipeline.stages.fetch import (
    fetch_all as _fetch_all_impl,
)
from dualdocking.pipeline.stages.fetch import (
    verify_offline_readiness as _verify_offline_readiness_impl,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DOCKER_BIN: str = shutil.which("docker") or "docker"


def verify_offline_readiness() -> list[str]:
    """Check that all critical artifacts exist.

    Wraps the library function with the module-level PROJECT_ROOT.

    Returns:
        List of human-readable descriptions of missing items.
    """
    return _verify_offline_readiness_impl(PROJECT_ROOT)


# ── Backward-compat shims (used by tests via monkeypatch) ────────────────


def _run(
    cmd: list[str],
    *,
    dry_run: bool,
    timeout: int | None = None,
    label: str = "",
    allow_inprocess: bool = False,
) -> bool:
    """Run a shell command, logging output and status.

    Args:
        cmd: Command and arguments to execute.
        dry_run: If True, log the command without executing.
        timeout: Maximum seconds to wait before killing the process.
        label: Optional prefix for log messages.
        allow_inprocess: Try in-process execution for internal Python cli.

    Returns:
        True if the command succeeded (or was skipped via dry_run).
    """
    tag = f"[{label}] " if label else ""
    logger.info("{}RUN: {}", tag, " ".join(cmd))
    if dry_run:
        return True

    if allow_inprocess:
        try:
            result = run_project_command(
                cmd,
                project_root=PROJECT_ROOT,
                timeout=timeout,
                capture_output=True,
                allow_inprocess=True,
            )
        except RuntimeError as exc:
            logger.error("{}FAILED (in-process): {}\n{}", tag, " ".join(cmd), str(exc)[-3000:])
            return False
        if result.returncode != 0:
            tail = ((result.stdout or "") + (result.stderr or ""))[-3000:]
            logger.error("{}FAILED (rc={}): {}\n{}", tag, result.returncode, " ".join(cmd), tail)
            return False
        return True

    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.error("{}TIMEOUT after {}s: {}", tag, timeout, " ".join(cmd))
        return False
    if result.returncode != 0:
        tail = ((result.stdout or "") + (result.stderr or ""))[-3000:]
        logger.error("{}FAILED (rc={}): {}\n{}", tag, result.returncode, " ".join(cmd), tail)
        return False
    return True


def _docker_image_exists(image: str) -> bool:
    """Check whether a Docker image exists locally."""
    from dualdocking.data.fetch.docker import docker_image_exists

    return docker_image_exists(image)


def step_uv_sync(dry_run: bool) -> bool:
    """Ensure .venv/ has all Python packages via ``uv sync``."""
    return _run(["uv", "sync"], dry_run=dry_run, timeout=600, label="uv-sync")


def step_fetch_freed_data(dry_run: bool, force: bool = False) -> bool:
    """Download zinc_crem.json and alert_collections.csv for Docker build."""
    cmd = [sys.executable, str(PROJECT_ROOT / "cli" / "s0_fetch" / "fetch_freed_fragments.py")]
    if force:
        cmd.append("--force")
    return _run(cmd, dry_run=dry_run, timeout=120, label="freed-data", allow_inprocess=True)


def step_docker_build(dry_run: bool, image: str = "dualdocking/freedpp:latest", force: bool = False) -> bool:
    """Build the FREED++ Docker image."""
    if not force and _docker_image_exists(image):
        logger.info("[docker-build] Image already exists: {}", image)
        return True
    cmd = [
        DOCKER_BIN,
        "build",
        "-f",
        str(PROJECT_ROOT / "docker" / "freedpp" / "Dockerfile"),
        "-t",
        image,
        str(PROJECT_ROOT),
    ]
    return _run(cmd, dry_run=dry_run, timeout=3600, label="docker-build")


def step_fetch_calibration(dry_run: bool) -> bool:
    """Download calibration SMILES datasets."""
    cmd = [sys.executable, str(PROJECT_ROOT / "cli" / "s0_fetch" / "fetch_calibration_smiles.py")]
    return _run(cmd, dry_run=dry_run, timeout=300, label="calibration", allow_inprocess=True)


def step_fetch_baseline(dry_run: bool, url: str | None = None) -> bool:
    """Download the baseline ZINC archive."""
    cmd = [sys.executable, str(PROJECT_ROOT / "cli" / "s0_fetch" / "fetch_zinc_baseline.py")]
    if url:
        cmd.extend(["--url", url])
    return _run(cmd, dry_run=dry_run, timeout=1800, label="baseline", allow_inprocess=True)


def step_prepare_targets(dry_run: bool) -> bool:
    """Download PDB, FASTA, pre-compute MSA, convert to PDBQT."""
    cmd = [sys.executable, str(PROJECT_ROOT / "cli" / "s1_prepare" / "prepare_targets.py"), "--prepare-all"]
    return _run(cmd, dry_run=dry_run, timeout=7200, label="targets", allow_inprocess=True)


def step_fetch_chembl_actives(dry_run: bool) -> bool:
    """Download ChEMBL actives for all targets in pairs.yaml."""
    cmd = [sys.executable, str(PROJECT_ROOT / "cli" / "s0_fetch" / "fetch_chembl_actives.py")]
    return _run(cmd, dry_run=dry_run, timeout=600, label="chembl", allow_inprocess=True)


def step_fetch_gnina(dry_run: bool) -> bool:
    """Download GNINA static binary for cross-validation scoring."""
    cmd = [sys.executable, str(PROJECT_ROOT / "cli" / "s0_fetch" / "fetch_gnina.py")]
    return _run(cmd, dry_run=dry_run, timeout=300, label="gnina", allow_inprocess=True)


# ── CLI ──────────────────────────────────────────────────────────────────


@click.command()
@click.option("--dry-run", is_flag=True, default=False, help="Print commands without executing")
@click.option("--skip-uv-sync", is_flag=True, default=False, help="Skip uv sync")
@click.option("--skip-freed-data", is_flag=True, default=False, help="Skip FREED data files")
@click.option("--skip-docker-build", is_flag=True, default=False, help="Skip Docker image build")
@click.option("--skip-calibration", is_flag=True, default=False, help="Skip calibration SMILES")
@click.option("--skip-baseline", is_flag=True, default=False, help="Skip baseline ZINC")
@click.option("--skip-targets", is_flag=True, default=False, help="Skip PDB/FASTA/MSA/PDBQT")
@click.option("--skip-chembl", is_flag=True, default=False, help="Skip ChEMBL actives fetch")
@click.option("--skip-gnina", is_flag=True, default=False, help="Skip GNINA binary download")
@click.option("--baseline-url", default=None, help="URL for baseline archive")
@click.option("--docker-image", default="dualdocking/freedpp:latest", show_default=True)
@click.option("--force-docker", is_flag=True, default=False, help="Force Docker image rebuild")
@click.option("--verify-only", is_flag=True, default=False, help="Only check readiness, do not download")
def main(  # noqa: PLR0913
    dry_run: bool,
    skip_uv_sync: bool,
    skip_freed_data: bool,
    skip_docker_build: bool,
    skip_calibration: bool,
    skip_baseline: bool,
    skip_targets: bool,
    skip_chembl: bool,
    skip_gnina: bool,
    baseline_url: str | None,
    docker_image: str,
    force_docker: bool,
    verify_only: bool,
) -> None:
    """Fetch all artifacts needed for offline experiment execution.

    Runs eight sequential steps (uv sync, FREED data, Docker build,
    calibration SMILES, baseline ZINC, target preparation, ChEMBL
    actives, GNINA binary) with per-step skip flags.  Use
    ``--verify-only`` to check readiness without downloading anything.
    """
    if verify_only:
        missing = verify_offline_readiness()
        if missing:
            logger.warning("Offline readiness: {} missing artifacts:", len(missing))
            for item in missing:
                logger.warning("  - {}", item)
            sys.exit(1)
        else:
            logger.success("Offline readiness: ALL artifacts present")
            sys.exit(0)

    t0 = time.monotonic()
    steps, missing = _fetch_all_impl(
        PROJECT_ROOT,
        dry_run=dry_run,
        skip_uv_sync=skip_uv_sync,
        skip_freed_data=skip_freed_data,
        skip_docker_build=skip_docker_build,
        skip_calibration=skip_calibration,
        skip_baseline=skip_baseline,
        skip_targets=skip_targets,
        skip_chembl=skip_chembl,
        skip_gnina=skip_gnina,
        baseline_url=baseline_url,
        docker_image=docker_image,
        force_docker=force_docker,
    )
    elapsed = time.monotonic() - t0

    logger.info("--- Stage 0: Fetch Summary ---")
    for label, ok in steps:
        status = "OK" if ok else "FAIL"
        logger.info("  [{}] {}", status, label)
    logger.info("  Total time: {:.0f}s", elapsed)

    failed = [label for label, ok in steps if not ok]
    if failed:
        logger.error("Stage 0 FAILED: {}", ", ".join(failed))
        sys.exit(1)

    if not dry_run:
        if missing:
            logger.warning("Post-fetch verification: {} missing artifacts:", len(missing))
            for item in missing:
                logger.warning("  - {}", item)
        else:
            logger.success("Stage 0 complete -- ready for offline operation")


if __name__ == "__main__":
    main()
