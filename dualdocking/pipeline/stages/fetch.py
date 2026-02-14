"""Orchestrator for the fetch stage (Stage 0).

Coordinates all data fetching: uv sync, FREED data, Docker build,
calibration SMILES, baseline ZINC, target preparation, ChEMBL actives,
and GNINA binary.

Extracted from ``cli/s0_fetch/fetch_all.py``.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import yaml
from loguru import logger

from dualdocking.data.fetch.docker import (
    build_freedpp_image,
    docker_image_exists,
    fetch_freed_data,
)

_SCRIPT_STEP_SPECS: dict[str, tuple[str, int]] = {
    "calibration": ("cli/s0_fetch/fetch_calibration_smiles.py", 300),
    "baseline": ("cli/s0_fetch/fetch_zinc_baseline.py", 1800),
    "targets": ("cli/s1_prepare/prepare_targets.py", 7200),
    "chembl": ("cli/s0_fetch/fetch_chembl_actives.py", 600),
    "gnina": ("cli/s0_fetch/fetch_gnina.py", 300),
}


def _run_logged_subprocess(
    project_root: Path,
    cmd: list[str],
    *,
    label: str,
    dry_run: bool,
    timeout: int,
) -> bool:
    """Run a subprocess with standardized logging and timeout handling."""
    logger.info("[{}] RUN: {}", label, " ".join(cmd))
    if dry_run:
        logger.info("[{}] DRY RUN", label)
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
        logger.error("[{}] TIMEOUT", label)
        return False
    if result.returncode != 0:
        logger.error("[{}] FAILED (rc={})", label, result.returncode)
        return False
    return True


def _run_python_script_step(
    project_root: Path,
    *,
    label: str,
    script_rel: str,
    timeout: int,
    dry_run: bool,
    extra_args: list[str] | None = None,
) -> bool:
    """Run a stage script via the current Python interpreter."""
    cmd = [sys.executable, str(project_root / script_rel)]
    if extra_args:
        cmd.extend(extra_args)
    return _run_logged_subprocess(project_root, cmd, label=label, dry_run=dry_run, timeout=timeout)


def _run_named_script_step(
    project_root: Path,
    *,
    step_key: str,
    dry_run: bool,
    extra_args: list[str] | None = None,
) -> bool:
    """Run a script step configured in ``_SCRIPT_STEP_SPECS``."""
    script_rel, timeout = _SCRIPT_STEP_SPECS[step_key]
    return _run_python_script_step(
        project_root,
        label=step_key,
        script_rel=script_rel,
        timeout=timeout,
        dry_run=dry_run,
        extra_args=extra_args,
    )


def step_uv_sync(dry_run: bool, project_root: Path) -> bool:
    """Ensure .venv/ has all Python packages via ``uv sync``.

    Args:
        dry_run: If True, only log the command.
        project_root: Project root directory.

    Returns:
        True on success.
    """
    return _run_logged_subprocess(project_root, ["uv", "sync"], label="uv-sync", dry_run=dry_run, timeout=600)


def step_fetch_freed_data(project_root: Path, *, dry_run: bool, force: bool = False) -> bool:
    """Download zinc_crem.json and alert_collections.csv for Docker build.

    Args:
        project_root: Project root directory.
        dry_run: If True, only log the command.
        force: If True, overwrite existing files.

    Returns:
        True on success.
    """
    if dry_run:
        logger.info("[freed-data] DRY RUN: fetch_freed_data")
        return True
    try:
        out_dir = project_root / "docker" / "freedpp" / "data"
        fetch_freed_data(out_dir, force=force)
        return True
    except Exception as exc:
        logger.error("[freed-data] FAILED: {}", exc)
        return False


def step_docker_build(
    project_root: Path,
    *,
    dry_run: bool,
    image: str = "dualdocking/freedpp:latest",
    force: bool = False,
) -> bool:
    """Build the FREED++ Docker image.

    Args:
        project_root: Project root directory.
        dry_run: If True, only log the command.
        image: Docker image tag to build.
        force: If True, rebuild even when the image already exists.

    Returns:
        True on success.
    """
    return build_freedpp_image(project_root, image=image, force=force, dry_run=dry_run)


def step_fetch_calibration(project_root: Path, *, dry_run: bool) -> bool:
    """Download calibration SMILES datasets.

    Args:
        project_root: Project root directory.
        dry_run: If True, only log the command.

    Returns:
        True on success.
    """
    return _run_named_script_step(project_root, step_key="calibration", dry_run=dry_run)


def step_fetch_baseline(project_root: Path, *, dry_run: bool, url: str | None = None) -> bool:
    """Download the baseline ZINC archive.

    Args:
        project_root: Project root directory.
        dry_run: If True, only log the command.
        url: Optional override URL for the archive.

    Returns:
        True on success.
    """
    extra_args = ["--url", url] if url else None
    return _run_named_script_step(project_root, step_key="baseline", dry_run=dry_run, extra_args=extra_args)


def step_prepare_targets(project_root: Path, *, dry_run: bool) -> bool:
    """Download PDB, FASTA, pre-compute MSA, and convert to PDBQT.

    Args:
        project_root: Project root directory.
        dry_run: If True, only log the command.

    Returns:
        True on success.
    """
    return _run_named_script_step(
        project_root,
        step_key="targets",
        dry_run=dry_run,
        extra_args=["--prepare-all"],
    )


def step_fetch_chembl_actives(project_root: Path, *, dry_run: bool) -> bool:
    """Download ChEMBL actives for all targets.

    Args:
        project_root: Project root directory.
        dry_run: If True, only log the command.

    Returns:
        True on success.
    """
    return _run_named_script_step(project_root, step_key="chembl", dry_run=dry_run)


def step_fetch_gnina(project_root: Path, *, dry_run: bool) -> bool:
    """Download GNINA static binary.

    Args:
        project_root: Project root directory.
        dry_run: If True, only log the command.

    Returns:
        True on success.
    """
    return _run_named_script_step(project_root, step_key="gnina", dry_run=dry_run)


def verify_offline_readiness(project_root: Path) -> list[str]:
    """Check that all critical artifacts exist.

    Args:
        project_root: Project root directory.

    Returns:
        List of human-readable descriptions of missing items.
    """
    missing: list[str] = []

    pairs_path = project_root / "configs" / "pairs.yaml"
    if not pairs_path.exists():
        missing.append("configs/pairs.yaml")
        return missing

    pairs = yaml.safe_load(pairs_path.read_text())["pairs"]

    def _find_path(relative: str) -> bool:
        return (project_root / "output" / relative).exists() or (project_root / relative).exists()

    for pair_id, pair_cfg in pairs.items():
        for role in ("target_a", "target_b"):
            target = pair_cfg[role]
            tid = target["id"]
            pdb_id = target["pdb_id"]
            role_letter = role[-1].upper()

            pdb_rel = f"data/targets/{pair_id}/{role_letter}/{pdb_id.lower()}.pdb"
            if not _find_path(pdb_rel):
                missing.append(f"PDB: {pdb_rel}")

            fasta_rel = f"data/targets/{tid}.fasta"
            if not _find_path(fasta_rel):
                missing.append(f"FASTA: {fasta_rel}")

            msa_rel = f"cache/boltz_msa/{tid}.csv"
            if not _find_path(msa_rel):
                missing.append(f"MSA: {msa_rel}")

    for fname in ("zinc_crem.json", "alert_collections.csv"):
        p = project_root / "docker" / "freedpp" / "data" / fname
        if not p.exists():
            missing.append(f"FREED: docker/freedpp/data/{fname}")

    cal_dirs = [project_root / "output" / "data" / "calibration", project_root / "data" / "calibration"]
    smi_files: list[Path] = []
    for cal_dir in cal_dirs:
        if cal_dir.exists():
            smi_files.extend(cal_dir.glob("*.smi"))
    if not smi_files:
        missing.append("Calibration: data/calibration/*.smi (none found)")

    if not docker_image_exists("dualdocking/freedpp:latest"):
        missing.append("Docker: dualdocking/freedpp:latest image not found")

    return missing


def fetch_all(  # noqa: PLR0913
    project_root: Path,
    *,
    dry_run: bool = False,
    skip_uv_sync: bool = False,
    skip_freed_data: bool = False,
    skip_docker_build: bool = False,
    skip_calibration: bool = False,
    skip_baseline: bool = False,
    skip_targets: bool = False,
    skip_chembl: bool = False,
    skip_gnina: bool = False,
    baseline_url: str | None = None,
    docker_image: str = "dualdocking/freedpp:latest",
    force_docker: bool = False,
) -> tuple[list[tuple[str, bool]], list[str]]:
    """Run all fetch steps and return results.

    Args:
        project_root: Project root directory.
        dry_run: If True, only log commands.
        skip_uv_sync: Skip uv sync step.
        skip_freed_data: Skip FREED data files.
        skip_docker_build: Skip Docker image build.
        skip_calibration: Skip calibration SMILES.
        skip_baseline: Skip baseline ZINC.
        skip_targets: Skip PDB/FASTA/MSA/PDBQT.
        skip_chembl: Skip ChEMBL actives fetch.
        skip_gnina: Skip GNINA binary download.
        baseline_url: Optional URL for baseline archive.
        docker_image: Docker image tag.
        force_docker: Force Docker image rebuild.

    Returns:
        Tuple of (step results, missing items from verification).
    """
    t0 = time.monotonic()
    steps: list[tuple[str, bool]] = []

    if not skip_uv_sync:
        ok = step_uv_sync(dry_run, project_root)
        steps.append(("uv sync", ok))
        if not ok:
            return steps, ["uv sync failed"]

    if not skip_freed_data:
        ok = step_fetch_freed_data(project_root, dry_run=dry_run)
        steps.append(("FREED data", ok))

    if not skip_docker_build:
        ok = step_docker_build(project_root, dry_run=dry_run, image=docker_image, force=force_docker)
        steps.append(("Docker build", ok))

    if not skip_calibration:
        ok = step_fetch_calibration(project_root, dry_run=dry_run)
        steps.append(("Calibration SMILES", ok))

    if not skip_baseline:
        ok = step_fetch_baseline(project_root, dry_run=dry_run, url=baseline_url)
        steps.append(("Baseline ZINC", ok))

    if not skip_targets:
        ok = step_prepare_targets(project_root, dry_run=dry_run)
        steps.append(("Targets (PDB/FASTA/MSA/PDBQT)", ok))

    if not skip_chembl:
        ok = step_fetch_chembl_actives(project_root, dry_run=dry_run)
        steps.append(("ChEMBL actives", ok))

    if not skip_gnina:
        ok = step_fetch_gnina(project_root, dry_run=dry_run)
        steps.append(("GNINA binary", ok))

    elapsed = time.monotonic() - t0
    logger.info("Stage 0: Fetch completed in {:.0f}s", elapsed)

    missing: list[str] = []
    if not dry_run:
        missing = verify_offline_readiness(project_root)

    return steps, missing
