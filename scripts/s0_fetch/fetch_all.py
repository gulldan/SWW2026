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

    uv run python scripts/s0_fetch/fetch_all.py
    uv run python scripts/s0_fetch/fetch_all.py --skip-baseline --skip-docker-build
    uv run python scripts/s0_fetch/fetch_all.py --dry-run
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

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DOCKER_BIN: str = shutil.which("docker") or "docker"


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
        allow_inprocess: Try in-process execution for internal Python scripts.

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


def _check_artifact(path: Path, label: str) -> bool:
    """Check whether an expected artifact exists on disk.

    Args:
        path: Filesystem path to check.
        label: Human-readable name for log messages.

    Returns:
        True if the path exists.
    """
    exists = path.exists()
    status = "OK" if exists else "MISSING"
    logger.info("[CHECK] {} — {} ({})", label, status, path)
    return exists


# ── Step 1: Python dependencies ──────────────────────────────────────────


def step_uv_sync(dry_run: bool) -> bool:
    """Ensure .venv/ has all Python packages via ``uv sync``.

    Args:
        dry_run: If True, only log the command.

    Returns:
        True on success.
    """
    return _run(["uv", "sync"], dry_run=dry_run, timeout=600, label="uv-sync")


# ── Step 2: FREED data files ─────────────────────────────────────────────


def step_fetch_freed_data(dry_run: bool, force: bool = False) -> bool:
    """Download zinc_crem.json and alert_collections.csv for Docker build.

    Args:
        dry_run: If True, only log the command.
        force: If True, overwrite existing files.

    Returns:
        True on success.
    """
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "s0_fetch" / "fetch_freed_fragments.py")]
    if force:
        cmd.append("--force")
    return _run(cmd, dry_run=dry_run, timeout=120, label="freed-data", allow_inprocess=True)


# ── Step 3: Docker image build ───────────────────────────────────────────


def _docker_image_exists(image: str) -> bool:
    """Check whether a Docker image exists locally.

    Args:
        image: Docker image name/tag to inspect.

    Returns:
        True if the image is available locally.
    """
    result = subprocess.run(  # noqa: S603
        [DOCKER_BIN, "image", "inspect", image],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def step_docker_build(dry_run: bool, image: str = "dualdocking/freedpp:latest", force: bool = False) -> bool:
    """Build the FREED++ Docker image (downloads CUDA, PyTorch, DGL, FFREED, UniDock).

    Args:
        dry_run: If True, only log the command.
        image: Docker image tag to build.
        force: If True, rebuild even when the image already exists.

    Returns:
        True on success.
    """
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


# ── Step 4: Calibration SMILES ───────────────────────────────────────────


def step_fetch_calibration(dry_run: bool) -> bool:
    """Download calibration SMILES datasets (HuggingFace/URL).

    Args:
        dry_run: If True, only log the command.

    Returns:
        True on success.
    """
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "s0_fetch" / "fetch_calibration_smiles.py")]
    return _run(cmd, dry_run=dry_run, timeout=300, label="calibration", allow_inprocess=True)


# ── Step 5: Baseline ZINC ────────────────────────────────────────────────


def step_fetch_baseline(dry_run: bool, url: str | None = None) -> bool:
    """Download the baseline ZINC archive.

    Args:
        dry_run: If True, only log the command.
        url: Optional override URL for the archive.

    Returns:
        True on success.
    """
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "s0_fetch" / "fetch_zinc_baseline.py")]
    if url:
        cmd.extend(["--url", url])
    return _run(cmd, dry_run=dry_run, timeout=1800, label="baseline", allow_inprocess=True)


# ── Step 6: Prepare targets (PDB + FASTA + MSA + PDBQT) ─────────────────


def step_prepare_targets(dry_run: bool) -> bool:
    """Download PDB (RCSB), FASTA (UniProt), pre-compute MSA (ColabFold), PDB to PDBQT.

    Args:
        dry_run: If True, only log the command.

    Returns:
        True on success.
    """
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "s1_prepare" / "prepare_targets.py"), "--prepare-all"]
    return _run(cmd, dry_run=dry_run, timeout=7200, label="targets", allow_inprocess=True)


# ── Step 7: ChEMBL actives ───────────────────────────────────────────────


def step_fetch_chembl_actives(dry_run: bool) -> bool:
    """Download ChEMBL actives for all targets in pairs.yaml.

    Fetches known-active SMILES per target and computes dual-active
    overlaps for each pair.  Used by enrichment validation (Stage 1).

    Args:
        dry_run: If True, only log the command.

    Returns:
        True on success.
    """
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "s0_fetch" / "fetch_chembl_actives.py")]
    return _run(cmd, dry_run=dry_run, timeout=600, label="chembl", allow_inprocess=True)


# ── Step 8: GNINA binary ─────────────────────────────────────────────────


def step_fetch_gnina(dry_run: bool) -> bool:
    """Download GNINA static binary for cross-validation scoring.

    Args:
        dry_run: If True, only log the command.

    Returns:
        True on success.
    """
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "s0_fetch" / "fetch_gnina.py")]
    return _run(cmd, dry_run=dry_run, timeout=300, label="gnina", allow_inprocess=True)


# ── Verification ─────────────────────────────────────────────────────────


def _find_path(relative: str) -> bool:
    """Check if a file exists under ``output/`` or flat layout.

    Tries ``PROJECT_ROOT / "output" / relative`` first, then
    ``PROJECT_ROOT / relative`` (server layout without ``output/`` prefix).

    Args:
        relative: Path relative to the data root (e.g. ``"data/targets/CDK2.fasta"``).

    Returns:
        True if the file exists under either layout.
    """
    return (PROJECT_ROOT / "output" / relative).exists() or (PROJECT_ROOT / relative).exists()


def verify_offline_readiness() -> list[str]:
    """Check that all critical artifacts exist.

    Inspects pairs.yaml targets for PDB, FASTA, MSA files; checks FREED
    docker data; calibration SMILES; and the Docker image.
    Supports both ``output/`` prefix (local dev) and flat (server) layouts.

    Returns:
        List of human-readable descriptions of missing items (empty if all present).
    """
    import yaml

    missing: list[str] = []

    # pairs.yaml targets
    pairs_path = PROJECT_ROOT / "configs" / "pairs.yaml"
    if not pairs_path.exists():
        missing.append("configs/pairs.yaml")
        return missing

    pairs = yaml.safe_load(pairs_path.read_text())["pairs"]
    for pair_id, pair_cfg in pairs.items():
        for role in ("target_a", "target_b"):
            target = pair_cfg[role]
            tid = target["id"]
            pdb_id = target["pdb_id"]
            role_letter = role[-1].upper()

            # PDB
            pdb_rel = f"data/targets/{pair_id}/{role_letter}/{pdb_id.lower()}.pdb"
            if not _find_path(pdb_rel):
                missing.append(f"PDB: {pdb_rel}")

            # FASTA
            fasta_rel = f"data/targets/{tid}.fasta"
            if not _find_path(fasta_rel):
                missing.append(f"FASTA: {fasta_rel}")

            # MSA
            msa_rel = f"cache/boltz_msa/{tid}.csv"
            if not _find_path(msa_rel):
                missing.append(f"MSA: {msa_rel}")

    # FREED data
    for fname in ("zinc_crem.json", "alert_collections.csv"):
        p = PROJECT_ROOT / "docker" / "freedpp" / "data" / fname
        if not p.exists():
            missing.append(f"FREED: docker/freedpp/data/{fname}")

    # Calibration
    cal_dirs = [PROJECT_ROOT / "output" / "data" / "calibration", PROJECT_ROOT / "data" / "calibration"]
    smi_files: list[Path] = []
    for cal_dir in cal_dirs:
        if cal_dir.exists():
            smi_files.extend(cal_dir.glob("*.smi"))
    if not smi_files:
        missing.append("Calibration: data/calibration/*.smi (none found)")

    # Docker image
    if not _docker_image_exists("dualdocking/freedpp:latest"):
        missing.append("Docker: dualdocking/freedpp:latest image not found")

    return missing


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
    steps: list[tuple[str, bool]] = []

    # Step 1: Python deps
    if not skip_uv_sync:
        ok = step_uv_sync(dry_run)
        steps.append(("uv sync", ok))
        if not ok:
            logger.error("uv sync failed — aborting")
            sys.exit(1)

    # Step 2: FREED data (needed BEFORE docker build)
    if not skip_freed_data:
        ok = step_fetch_freed_data(dry_run)
        steps.append(("FREED data", ok))

    # Step 3: Docker build (downloads CUDA, PyTorch, DGL, FFREED, UniDock)
    if not skip_docker_build:
        ok = step_docker_build(dry_run, image=docker_image, force=force_docker)
        steps.append(("Docker build", ok))

    # Step 4: Calibration SMILES
    if not skip_calibration:
        ok = step_fetch_calibration(dry_run)
        steps.append(("Calibration SMILES", ok))

    # Step 5: Baseline ZINC (optional)
    if not skip_baseline:
        ok = step_fetch_baseline(dry_run, url=baseline_url)
        steps.append(("Baseline ZINC", ok))

    # Step 6: Targets (PDB + FASTA + MSA + PDBQT) — heaviest, last
    if not skip_targets:
        ok = step_prepare_targets(dry_run)
        steps.append(("Targets (PDB/FASTA/MSA/PDBQT)", ok))

    # Step 7: ChEMBL actives (for enrichment validation)
    if not skip_chembl:
        ok = step_fetch_chembl_actives(dry_run)
        steps.append(("ChEMBL actives", ok))

    # Step 8: GNINA binary (for cross-validation scoring)
    if not skip_gnina:
        ok = step_fetch_gnina(dry_run)
        steps.append(("GNINA binary", ok))

    elapsed = time.monotonic() - t0

    # Summary
    logger.info("─── Stage 0: Fetch Summary ───")
    for label, ok in steps:
        status = "OK" if ok else "FAIL"
        logger.info("  [{}] {}", status, label)
    logger.info("  Total time: {:.0f}s", elapsed)

    failed = [label for label, ok in steps if not ok]
    if failed:
        logger.error("Stage 0 FAILED: {}", ", ".join(failed))
        sys.exit(1)

    # Verify
    if not dry_run:
        missing = verify_offline_readiness()
        if missing:
            logger.warning("Post-fetch verification: {} missing artifacts:", len(missing))
            for item in missing:
                logger.warning("  - {}", item)
        else:
            logger.success("Stage 0 complete — ready for offline operation")


if __name__ == "__main__":
    main()
