"""Stage 0: Fetch — скачать ВСЁ необходимое для offline-работы.

После успешного выполнения этого скрипта дальнейшие стадии
(calibration, generation, scoring, analysis) не требуют интернета.

Порядок операций:
  1. uv sync              — Python-зависимости в .venv/
  2. fetch_freed_data     — docker/freedpp/data/{zinc_crem.json, alert_collections.csv}
  3. docker build         — Docker-образ FREED++ (CUDA, PyTorch, DGL, UniDock, FFREED)
  4. fetch_calibration    — data/calibration/*.smi (HuggingFace/URL)
  5. fetch_baseline       — data/baseline/*.tar.xz (опционально, URL)
  6. prepare_targets      — PDB (RCSB), FASTA (UniProt), MSA (ColabFold), PDBQT

Использование:
  uv run python scripts/stage0_fetch.py
  uv run python scripts/stage0_fetch.py --skip-baseline --skip-docker-build
  uv run python scripts/stage0_fetch.py --dry-run
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import click
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCKER_BIN = shutil.which("docker") or "docker"


def _run(cmd: list[str], *, dry_run: bool, timeout: int | None = None, label: str = "") -> bool:
    """Run command, return True on success."""
    tag = f"[{label}] " if label else ""
    logger.info("{}RUN: {}", tag, " ".join(cmd))
    if dry_run:
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
    """Check if artifact exists; log status."""
    exists = path.exists()
    status = "OK" if exists else "MISSING"
    logger.info("[CHECK] {} — {} ({})", label, status, path)
    return exists


# ── Step 1: Python dependencies ──────────────────────────────────────────


def step_uv_sync(dry_run: bool) -> bool:
    """Ensure .venv/ has all Python packages."""
    return _run(["uv", "sync"], dry_run=dry_run, timeout=600, label="uv-sync")


# ── Step 2: FREED data files ─────────────────────────────────────────────


def step_fetch_freed_data(dry_run: bool, force: bool = False) -> bool:
    """Download zinc_crem.json + alert_collections.csv for Docker build."""
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "fetch_freed_data.py")]
    if force:
        cmd.append("--force")
    return _run(cmd, dry_run=dry_run, timeout=120, label="freed-data")


# ── Step 3: Docker image build ───────────────────────────────────────────


def _docker_image_exists(image: str) -> bool:
    result = subprocess.run(  # noqa: S603
        [DOCKER_BIN, "image", "inspect", image],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def step_docker_build(dry_run: bool, image: str = "dualdocking/freedpp:latest", force: bool = False) -> bool:
    """Build FREED++ Docker image (downloads CUDA, PyTorch, DGL, FFREED, UniDock)."""
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
        str(PROJECT_ROOT / "docker" / "freedpp"),
    ]
    return _run(cmd, dry_run=dry_run, timeout=3600, label="docker-build")


# ── Step 4: Calibration SMILES ───────────────────────────────────────────


def step_fetch_calibration(dry_run: bool) -> bool:
    """Download calibration SMILES datasets (HuggingFace/URL)."""
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "fetch_calibration_sets.py")]
    return _run(cmd, dry_run=dry_run, timeout=300, label="calibration")


# ── Step 5: Baseline ZINC ────────────────────────────────────────────────


def step_fetch_baseline(dry_run: bool, url: str | None = None) -> bool:
    """Download baseline ZINC archive."""
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "fetch_baseline_zinc.py")]
    if url:
        cmd.extend(["--url", url])
    return _run(cmd, dry_run=dry_run, timeout=1800, label="baseline")


# ── Step 6: Prepare targets (PDB + FASTA + MSA + PDBQT) ─────────────────


def step_prepare_targets(dry_run: bool) -> bool:
    """Download PDB (RCSB), FASTA (UniProt), pre-compute MSA (ColabFold), PDB→PDBQT."""
    cmd = ["uv", "run", "python", str(PROJECT_ROOT / "scripts" / "prepare_targets.py"), "--prepare-all"]
    return _run(cmd, dry_run=dry_run, timeout=7200, label="targets")


# ── Verification ─────────────────────────────────────────────────────────


def verify_offline_readiness() -> list[str]:
    """Check that all critical artifacts exist. Return list of missing items."""
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
            pdb_path = PROJECT_ROOT / "data" / "targets" / pair_id / role_letter / f"{pdb_id}.pdb"
            if not pdb_path.exists():
                missing.append(f"PDB: {pdb_path.relative_to(PROJECT_ROOT)}")

            # FASTA
            fasta_path = PROJECT_ROOT / "data" / "targets" / f"{tid}.fasta"
            if not fasta_path.exists():
                missing.append(f"FASTA: {fasta_path.relative_to(PROJECT_ROOT)}")

            # MSA
            msa_path = PROJECT_ROOT / "cache" / "boltz_msa" / f"{tid}.csv"
            if not msa_path.exists():
                missing.append(f"MSA: cache/boltz_msa/{tid}.csv")

    # FREED data
    for fname in ("zinc_crem.json", "alert_collections.csv"):
        p = PROJECT_ROOT / "docker" / "freedpp" / "data" / fname
        if not p.exists():
            missing.append(f"FREED: docker/freedpp/data/{fname}")

    # Calibration
    cal_dir = PROJECT_ROOT / "data" / "calibration"
    smi_files = list(cal_dir.glob("*.smi")) if cal_dir.exists() else []
    if not smi_files:
        missing.append("Calibration: data/calibration/*.smi (none found)")

    # Docker image
    if not _docker_image_exists("dualdocking/freedpp:latest"):
        missing.append("Docker: dualdocking/freedpp:latest image not found")

    return missing


# ── CLI ──────────────────────────────────────────────────────────────────


@click.command()
@click.option("--dry-run", is_flag=True, default=False, help="Показать команды без выполнения")
@click.option("--skip-uv-sync", is_flag=True, default=False, help="Пропустить uv sync")
@click.option("--skip-freed-data", is_flag=True, default=False, help="Пропустить FREED data files")
@click.option("--skip-docker-build", is_flag=True, default=False, help="Пропустить сборку Docker-образа")
@click.option("--skip-calibration", is_flag=True, default=False, help="Пропустить calibration SMILES")
@click.option("--skip-baseline", is_flag=True, default=False, help="Пропустить baseline ZINC")
@click.option("--skip-targets", is_flag=True, default=False, help="Пропустить PDB/FASTA/MSA/PDBQT")
@click.option("--baseline-url", default=None, help="URL для baseline archive")
@click.option("--docker-image", default="dualdocking/freedpp:latest", show_default=True)
@click.option("--force-docker", is_flag=True, default=False, help="Пересобрать Docker-образ")
@click.option("--verify-only", is_flag=True, default=False, help="Только проверить готовность, не скачивать")
def main(  # noqa: PLR0913
    dry_run: bool,
    skip_uv_sync: bool,
    skip_freed_data: bool,
    skip_docker_build: bool,
    skip_calibration: bool,
    skip_baseline: bool,
    skip_targets: bool,
    baseline_url: str | None,
    docker_image: str,
    force_docker: bool,
    verify_only: bool,
) -> None:
    """Stage 0: Fetch — скачать ВСЁ для offline-работы.

    После выполнения дальнейшие стадии не требуют интернета.
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
