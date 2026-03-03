"""Build FREED++ Docker image and fetch fragment/alert data.

Extracted from ``cli/s0_fetch/fetch_freed_fragments.py`` and
``cli/s0_fetch/fetch_all.py``.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from loguru import logger

from dualdocking.data.http_utils import download_file

DOCKER_BIN: str = shutil.which("docker") or "docker"

# AITRICS/FREED motifs text file (line-separated CReM fragments)
MOTIFS_URL: str = "https://raw.githubusercontent.com/AITRICS/FREED/main/gym_molecule/dataset/motifs_zinc_random_92.txt"

# Pat Walters rd_filters merged alert collections (PAINS / SureChEMBL / Glaxo)
ALERT_COLLECTIONS_URL: str = "https://raw.githubusercontent.com/PatWalters/rd_filters/master/rd_filters/data/alert_collection.csv"


def docker_image_exists(image: str) -> bool:
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


def _validate_alert_csv(path: Path) -> None:
    """Validate that the alert collections CSV has a non-empty header.

    Args:
        path: Path to the downloaded CSV file.

    Raises:
        RuntimeError: If the CSV is empty or has no header row.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    if not header:
        raise RuntimeError(f"Empty alert collections CSV: {path}")


def _build_zinc_crem(out_path: Path, timeout: int) -> None:
    """Download FREED motifs and convert to a JSON fragment list.

    Args:
        out_path: Destination path for the JSON file.
        timeout: HTTP request timeout in seconds.

    Raises:
        RuntimeError: If no fragments are parsed from the motifs file.
    """
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        download_file(MOTIFS_URL, tmp_path, timeout=timeout)
        fragments = [line.strip() for line in tmp_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not fragments:
            raise RuntimeError("No fragments parsed from motifs file")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(fragments, indent=2), encoding="utf-8")
        logger.info("Wrote {} fragments -> {}", len(fragments), out_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def fetch_freed_data(out_dir: Path, *, timeout: int = 60, force: bool = False) -> None:
    """Fetch FREED Docker data files (fragment library and alert CSV).

    Args:
        out_dir: Destination directory for data files.
        timeout: HTTP request timeout in seconds.
        force: If True, overwrite existing files.
    """
    zinc_crem = out_dir / "zinc_crem.json"
    alerts = out_dir / "alert_collections.csv"

    if zinc_crem.exists() and alerts.exists() and not force:
        logger.info("Files already exist, skip: {}", out_dir)
        return

    if force:
        zinc_crem.unlink(missing_ok=True)
        alerts.unlink(missing_ok=True)

    _build_zinc_crem(zinc_crem, timeout=timeout)
    download_file(ALERT_COLLECTIONS_URL, alerts, timeout=timeout)
    _validate_alert_csv(alerts)
    logger.info("FREED data files ready: {}", out_dir)


def build_freedpp_image(
    project_root: Path,
    *,
    image: str = "dualdocking/freedpp:latest",
    force: bool = False,
    dry_run: bool = False,
    timeout: int = 3600,
) -> bool:
    """Build the FREED++ Docker image.

    Args:
        project_root: Repository root (used as Docker build context).
        image: Docker image tag to build.
        force: If True, rebuild even when the image already exists.
        dry_run: If True, log the command without executing.
        timeout: Maximum seconds to wait for the build.

    Returns:
        True if the image exists or was built successfully.
    """
    if not force and docker_image_exists(image):
        logger.info("[docker-build] Image already exists: {}", image)
        return True

    cmd = [
        DOCKER_BIN,
        "build",
        "-f",
        str(project_root / "docker" / "freedpp" / "Dockerfile"),
        "-t",
        image,
        str(project_root),
    ]

    logger.info("[docker-build] RUN: {}", " ".join(cmd))
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
        logger.error("[docker-build] TIMEOUT after {}s", timeout)
        return False

    if result.returncode != 0:
        tail = ((result.stdout or "") + (result.stderr or ""))[-3000:]
        logger.error("[docker-build] FAILED (rc={}): {}", result.returncode, tail)
        return False

    return True
