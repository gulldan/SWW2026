"""Download GNINA static binary for cross-validation scoring (Step 6).

GNINA provides CNN-based docking/scoring as a third independent scorer
alongside Boltz-2 and UniDock.

Writes:
    - output/bin/gnina  (Linux x86_64 static binary)

Usage::

    uv run python scripts/s0_fetch/fetch_gnina.py
    uv run python scripts/s0_fetch/fetch_gnina.py --version 1.1
"""

from __future__ import annotations

import platform
import stat
from pathlib import Path

import click
from loguru import logger

from dualdocking.data.http_utils import download_file

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT: Path = PROJECT_ROOT / "output" / "bin" / "gnina"

# GNINA releases on GitHub (Linux x86_64 static build)
_GNINA_RELEASE_URL: str = "https://github.com/gnina/gnina/releases/download/v{version}/gnina"
_DEFAULT_VERSION: str = "1.1"


@click.command()
@click.option("--version", "gnina_version", default=_DEFAULT_VERSION, show_default=True, help="GNINA release version")
@click.option("--output", default=str(DEFAULT_OUTPUT), show_default=True, help="Local path for the binary")
@click.option("--force", is_flag=True, default=False, help="Overwrite existing binary")
def main(gnina_version: str, output: str, force: bool) -> None:
    """Fetch the GNINA static binary from GitHub releases.

    Downloads the Linux x86_64 static binary and makes it executable.
    On non-Linux platforms, the binary is still downloaded (for use
    on remote GPU servers) but a warning is emitted.

    Args:
        gnina_version: GNINA release tag (e.g. ``"1.1"``).
        output: Local destination path.
        force: If True, re-download even if the file exists.
    """
    output_path = Path(output)

    if output_path.exists() and not force:
        logger.info("GNINA binary already exists: {}", output_path)
        return

    if platform.system() != "Linux":
        logger.warning("GNINA static binary is Linux x86_64 only. Downloading anyway for deployment to GPU server.")

    url = _GNINA_RELEASE_URL.format(version=gnina_version)
    sha = download_file(url, output_path, timeout=120)
    logger.info("GNINA SHA-256: {}", sha)

    # Make executable
    output_path.chmod(output_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    logger.success("GNINA v{} ready: {}", gnina_version, output_path)


if __name__ == "__main__":
    main()
