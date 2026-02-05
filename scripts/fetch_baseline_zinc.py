"""Download baseline ZINC tar.xz dataset for local experiments."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import click
import requests
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "baseline" / "zinc-druglike-cano.tar.xz"
DEFAULT_URL_ENV = "ZINC_BASELINE_URL"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, output: Path, timeout: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_suffix(output.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    logger.info("GET {} -> {}", url, output)
    with requests.get(url, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)

    tmp_path.replace(output)
    logger.info("Downloaded {:.2f} GiB", output.stat().st_size / (1024**3))


@click.command()
@click.option("--url", default=None, help=f"Direct URL for zinc-druglike-cano.tar.xz (or ${DEFAULT_URL_ENV})")
@click.option("--output", default=str(DEFAULT_OUTPUT), show_default=True)
@click.option("--timeout", default=120, type=int, show_default=True)
@click.option("--overwrite", is_flag=True, default=False)
def main(url: str | None, output: str, timeout: int, overwrite: bool) -> None:
    """Fetch baseline archive and write .sha256 next to it."""
    output_path = Path(output)
    sha_path = output_path.with_suffix(output_path.suffix + ".sha256")

    needs_download = not output_path.exists() or overwrite
    if not needs_download:
        logger.info("File already exists, skip download: {}", output_path)
    else:
        resolved_url = url or os.environ.get(DEFAULT_URL_ENV)
        if not resolved_url:
            raise click.UsageError(
                f"Baseline URL is required: pass --url or export {DEFAULT_URL_ENV}=<https://.../zinc-druglike-cano.tar.xz>"
            )
        _download(resolved_url, output_path, timeout=timeout)

    checksum = _sha256(output_path)
    sha_path.write_text(checksum + "\n", encoding="utf-8")
    logger.success("SHA256 {} -> {}", checksum, sha_path)


if __name__ == "__main__":
    main()
