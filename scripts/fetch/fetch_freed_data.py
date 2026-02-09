"""Download FREED data dependencies: fragment library + alert filters.

Writes:
- docker/freedpp/data/zinc_crem.json
- docker/freedpp/data/alert_collections.csv
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import click
import requests
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "docker" / "freedpp" / "data"

# AITRICS/FREED motifs text file (line-separated CReM fragments)
MOTIFS_URL = "https://raw.githubusercontent.com/AITRICS/FREED/main/gym_molecule/dataset/motifs_zinc_random_92.txt"

# Pat Walters rd_filters merged alert collections (PAINS / SureChEMBL / Glaxo)
ALERT_COLLECTIONS_URL = "https://raw.githubusercontent.com/PatWalters/rd_filters/master/rd_filters/data/alert_collection.csv"


def _download(url: str, dest: Path, timeout: int) -> None:
    logger.info("GET {} -> {}", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    logger.info("Saved {:.1f} KiB", dest.stat().st_size / 1024)


def _validate_alert_csv(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    if not header:
        raise RuntimeError(f"Empty alert collections CSV: {path}")


def _build_zinc_crem(out_path: Path, timeout: int) -> None:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _download(MOTIFS_URL, tmp_path, timeout=timeout)
        fragments = [line.strip() for line in tmp_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not fragments:
            raise RuntimeError("No fragments parsed from motifs file")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(fragments, indent=2), encoding="utf-8")
        logger.info("Wrote {} fragments -> {}", len(fragments), out_path)
    finally:
        tmp_path.unlink(missing_ok=True)


@click.command()
@click.option("--out-dir", default=str(DEFAULT_OUT_DIR), show_default=True)
@click.option("--timeout", default=60, type=int, show_default=True)
@click.option("--force", is_flag=True, default=False, help="Overwrite existing files")
def main(out_dir: str, timeout: int, force: bool) -> None:
    """Fetch FREED Docker data files used by docker/freedpp/Dockerfile."""
    dest_dir = Path(out_dir)
    zinc_crem = dest_dir / "zinc_crem.json"
    alerts = dest_dir / "alert_collections.csv"

    if zinc_crem.exists() and alerts.exists() and not force:
        logger.info("Files already exist, skip: {}", dest_dir)
        return

    if force:
        zinc_crem.unlink(missing_ok=True)
        alerts.unlink(missing_ok=True)

    _build_zinc_crem(zinc_crem, timeout=timeout)
    _download(ALERT_COLLECTIONS_URL, alerts, timeout=timeout)
    _validate_alert_csv(alerts)
    logger.success("FREED data files ready: {}", dest_dir)


if __name__ == "__main__":
    main()
