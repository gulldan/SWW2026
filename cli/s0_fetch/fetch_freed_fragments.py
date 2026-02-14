"""Download FREED data dependencies: fragment library and alert filters.

Thin CLI wrapper -- core logic lives in :mod:`dualdocking.data.fetch.docker`.

Writes:
    - docker/freedpp/data/zinc_crem.json
    - docker/freedpp/data/alert_collections.csv
"""

from __future__ import annotations

from pathlib import Path

import click

from dualdocking.data.fetch.docker import fetch_freed_data

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_DIR: Path = PROJECT_ROOT / "docker" / "freedpp" / "data"


@click.command()
@click.option("--out-dir", default=str(DEFAULT_OUT_DIR), show_default=True)
@click.option("--timeout", default=60, type=int, show_default=True)
@click.option("--force", is_flag=True, default=False, help="Overwrite existing files")
def main(out_dir: str, timeout: int, force: bool) -> None:
    """Fetch FREED Docker data files used by docker/freedpp/Dockerfile.

    Downloads the CReM fragment library and structural alert CSV, writing
    them into the specified output directory.
    """
    fetch_freed_data(Path(out_dir), timeout=timeout, force=force)


if __name__ == "__main__":
    main()
