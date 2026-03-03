"""Fetch and prepare calibration SMILES sets based on configs/calibration.yaml.

Thin CLI wrapper -- core logic lives in :mod:`dualdocking.data.fetch.smiles`.
"""

from __future__ import annotations

from pathlib import Path

import click
import yaml

from dualdocking.data.fetch.smiles import fetch_calibration_smiles

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


@click.command()
@click.option("--config", default="configs/calibration.yaml")
@click.option("--only-selected", is_flag=True, default=False)
@click.option("--force", is_flag=True, default=False)
def main(config: str, only_selected: bool, force: bool) -> None:
    """Fetch calibration SMILES from configured sources.

    Reads ``configs/calibration.yaml``, downloads and reservoir-samples
    SMILES datasets, writes ``.smi`` files to the output directory, and
    copies the selected dataset to the canonical output path.
    """
    cfg_path = PROJECT_ROOT / config
    cfg = yaml.safe_load(cfg_path.read_text())

    selected = cfg.get("selected")
    output_path = PROJECT_ROOT / cfg.get("output_path")
    sample_n = int(cfg.get("sample", {}).get("n", 1000))
    seed = int(cfg.get("sample", {}).get("seed", 137))
    download_all = bool(cfg.get("options", {}).get("download_all", True))
    do_canonicalize = bool(cfg.get("options", {}).get("canonicalize", False))

    sources = cfg.get("sources", {})

    if only_selected:
        download_all = False

    out_dir = PROJECT_ROOT / "output" / "data" / "calibration"

    fetch_calibration_smiles(
        sources,
        selected,
        output_dir=out_dir,
        output_path=output_path,
        sample_n=sample_n,
        seed=seed,
        download_all=download_all,
        do_canonicalize=do_canonicalize,
        force=force,
    )


if __name__ == "__main__":
    main()
