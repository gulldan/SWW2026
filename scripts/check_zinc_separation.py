"""Check that baseline and calibration ZINC sets have zero overlap."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from loguru import logger

from dualdocking.chem_sanitize import canonicalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_smiles(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"SMILES file not found: {path}")
    smiles: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        smiles.append(line.split()[0])
    return smiles


def _canonical_set(smiles: list[str]) -> set[str]:
    out: set[str] = set()
    for smi in smiles:
        canon = canonicalize(smi)
        if canon:
            out.add(canon)
    return out


@click.command()
@click.option("--baseline", default="data/baseline/zinc_druglike_baseline_1000_seed42.smi")
@click.option("--calibration", default="data/calibration/zinc_druglike_calibration_1000_v2024.smi")
def main(baseline: str, calibration: str) -> None:
    baseline_path = PROJECT_ROOT / baseline
    calibration_path = PROJECT_ROOT / calibration

    base_smiles = _load_smiles(baseline_path)
    cal_smiles = _load_smiles(calibration_path)

    base_set = _canonical_set(base_smiles)
    cal_set = _canonical_set(cal_smiles)

    overlap = base_set.intersection(cal_set)
    if overlap:
        logger.error("Overlap detected: {} molecules", len(overlap))
        sys.exit(1)

    logger.info("No overlap between baseline and calibration sets")


if __name__ == "__main__":
    main()
