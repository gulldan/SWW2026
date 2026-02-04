"""Build proxy score CSV (smiles, score) from calibration outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import click
from loguru import logger

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


def _load_scores(path: Path) -> list[float]:
    if not path.exists():
        raise FileNotFoundError(f"Scores file not found: {path}")
    scores: list[float] = []
    with path.open() as f:
        header = f.readline()
        if "score" not in header.lower():
            raise ValueError(f"Expected CSV with score column: {path}")
        for line in f:
            line = line.strip()
            if not line:
                continue
            scores.append(float(line.split(",")[0]))
    return scores


@click.command()
@click.option("--target", default="", help="Target ID (e.g., CDK2)")
@click.option("--ligands", default="data/calibration/ligands/ligands.smi")
@click.option("--scores", default="", help="Scores CSV (defaults to data/calibration/scores/<target>.csv)")
@click.option("--output", default="", help="Output CSV (defaults to data/calibration/proxy_scores/<target>.csv)")
def main(target: str, ligands: str, scores: str, output: str) -> None:
    if not scores:
        if not target:
            raise ValueError("Provide --target or --scores")
        scores = f"data/calibration/scores/{target}.csv"
    if not output:
        if not target:
            raise ValueError("Provide --target or --output")
        output = f"data/calibration/proxy_scores/{target}.csv"

    ligands_path = PROJECT_ROOT / ligands
    scores_path = PROJECT_ROOT / scores
    out_path = PROJECT_ROOT / output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    smiles = _load_smiles(ligands_path)
    sc = _load_scores(scores_path)

    if len(smiles) != len(sc):
        raise ValueError(f"Length mismatch: {len(smiles)} smiles vs {len(sc)} scores")

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "score"])
        for smi, score in zip(smiles, sc, strict=False):
            writer.writerow([smi, f"{score:.6f}"])

    logger.info("Proxy scores saved -> {} (n={})", out_path, len(smiles))


if __name__ == "__main__":
    main()
