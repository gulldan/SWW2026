"""Analyze correlation between proxy docking scores and Boltz-2."""

from __future__ import annotations

import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

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


def _load_scores_csv(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    if "smiles" not in df.columns or "score" not in df.columns:
        raise ValueError(f"Expected columns smiles, score in {path}")
    return dict(zip(df["smiles"].astype(str), df["score"].astype(float), strict=False))


@click.command()
@click.option("--smiles", "smiles_path", default="data/baseline/zinc_druglike_baseline_1000_seed42.smi")
@click.option("--proxy-scores", default="")
@click.option("--boltz-scores", default="")
@click.option("--target", default="")
@click.option("--n", default=100, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--output", default="report/proxy_correlation.json")
@click.option("--figure", default="report/figures/proxy_vs_boltz.png")
def main(
    smiles_path: str,
    proxy_scores: str,
    boltz_scores: str,
    target: str,
    n: int,
    seed: int,
    output: str,
    figure: str,
) -> None:
    smiles = _load_smiles(PROJECT_ROOT / smiles_path)
    rng = np.random.default_rng(seed)
    n = min(n, len(smiles))
    sample = [smiles[i] for i in rng.choice(len(smiles), size=n, replace=False)]

    if not proxy_scores:
        raise FileNotFoundError("Provide --proxy-scores CSV (columns: smiles, score)")
    proxy_map = _load_scores_csv(PROJECT_ROOT / proxy_scores)

    if boltz_scores:
        boltz_map = _load_scores_csv(PROJECT_ROOT / boltz_scores)
    else:
        if not target:
            raise ValueError("Provide --target to compute Boltz-2 scores")
        from dualdocking.scorer_boltz2 import score_batch

        boltz_values = score_batch(target, sample)
        boltz_map = dict(zip(sample, boltz_values, strict=False))

    xs: list[float] = []
    ys: list[float] = []
    for smi in sample:
        if smi not in proxy_map or smi not in boltz_map:
            continue
        xs.append(float(proxy_map[smi]))
        ys.append(float(boltz_map[smi]))

    if len(xs) == 0:
        raise ValueError("No overlapping scores between proxy and Boltz")

    rho, _ = spearmanr(xs, ys)

    fig_path = PROJECT_ROOT / figure
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.scatter(xs, ys, s=12, alpha=0.7)
    plt.title(f"Proxy vs Boltz-2 (Spearman rho={rho:.3f})")
    plt.xlabel("Proxy score")
    plt.ylabel("Boltz-2 score")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)

    out = {"n": len(xs), "rho": float(rho)}
    out_path = PROJECT_ROOT / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    logger.info("Proxy correlation rho={:.3f} -> {}", rho, out_path)


if __name__ == "__main__":
    main()
