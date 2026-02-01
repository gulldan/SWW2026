"""Скоринг сгенерированных молекул по обеим мишеням пары.

Читает молекулы из results/<pair>/<mode>/generated.csv, оценивает по
обеим мишеням, записывает scores_A.csv, scores_B.csv, scored_joined.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@click.command()
@click.option("--pair", required=True, help="ID пары из configs/pairs.yaml")
@click.option("--mode", type=click.Choice(["single", "double"]), required=True)
@click.option("--config", default="configs/pairs.yaml")
def main(pair: str, mode: str, config: str) -> None:
    """Оценить молекулы по обеим мишеням."""
    from dualdocking.config import load_pairs
    from dualdocking.objectives import ObjectiveKind, compute_combined
    from dualdocking.scorer_boltz2 import score_batch

    pairs = load_pairs(PROJECT_ROOT / "configs")["pairs"]
    if pair not in pairs:
        logger.error("Пара '{}' не найдена", pair)
        sys.exit(1)

    pair_cfg = pairs[pair]
    result_dir = PROJECT_ROOT / "results" / pair / mode
    mol_path = result_dir / "generated.csv"

    if not mol_path.exists():
        logger.error("Молекулы не найдены: {}", mol_path)
        sys.exit(1)

    df = pd.read_csv(mol_path)
    smiles_list = df["smiles"].tolist()

    target_a_id = pair_cfg["target_a"]["id"]
    target_b_id = pair_cfg["target_b"]["id"]

    logger.info("Скоринг {} молекул по мишени {}", len(smiles_list), target_a_id)
    scores_a = score_batch(target_a_id, smiles_list)

    logger.info("Скоринг {} молекул по мишени {}", len(smiles_list), target_b_id)
    scores_b = score_batch(target_b_id, smiles_list)

    combined = compute_combined(scores_a, scores_b, ObjectiveKind.MIN, lower_is_better=True)

    # Сохранить отдельные скоры
    pd.DataFrame({"smiles": smiles_list, "score": scores_a}).to_csv(result_dir / "scores_A.csv", index=False)
    pd.DataFrame({"smiles": smiles_list, "score": scores_b}).to_csv(result_dir / "scores_B.csv", index=False)

    # Сохранить объединённую таблицу
    joined = pd.DataFrame(
        {
            "smiles": smiles_list,
            "score_a": scores_a,
            "score_b": scores_b,
            "combined_min": combined,
            "combined_harmonic": compute_combined(scores_a, scores_b, ObjectiveKind.HARMONIC, lower_is_better=True),
        }
    )
    joined = joined.sort_values("combined_min")
    joined.to_csv(result_dir / "scored_joined.csv", index=False)

    logger.info("Скоры сохранены в {}", result_dir)
    logger.info(
        "Статистика скоров - A: mean={:.2f} std={:.2f} | B: mean={:.2f} std={:.2f}",
        np.nanmean(scores_a),
        np.nanstd(scores_a),
        np.nanmean(scores_b),
        np.nanstd(scores_b),
    )


if __name__ == "__main__":
    main()
