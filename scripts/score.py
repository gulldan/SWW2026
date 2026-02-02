"""Скоринг сгенерированных молекул по обеим мишеням пары.

Читает молекулы из results/<pair>/<mode>/generated.csv, оценивает по
обеим мишеням, записывает scores_A.csv, scores_B.csv, scored_joined.csv.

Поддерживает чекпоинт: если scores_A.csv уже существует, скоринг мишени A
пропускается. При прерывании скоринга отдельных молекул кеш Boltz-2
(cache/{target_id}.json) сохраняется инкрементально.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@click.command()
@click.option("--pair", required=True, help="ID пары из configs/pairs.yaml")
@click.option("--mode", type=click.Choice(["single", "double"]), required=True)
@click.option("--config", default="configs/pairs.yaml")
@click.option("--force", is_flag=True, default=False, help="Пересчитать даже если scores уже есть")
def main(pair: str, mode: str, config: str, force: bool) -> None:
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

    scores_a_path = result_dir / "scores_A.csv"
    scores_b_path = result_dir / "scores_B.csv"

    # --- Скоринг мишени A (с чекпоинтом) ---
    targets = [
        (target_a_id, scores_a_path, "A"),
        (target_b_id, scores_b_path, "B"),
    ]

    all_scores: dict[str, np.ndarray] = {}

    progress = tqdm(targets, desc=f"Скоринг [{pair}/{mode}]", unit="target")
    for target_id, scores_path, label in progress:
        progress.set_postfix(target=target_id)

        if not force and scores_path.exists():
            logger.info("Чекпоинт: скоры мишени {} уже существуют ({}), пропускаем", target_id, scores_path)
            saved = pd.read_csv(scores_path)
            all_scores[label] = saved["score"].values
            continue

        logger.info("Скоринг {} молекул по мишени {}", len(smiles_list), target_id)
        scores = score_batch(target_id, smiles_list)
        all_scores[label] = scores

        # Сохранить промежуточный результат сразу (чекпоинт)
        pd.DataFrame({"smiles": smiles_list, "score": scores}).to_csv(scores_path, index=False)
        logger.info("Скоры мишени {} сохранены: {}", target_id, scores_path)

    scores_a = all_scores["A"]
    scores_b = all_scores["B"]

    combined = compute_combined(scores_a, scores_b, ObjectiveKind.MIN, lower_is_better=True)

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
