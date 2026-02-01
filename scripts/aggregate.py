"""Агрегация результатов по всем парам в единый датасет анализа.

Читает results/<pair>/<mode>/scored_joined.csv для каждой пары и режима,
объединяет и записывает report/data/all_results.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@click.command()
@click.option("--config", default="configs/pairs.yaml")
@click.option("--output", default="report/data/all_results.parquet")
def main(config: str, output: str) -> None:
    """Агрегировать результаты всех пар и режимов."""
    from dualdocking.config import load_pairs

    pairs = load_pairs(PROJECT_ROOT / "configs")["pairs"]
    frames: list[pd.DataFrame] = []

    for pair_id in pairs:
        for mode in ("single", "double"):
            path = PROJECT_ROOT / "results" / pair_id / mode / "scored_joined.csv"
            if not path.exists():
                logger.warning("Отсутствует: {}", path)
                continue
            df = pd.read_csv(path)
            df["pair"] = pair_id
            df["mode"] = mode
            frames.append(df)
            logger.info("Загружено {} строк из {}", len(df), path)

    if not frames:
        logger.error("Результаты не найдены. Сначала запустите генерацию и скоринг.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    out_path = PROJECT_ROOT / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    logger.info("Агрегировано {} строк -> {}", len(combined), out_path)


if __name__ == "__main__":
    main()
