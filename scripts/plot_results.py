"""Генерация графиков сравнения single vs double.

Читает report/data/all_results.parquet и строит графики для каждой пары.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def plot_score_comparison(df_pair: pd.DataFrame, pair_id: str, fig_dir: Path) -> None:
    """Построить графики сравнения распределений скоров для пары."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    for col, label in [("score_a", "Скор A"), ("score_b", "Скор B"), ("combined_min", "Комбинированный (worst-case)")]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Гистограмма
        for mode in ("single", "double"):
            subset = df_pair[df_pair["mode"] == mode][col].dropna()
            axes[0].hist(subset, bins=40, alpha=0.6, label=mode, density=True)
        axes[0].set_xlabel(label)
        axes[0].set_ylabel("Плотность")
        axes[0].legend()
        axes[0].set_title(f"{pair_id} - распределение {label}")

        # ECDF
        for mode in ("single", "double"):
            subset = np.sort(df_pair[df_pair["mode"] == mode][col].dropna())
            ecdf = np.arange(1, len(subset) + 1) / len(subset)
            axes[1].step(subset, ecdf, label=mode)
        axes[1].set_xlabel(label)
        axes[1].set_ylabel("ECDF")
        axes[1].legend()
        axes[1].set_title(f"{pair_id} - ECDF {label}")

        fig.tight_layout()
        safe_col = col.replace("_", "")
        fig.savefig(fig_dir / f"{pair_id}_{safe_col}_single_vs_double.png", dpi=150)
        plt.close(fig)
        logger.info("Сохранено: {}", fig_dir / f"{pair_id}_{safe_col}_single_vs_double.png")

    # Боксплот
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = df_pair.melt(
        id_vars=["mode"],
        value_vars=["score_a", "score_b", "combined_min"],
        var_name="metric",
        value_name="score",
    )
    sns.boxplot(data=plot_df, x="metric", y="score", hue="mode", ax=ax)
    ax.set_title(f"{pair_id} - сравнение скоров")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{pair_id}_boxplot.png", dpi=150)
    plt.close(fig)


@click.command()
@click.option("--input", "input_path", default="report/data/all_results.parquet")
@click.option("--output-dir", default="report/figures")
def main(input_path: str, output_dir: str) -> None:
    """Сгенерировать графики сравнения."""
    in_path = PROJECT_ROOT / input_path
    if not in_path.exists():
        logger.error("Входной файл не найден: {}. Сначала запустите агрегацию.", in_path)
        sys.exit(1)

    df = pd.read_parquet(in_path)
    fig_dir = PROJECT_ROOT / output_dir

    for pair_id in df["pair"].unique():
        logger.info("Построение графиков для пары: {}", pair_id)
        df_pair = df[df["pair"] == pair_id]
        plot_score_comparison(df_pair, pair_id, fig_dir)

    logger.info("Все графики сохранены в {}", fig_dir)


if __name__ == "__main__":
    main()
