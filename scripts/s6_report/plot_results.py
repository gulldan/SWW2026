"""Generate comparison plots for dual-target experiment modes.

Reads ``report/data/all_results.parquet`` and produces per-pair histograms,
ECDF curves, and boxplots comparing score distributions across generation
modes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def plot_score_comparison(
    df_pair: pd.DataFrame,
    pair_id: str,
    fig_dir: Path,
    modes: list[str],
    objective_columns: list[tuple[str, str]] | None = None,
) -> None:
    """Plot score distribution comparisons (histograms, ECDFs, boxplot) for a pair.

    Args:
        df_pair: DataFrame filtered to a single pair, with a ``mode`` column.
        pair_id: Pair identifier used in titles and filenames.
        fig_dir: Output directory for PNG figures.
        modes: List of generation modes to compare.
        objective_columns: Optional list of ``(column_name, label)`` tuples
            for additional objective scores to plot.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    col_a = "affinity_a" if "affinity_a" in df_pair.columns else "score_a"
    col_b = "affinity_b" if "affinity_b" in df_pair.columns else "score_b"
    score_cols: list[tuple[str, str]] = [(col_a, "Скор A"), (col_b, "Скор B")]
    if objective_columns:
        score_cols.extend((c, lbl) for c, lbl in objective_columns if c in df_pair.columns)
    elif "combined_min" in df_pair.columns:
        score_cols.append(("combined_min", "Комбинированный (worst-case)"))
    for col, label in score_cols:
        if col not in df_pair.columns:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Гистограмма
        for mode in modes:
            subset = df_pair[df_pair["mode"] == mode][col].dropna()
            axes[0].hist(subset, bins=40, alpha=0.6, label=mode, density=True)
        axes[0].set_xlabel(label)
        axes[0].set_ylabel("Плотность")
        axes[0].legend()
        axes[0].set_title(f"{pair_id} - распределение {label}")

        # ECDF
        for mode in modes:
            subset = np.sort(df_pair[df_pair["mode"] == mode][col].dropna())
            ecdf = np.arange(1, len(subset) + 1) / len(subset)
            axes[1].step(subset, ecdf, label=mode)
        axes[1].set_xlabel(label)
        axes[1].set_ylabel("ECDF")
        axes[1].legend()
        axes[1].set_title(f"{pair_id} - ECDF {label}")

        fig.tight_layout()
        safe_col = col.replace("_", "")
        fig.savefig(fig_dir / f"{pair_id}_{safe_col}_modes.png", dpi=150)
        plt.close(fig)
        logger.info("Сохранено: {}", fig_dir / f"{pair_id}_{safe_col}_modes.png")

    # Боксплот (matplotlib, чтобы избежать deprecation warning seaborn)
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = [col_a, col_b] + [c for c, _ in score_cols if c not in (col_a, col_b) and c in df_pair.columns]
    data: list[np.ndarray] = []
    positions: list[float] = []
    group_width = len(modes) + 1
    tick_positions: list[float] = []
    tick_labels: list[str] = []

    for i, metric in enumerate(metrics):
        base = i * group_width
        tick_positions.append(base + (len(modes) - 1) / 2)
        tick_labels.append(metric)
        for j, mode in enumerate(modes):
            vals = df_pair[df_pair["mode"] == mode][metric].dropna().to_numpy()
            data.append(vals)
            positions.append(base + j)

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
    )
    palette = plt.cm.tab10.colors
    for idx, patch in enumerate(box["boxes"]):
        mode_idx = idx % len(modes)
        patch.set_facecolor(palette[mode_idx])
        patch.set_alpha(0.6)

    ax.set_xticks(tick_positions, tick_labels)
    ax.set_title(f"{pair_id} - сравнение скоров")
    ax.set_ylabel("score")

    legend_handles = [Patch(facecolor=palette[i], label=modes[i], alpha=0.6) for i in range(len(modes))]
    ax.legend(handles=legend_handles, title="mode")

    fig.tight_layout()
    fig.savefig(fig_dir / f"{pair_id}_boxplot.png", dpi=150)
    plt.close(fig)


@click.command()
@click.option("--input", "input_path", default="report/data/all_results.parquet")
@click.option("--output-dir", default="report/figures")
def main(input_path: str, output_dir: str) -> None:
    """Generate comparison plots for all pairs.

    Reads the aggregated Parquet file and produces per-pair figures
    comparing score distributions across generation modes.

    Args:
        input_path: Path to the aggregated results Parquet file.
        output_dir: Output directory for figure PNGs.
    """
    in_path = PROJECT_ROOT / input_path
    if not in_path.exists():
        logger.error("Входной файл не найден: {}. Сначала запустите агрегацию.", in_path)
        sys.exit(1)

    df = pd.read_parquet(in_path)
    fig_dir = PROJECT_ROOT / output_dir

    from dualdocking.config import load_experiment
    from dualdocking.scoring.objectives import parse_objectives_config

    try:
        exp_cfg = load_experiment(PROJECT_ROOT / "configs")
        modes = exp_cfg.get("modes", ["single_a", "single_b", "double"])
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Не удалось загрузить experiment.yaml: {}. Использую дефолтные режимы.", exc)
        exp_cfg = {}
        modes = ["single_a", "single_b", "double"]

    objectives = parse_objectives_config(exp_cfg)
    objective_columns = [(obj.column_name, obj.column_name) for obj in objectives]

    for pair_id in df["pair"].unique():
        logger.info("Построение графиков для пары: {}", pair_id)
        df_pair = df[df["pair"] == pair_id]
        plot_score_comparison(df_pair, pair_id, fig_dir, modes, objective_columns=objective_columns)

    logger.info("Все графики сохранены в {}", fig_dir)


if __name__ == "__main__":
    main()
