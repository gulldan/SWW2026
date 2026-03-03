"""Score distribution plotting for dual-target experiments.

Generates histograms, ECDF curves, and boxplots comparing score distributions
across generation modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.patches import Patch


@dataclass(frozen=True)
class PlotConfig:
    """Configuration for score distribution plots.

    Attributes:
        data_path: Path to aggregated results Parquet file.
        output_dir: Directory for output PNG figures.
        modes: List of generation modes to compare.
        objective_columns: Optional list of (column_name, label) tuples for
            additional objective scores to plot.
    """

    data_path: Path
    output_dir: Path
    modes: list[str]
    objective_columns: list[tuple[str, str]] | None = None


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
    score_cols: list[tuple[str, str]] = [(col_a, "Score A"), (col_b, "Score B")]
    if objective_columns:
        score_cols.extend((c, lbl) for c, lbl in objective_columns if c in df_pair.columns)
    elif "combined_min" in df_pair.columns:
        score_cols.append(("combined_min", "Combined (worst-case)"))
    for col, label in score_cols:
        if col not in df_pair.columns:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        for mode in modes:
            subset = df_pair[df_pair["mode"] == mode][col].dropna()
            axes[0].hist(subset, bins=40, alpha=0.6, label=mode, density=True)
        axes[0].set_xlabel(label)
        axes[0].set_ylabel("Density")
        axes[0].legend()
        axes[0].set_title(f"{pair_id} - {label} distribution")

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
        logger.info("Saved: {}", fig_dir / f"{pair_id}_{safe_col}_modes.png")

    # Boxplot (matplotlib to avoid seaborn deprecation warning)
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
    ax.set_title(f"{pair_id} - score comparison")
    ax.set_ylabel("score")

    legend_handles = [Patch(facecolor=palette[i], label=modes[i], alpha=0.6) for i in range(len(modes))]
    ax.legend(handles=legend_handles, title="mode")

    fig.tight_layout()
    fig.savefig(fig_dir / f"{pair_id}_boxplot.png", dpi=150)
    plt.close(fig)


def plot_all_pairs(config: PlotConfig) -> Path:
    """Generate comparison plots for all pairs in the dataset.

    Reads the aggregated Parquet file and produces per-pair figures comparing
    score distributions across generation modes.

    Args:
        config: Plot configuration with input/output paths and modes.

    Returns:
        Path to output directory containing generated plots.

    Raises:
        FileNotFoundError: If input data file does not exist.
    """
    if not config.data_path.exists():
        msg = f"Input file not found: {config.data_path}. Run aggregation first."
        raise FileNotFoundError(msg)

    df = pd.read_parquet(config.data_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for pair_id in df["pair"].unique():
        logger.info("Generating plots for pair: {}", pair_id)
        df_pair = df[df["pair"] == pair_id]
        plot_score_comparison(
            df_pair,
            pair_id,
            config.output_dir,
            config.modes,
            objective_columns=config.objective_columns,
        )

    logger.info("All plots saved to {}", config.output_dir)
    return config.output_dir
