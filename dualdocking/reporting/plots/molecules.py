"""2D molecule grid rendering with score annotations.

Renders grids of RDKit 2D molecule images for each pair/mode with
readable score labels and colour-coded hit markers.

Extracted from ``cli/s6_report/render_molecules.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from loguru import logger
from matplotlib.patches import FancyBboxPatch
from rdkit import Chem
from rdkit.Chem import Draw

MAX_COLS = 5
CELL_W = 4.0  # inches per subplot
CELL_H = 4.5


@dataclass(frozen=True)
class MoleculeRenderConfig:
    """Configuration for molecule grid rendering.

    Attributes:
        modes: Generation modes to render.
        primary_col: Column used to sort molecules (ascending = best first).
        top_n: Number of top molecules for the combined top-hits grid.
        dpi: Resolution for saved figures.
        max_cols: Maximum columns in the grid.
    """

    modes: list[str] = field(default_factory=lambda: ["single_a", "single_b", "double"])
    primary_col: str = "combined_min"
    top_n: int = 5
    dpi: int = 150
    max_cols: int = MAX_COLS


def load_target_names(config_path: Path) -> dict[str, tuple[str, str]]:
    """Load ``{pair_id: (name_a, name_b)}`` from ``pairs.yaml``.

    Args:
        config_path: Path to ``pairs.yaml``.

    Returns:
        Dict mapping pair IDs to ``(target_a_id, target_b_id)`` tuples.
    """
    cfg = yaml.safe_load(config_path.read_text())
    names: dict[str, tuple[str, str]] = {}
    for pair_id, pair_cfg in cfg.get("pairs", {}).items():
        name_a = pair_cfg["target_a"]["id"]
        name_b = pair_cfg["target_b"]["id"]
        names[pair_id] = (name_a, name_b)
    return names


def load_thresholds(thresholds_path: Path) -> dict[str, tuple[float, float]]:
    """Load ``{pair_id: (threshold_a, threshold_b)}`` from ``metrics.yaml``.

    Args:
        thresholds_path: Path to ``metrics.yaml``.

    Returns:
        Dict mapping pair IDs to hit-rate threshold tuples.
    """
    cfg = yaml.safe_load(thresholds_path.read_text())
    result: dict[str, tuple[float, float]] = {}
    for pair_id, th in cfg.get("thresholds", {}).items():
        result[pair_id] = (float(th.get("target_a", 0.0)), float(th.get("target_b", 0.0)))
    return result


def is_hit(score_a: float, score_b: float, th_a: float, th_b: float) -> bool:
    """Check if a molecule is a dual hit (both scores below thresholds).

    Args:
        score_a: Affinity score for target A.
        score_b: Affinity score for target B.
        th_a: Hit threshold for target A.
        th_b: Hit threshold for target B.

    Returns:
        ``True`` if both scores are at or below their thresholds.
    """
    return score_a <= th_a and score_b <= th_b


def render_molecule_grid(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    name_a: str,
    name_b: str,
    th_a: float,
    th_b: float,
    primary_col: str = "combined_min",
    *,
    dpi: int = 150,
    max_cols: int = MAX_COLS,
) -> None:
    """Render a grid of 2D molecule images with score annotations.

    Args:
        df: DataFrame with molecule SMILES and affinity scores.
        output_path: Path to save the output PNG.
        title: Figure title.
        name_a: Display name for target A.
        name_b: Display name for target B.
        th_a: Hit threshold for target A.
        th_b: Hit threshold for target B.
        primary_col: Column used to sort molecules (ascending).
        dpi: Resolution for saved figures.
        max_cols: Maximum columns in the grid.
    """
    if df.empty:
        logger.warning("No data for {}", title)
        return

    sort_col = primary_col if primary_col in df.columns else "combined_min"
    df = df.sort_values(sort_col, ascending=True).reset_index(drop=True)
    n = len(df)
    n_cols = min(max_cols, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(CELL_W * n_cols, CELL_H * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [list(axes)]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    else:
        axes = [list(row) for row in axes]

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.0)

    for idx, row in df.iterrows():
        r, c = divmod(int(idx), n_cols)
        ax = axes[r][c]

        smiles_val = row.get("smiles") or row.get("smiles_canonical") or ""
        mol = Chem.MolFromSmiles(smiles_val)
        if mol is None:
            ax.text(0.5, 0.5, "Invalid\nSMILES", ha="center", va="center", fontsize=10)
            ax.axis("off")
            continue

        img = Draw.MolToImage(mol, size=(400, 400))
        ax.imshow(img)
        ax.axis("off")

        val_a = row.get("affinity_a", row.get("score_a", float("nan")))
        val_b = row.get("affinity_b", row.get("score_b", float("nan")))
        hit = is_hit(val_a, val_b, th_a, th_b)

        if hit:
            rect = FancyBboxPatch(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                boxstyle="round,pad=0.02",
                facecolor="#d4edda",
                edgecolor="#28a745",
                linewidth=2,
                zorder=-1,
            )
            ax.add_patch(rect)

        ax.set_title(f"#{int(idx) + 1}", fontsize=11, fontweight="bold")
        primary_val = row.get(sort_col, float("nan"))
        score_text = f"{name_a}: {val_a:.2f}  |  {name_b}: {val_b:.2f}\n{sort_col}: {primary_val:.2f}"
        if hit:
            score_text += "  [HIT]"
        ax.text(
            0.5,
            -0.02,
            score_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            family="monospace",
        )

    # Hide empty cells
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: {}", output_path)


def render_all_pairs(
    df: pd.DataFrame,
    fig_dir: Path,
    target_names: dict[str, tuple[str, str]],
    thresholds: dict[str, tuple[float, float]],
    config: MoleculeRenderConfig | None = None,
) -> None:
    """Render molecule grids for all pairs in the DataFrame.

    Produces per-pair per-mode molecule grid PNGs and a combined
    top-hits grid.

    Args:
        df: DataFrame with ``pair`` and ``mode`` columns plus score/SMILES columns.
        fig_dir: Output directory for figure PNGs.
        target_names: Mapping of pair_id to ``(name_a, name_b)`` tuples.
        thresholds: Mapping of pair_id to ``(threshold_a, threshold_b)`` tuples.
        config: Render configuration. Uses defaults if ``None``.
    """
    if config is None:
        config = MoleculeRenderConfig()

    for pair_id in df["pair"].unique():
        name_a, name_b = target_names.get(pair_id, ("A", "B"))
        th_a, th_b = thresholds.get(pair_id, (0.0, 0.0))
        df_pair = df[df["pair"] == pair_id]

        for mode in config.modes:
            df_mode = df_pair[df_pair["mode"] == mode]
            if df_mode.empty:
                continue
            render_molecule_grid(
                df_mode,
                fig_dir / f"{pair_id}_molecules_{mode}.png",
                f"{pair_id} — {mode} mode",
                name_a,
                name_b,
                th_a,
                th_b,
                primary_col=config.primary_col,
                dpi=config.dpi,
                max_cols=config.max_cols,
            )

        # Top hits: best across all modes
        sort_col = config.primary_col if config.primary_col in df_pair.columns else "combined_min"
        frames = []
        for mode in config.modes:
            df_mode = df_pair[df_pair["mode"] == mode].nsmallest(config.top_n, sort_col)
            if not df_mode.empty:
                frames.append(df_mode.assign(label=mode))
        df_top = (
            pd.concat(frames).sort_values(sort_col, ascending=True).reset_index(drop=True) if frames else pd.DataFrame()
        )

        if not df_top.empty:
            render_molecule_grid(
                df_top,
                fig_dir / f"{pair_id}_molecules_top_hits.png",
                f"{pair_id} — top {config.top_n} hits (all modes)",
                name_a,
                name_b,
                th_a,
                th_b,
                primary_col=config.primary_col,
                dpi=config.dpi,
                max_cols=config.max_cols,
            )

    logger.info("All molecules rendered in {}", fig_dir)
