"""2D-визуализация молекул с аннотациями скоров.

Читает report/data/all_results.parquet, для каждой пары строит сетку молекул
(matplotlib subplots) с читаемыми подписями и цветовой маркировкой хитов.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from loguru import logger
from matplotlib.patches import FancyBboxPatch
from rdkit import Chem
from rdkit.Chem import Draw

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MAX_COLS = 5
CELL_W = 4.0  # inches per subplot
CELL_H = 4.5


def _load_target_names(config_path: Path) -> dict[str, tuple[str, str]]:
    """Вернуть {pair_id: (name_a, name_b)} из pairs.yaml."""
    cfg = yaml.safe_load(config_path.read_text())
    names: dict[str, tuple[str, str]] = {}
    for pair_id, pair_cfg in cfg.get("pairs", {}).items():
        name_a = pair_cfg["target_a"]["id"]
        name_b = pair_cfg["target_b"]["id"]
        names[pair_id] = (name_a, name_b)
    return names


def _load_thresholds(thresholds_path: Path) -> dict[str, tuple[float, float]]:
    """Вернуть {pair_id: (threshold_a, threshold_b)} из metrics.yaml."""
    cfg = yaml.safe_load(thresholds_path.read_text())
    result: dict[str, tuple[float, float]] = {}
    for pair_id, th in cfg.get("thresholds", {}).items():
        result[pair_id] = (float(th.get("target_a", 0.0)), float(th.get("target_b", 0.0)))
    return result


def _is_hit(score_a: float, score_b: float, th_a: float, th_b: float) -> bool:
    return score_a <= th_a and score_b <= th_b


def _render_grid(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    name_a: str,
    name_b: str,
    th_a: float,
    th_b: float,
) -> None:
    """Отрисовка сетки молекул через matplotlib subplots."""
    if df.empty:
        logger.warning("Нет данных для {}", title)
        return

    df = df.sort_values("combined_min", ascending=True).reset_index(drop=True)
    n = len(df)
    n_cols = min(MAX_COLS, n)
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

        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            ax.text(0.5, 0.5, "Invalid\nSMILES", ha="center", va="center", fontsize=10)
            ax.axis("off")
            continue

        img = Draw.MolToImage(mol, size=(400, 400))
        ax.imshow(img)
        ax.axis("off")

        hit = _is_hit(row["score_a"], row["score_b"], th_a, th_b)

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
        score_text = f"{name_a}: {row['score_a']:.2f}  |  {name_b}: {row['score_b']:.2f}\nmin: {row['combined_min']:.2f}"
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

    # Скрыть пустые ячейки
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Сохранено: {}", output_path)


@click.command()
@click.option("--input", "input_path", default="report/data/all_results.parquet", help="Parquet с результатами")
@click.option("--output-dir", default="report/figures", help="Каталог для PNG")
@click.option("--config", default="configs/pairs.yaml", help="Конфиг пар мишеней")
@click.option("--thresholds", default="configs/metrics.yaml", help="Пороги хитов")
@click.option("--top-n", default=5, type=int, help="Число топовых молекул для top_hits")
def main(input_path: str, output_dir: str, config: str, thresholds: str, top_n: int) -> None:
    """Отрисовка 2D-молекул с аннотациями скоров для каждой пары."""
    in_path = PROJECT_ROOT / input_path
    if not in_path.exists():
        logger.error("Входной файл не найден: {}. Сначала запустите агрегацию.", in_path)
        sys.exit(1)

    df = pd.read_parquet(in_path)
    fig_dir = PROJECT_ROOT / output_dir
    target_names = _load_target_names(PROJECT_ROOT / config)
    thresh = _load_thresholds(PROJECT_ROOT / thresholds)

    for pair_id in df["pair"].unique():
        name_a, name_b = target_names.get(pair_id, ("A", "B"))
        th_a, th_b = thresh.get(pair_id, (0.0, 0.0))
        df_pair = df[df["pair"] == pair_id]

        for mode in ("single", "double"):
            df_mode = df_pair[df_pair["mode"] == mode]
            if df_mode.empty:
                continue
            _render_grid(
                df_mode,
                fig_dir / f"{pair_id}_molecules_{mode}.png",
                f"{pair_id} — {mode} mode",
                name_a,
                name_b,
                th_a,
                th_b,
            )

        # Top hits: лучшие из обоих режимов
        df_single = df_pair[df_pair["mode"] == "single"].nsmallest(top_n, "combined_min")
        df_double = df_pair[df_pair["mode"] == "double"].nsmallest(top_n, "combined_min")
        df_top = (
            pd.concat(
                [
                    df_single.assign(label="single"),
                    df_double.assign(label="double"),
                ]
            )
            .sort_values("combined_min", ascending=True)
            .reset_index(drop=True)
        )

        if not df_top.empty:
            _render_grid(
                df_top,
                fig_dir / f"{pair_id}_molecules_top_hits.png",
                f"{pair_id} — top {top_n} hits (single + double)",
                name_a,
                name_b,
                th_a,
                th_b,
            )

    logger.info("Все молекулы отрисованы в {}", fig_dir)


if __name__ == "__main__":
    main()
