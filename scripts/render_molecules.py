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
    primary_col: str = "combined_min",
) -> None:
    """Отрисовка сетки молекул через matplotlib subplots."""
    if df.empty:
        logger.warning("Нет данных для {}", title)
        return

    sort_col = primary_col if primary_col in df.columns else "combined_min"
    df = df.sort_values(sort_col, ascending=True).reset_index(drop=True)
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
        hit = _is_hit(val_a, val_b, th_a, th_b)

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

    from dualdocking.config import load_experiment
    from dualdocking.objectives import parse_objectives_config

    try:
        exp_cfg = load_experiment(PROJECT_ROOT / "configs")
        modes = exp_cfg.get("modes", ["single_a", "single_b", "double"])
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Не удалось загрузить experiment.yaml: {}. Использую дефолтные режимы.", exc)
        exp_cfg = {}
        modes = ["single_a", "single_b", "double"]

    objectives = parse_objectives_config(exp_cfg)
    primary_col = next((obj.column_name for obj in objectives if obj.is_primary), "combined_min")
    if primary_col not in df.columns:
        logger.warning("Primary objective {} missing in data, falling back to combined_min", primary_col)
        primary_col = "combined_min"

    for pair_id in df["pair"].unique():
        name_a, name_b = target_names.get(pair_id, ("A", "B"))
        th_a, th_b = thresh.get(pair_id, (0.0, 0.0))
        df_pair = df[df["pair"] == pair_id]

        for mode in modes:
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
                primary_col=primary_col,
            )

        # Top hits: лучшие из всех режимов
        sort_col = primary_col if primary_col in df_pair.columns else "combined_min"
        frames = []
        for mode in modes:
            df_mode = df_pair[df_pair["mode"] == mode].nsmallest(top_n, sort_col)
            if not df_mode.empty:
                frames.append(df_mode.assign(label=mode))
        df_top = pd.concat(frames).sort_values(sort_col, ascending=True).reset_index(drop=True) if frames else pd.DataFrame()

        if not df_top.empty:
            _render_grid(
                df_top,
                fig_dir / f"{pair_id}_molecules_top_hits.png",
                f"{pair_id} — top {top_n} hits (all modes)",
                name_a,
                name_b,
                th_a,
                th_b,
                primary_col=primary_col,
            )

    logger.info("Все молекулы отрисованы в {}", fig_dir)


if __name__ == "__main__":
    main()
