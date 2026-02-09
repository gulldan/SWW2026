"""Text summary report for dual-target experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _fmt(val: float | str, decimals: int = 3) -> str:
    try:
        return f"{float(val):.{decimals}f}"
    except Exception:
        return str(val)


def _score_distribution_block(scores: pd.Series, label: str) -> str:
    if scores.empty or scores.isna().all():
        return f"  {label}: нет данных\n"
    q1, median, q3 = np.nanpercentile(scores, [25, 50, 75])
    return (
        f"  {label}:\n"
        f"    median={_fmt(median)}  min={_fmt(scores.min())}  max={_fmt(scores.max())}  "
        f"IQR=[{_fmt(q1)}, {_fmt(q3)}]\n"
    )


def _top_molecules_block(df: pd.DataFrame, name_a: str, name_b: str, n: int = 3, primary_col: str = "combined_min") -> str:
    if df.empty:
        return "  Нет молекул для топа\n"
    sort_col = primary_col if primary_col in df.columns else "combined_min"
    top = df.nsmallest(n, sort_col)
    lines = [f"  Топ-{n} молекул (по {sort_col}):"]
    for i, (_, row) in enumerate(top.iterrows(), 1):
        smiles = row.get("smiles") or row.get("smiles_canonical") or ""
        if len(smiles) > 60:
            smiles = smiles[:57] + "..."
        lines.append(
            f"    {i}. {smiles}\n"
            f"       {name_a}={_fmt(row.get('score_a', row.get('affinity_a', float('nan'))))}  "
            f"{name_b}={_fmt(row.get('score_b', row.get('affinity_b', float('nan'))))}  "
            f"{sort_col}={_fmt(row.get(sort_col, float('nan')))}"
        )
    return "\n".join(lines) + "\n"


def _build_report(
    df: pd.DataFrame,
    metrics: dict,
    stats: dict,
    pairs_cfg: dict,
    thresholds_cfg: dict,
    modes: list[str],
    objective_columns: list[str] | None = None,
    primary_col: str = "combined_min",
) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("DUAL-TARGET DOCKING: СВОДНЫЙ ОТЧЁТ")
    lines.append("=" * 72)
    lines.append("")

    pairs = pairs_cfg.get("pairs", {})
    thresholds = thresholds_cfg.get("thresholds", {})

    for pair_id in df["pair"].unique():
        pair_info = pairs.get(pair_id, {})
        name_a = pair_info.get("target_a", {}).get("id", "A")
        name_b = pair_info.get("target_b", {}).get("id", "B")
        th_a = thresholds.get(pair_id, {}).get("target_a", 0.0)
        th_b = thresholds.get(pair_id, {}).get("target_b", 0.0)

        lines.append("-" * 72)
        lines.append(f"Пара: {pair_id}")
        desc = pair_info.get("description", "")
        if desc:
            lines.append(f"  {desc}")
        lines.append(f"  Пороги: {name_a} <= {th_a}, {name_b} <= {th_b}")
        lines.append("")

        df_pair = df[df["pair"] == pair_id]
        col_a = "affinity_a" if "affinity_a" in df_pair.columns else "score_a"
        col_b = "affinity_b" if "affinity_b" in df_pair.columns else "score_b"

        for mode in modes:
            df_mode = df_pair[df_pair["mode"] == mode]
            n = len(df_mode)
            lines.append(f"  [{mode.upper()}] N={n}")
            if df_mode.empty:
                lines.append("    нет данных\n")
                continue

            lines.append(_score_distribution_block(df_mode[col_a], name_a))
            lines.append(_score_distribution_block(df_mode[col_b], name_b))
            lines.extend(
                _score_distribution_block(df_mode[obj_col], obj_col)
                for obj_col in (objective_columns or ["combined_min"])
                if obj_col in df_mode.columns
            )

            pair_metrics = metrics.get(pair_id, {}).get(mode, {})
            if pair_metrics:
                hr_a = pair_metrics.get("hr_a_mean")
                hr_b = pair_metrics.get("hr_b_mean")
                hr_both = pair_metrics.get("hr_both_mean")
                balance = pair_metrics.get("balance_mean")
                lines.append(f"  Метрики: HR_A={_fmt(hr_a)}  HR_B={_fmt(hr_b)}  HR_both={_fmt(hr_both)}  balance={_fmt(balance)}")
                lines.append("")

            lines.append(_top_molecules_block(df_mode, name_a, name_b, primary_col=primary_col))

        pair_stats = stats.get(pair_id, {})
        if pair_stats:
            lines.append("  Статистические тесты:")
            for test_id, entry in pair_stats.items():
                p_adj = entry.get("p_value_adjusted", float("nan"))
                diff = entry.get("observed_diff", float("nan"))
                cliffs = entry.get("cliffs_delta", float("nan"))
                sig = "*" if entry.get("significant") else "ns"
                lines.append(
                    f"    {test_id}: metric={entry.get('metric')} diff={_fmt(diff)} p_adj={_fmt(p_adj, 4)} "
                    f"cliffs={_fmt(cliffs)} [{sig}]"
                )
            lines.append("")

    lines.append("")
    return "\n".join(lines)


@click.command()
@click.option("--input", "input_path", default="report/data/all_results.parquet", help="Parquet с результатами")
@click.option("--metrics", "metrics_path", default="report/metrics.json", help="Метрики")
@click.option("--stats", "stats_path", default="report/stats.json", help="Статистические тесты")
@click.option("--config", default="configs/pairs.yaml", help="Конфиг пар мишеней")
@click.option("--thresholds", default="configs/metrics.yaml", help="Пороги хитов")
@click.option("--output", default="report/summary.txt", help="Выходной файл")
def main(
    input_path: str,
    metrics_path: str,
    stats_path: str,
    config: str,
    thresholds: str,
    output: str,
) -> None:
    in_path = PROJECT_ROOT / input_path
    if not in_path.exists():
        logger.error("Входной файл не найден: {}. Сначала запустите агрегацию.", in_path)
        sys.exit(1)

    df = pd.read_parquet(in_path)

    metrics_file = PROJECT_ROOT / metrics_path
    stats_file = PROJECT_ROOT / stats_path
    metrics = _load_json(metrics_file) if metrics_file.exists() else {}
    stats = _load_json(stats_file) if stats_file.exists() else {}

    pairs_cfg = _load_yaml(PROJECT_ROOT / config)
    thresholds_cfg = _load_yaml(PROJECT_ROOT / thresholds)

    from dualdocking.config import load_experiment
    from dualdocking.scoring.objectives import parse_objectives_config

    exp_cfg = load_experiment(PROJECT_ROOT / "configs")
    modes = exp_cfg.get("modes", ["single_a", "single_b", "double"])
    objectives = parse_objectives_config(exp_cfg)
    obj_columns = [obj.column_name for obj in objectives]
    primary = next((obj.column_name for obj in objectives if obj.is_primary), "combined_min")

    report = _build_report(
        df,
        metrics,
        stats,
        pairs_cfg,
        thresholds_cfg,
        modes,
        objective_columns=obj_columns,
        primary_col=primary,
    )

    out_path = PROJECT_ROOT / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    logger.info("Отчёт сохранён: {}", out_path)


if __name__ == "__main__":
    main()
