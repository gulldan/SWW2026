"""Мини-аналитика: текстовый отчёт по результатам dual-target экспериментов.

Читает parquet, metrics.json, stats.json и конфиги, формирует
человекочитаемый текстовый отчёт с ключевыми выводами.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _fmt(val: float, decimals: int = 3) -> str:
    return f"{val:.{decimals}f}"


def _score_distribution_block(scores: pd.Series, label: str) -> str:
    """Статистика распределения скоров."""
    if scores.empty or scores.isna().all():
        return f"  {label}: нет данных\n"
    q1, median, q3 = np.nanpercentile(scores, [25, 50, 75])
    return (
        f"  {label}:\n"
        f"    median={_fmt(median)}  min={_fmt(scores.min())}  max={_fmt(scores.max())}  "
        f"IQR=[{_fmt(q1)}, {_fmt(q3)}]\n"
    )


def _top_molecules_block(df: pd.DataFrame, name_a: str, name_b: str, n: int = 3) -> str:
    """Топ-N молекул по combined_min."""
    top = df.nsmallest(n, "combined_min")
    lines = [f"  Топ-{n} молекул (по combined_min):"]
    for i, (_, row) in enumerate(top.iterrows(), 1):
        smiles = row["smiles"]
        if len(smiles) > 60:
            smiles = smiles[:57] + "..."
        lines.append(
            f"    {i}. {smiles}\n"
            f"       {name_a}={_fmt(row['score_a'])}  {name_b}={_fmt(row['score_b'])}  "
            f"min={_fmt(row['combined_min'])}"
        )
    return "\n".join(lines) + "\n"


def _build_report(
    df: pd.DataFrame,
    metrics: dict,
    stats: dict,
    pairs_cfg: dict,
    thresholds_cfg: dict,
) -> str:
    """Сформировать полный текстовый отчёт."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("DUAL-TARGET DOCKING: СВОДНЫЙ ОТЧЁТ")
    lines.append("=" * 72)
    lines.append("")

    pairs = pairs_cfg.get("pairs", {})
    thresholds = thresholds_cfg.get("thresholds", {})
    double_wins = 0
    total_pairs = 0

    for pair_id in df["pair"].unique():
        pair_info = pairs.get(pair_id, {})
        name_a = pair_info.get("target_a", {}).get("id", "A")
        name_b = pair_info.get("target_b", {}).get("id", "B")
        th_a = thresholds.get(pair_id, {}).get("target_a", 0.0)
        th_b = thresholds.get(pair_id, {}).get("target_b", 0.0)

        lines.append("-" * 72)
        desc = pair_info.get("description", "")
        lines.append(f"Пара: {pair_id}")
        if desc:
            lines.append(f"  {desc}")
        lines.append(f"  Пороги: {name_a} <= {th_a}, {name_b} <= {th_b}")
        lines.append("")

        df_pair = df[df["pair"] == pair_id]

        for mode in ("single", "double"):
            df_mode = df_pair[df_pair["mode"] == mode]
            n = len(df_mode)
            lines.append(f"  [{mode.upper()}] N={n}")
            if df_mode.empty:
                lines.append("    нет данных\n")
                continue

            lines.append(_score_distribution_block(df_mode["score_a"], name_a))
            lines.append(_score_distribution_block(df_mode["score_b"], name_b))
            lines.append(_score_distribution_block(df_mode["combined_min"], "combined_min"))

            # Hit rates из metrics.json
            pair_metrics = metrics.get(pair_id, {}).get(mode, {})
            if pair_metrics:
                hr_a = pair_metrics.get("hit_rate_a", "?")
                hr_b = pair_metrics.get("hit_rate_b", "?")
                hr_dual = pair_metrics.get("dual_hit_rate", "?")
                lines.append(f"  Hit rates: {name_a}={hr_a}  {name_b}={hr_b}  dual={hr_dual}")
                lines.append("")

            lines.append(_top_molecules_block(df_mode, name_a, name_b))

        # Статистическое сравнение
        pair_stats = stats.get(pair_id, {})
        if pair_stats:
            lines.append("  Статистическое сравнение (single vs double):")
            for col_key, col_label in [
                ("score_a", name_a),
                ("score_b", name_b),
                ("combined_min", "combined_min"),
            ]:
                st = pair_stats.get(col_key, {})
                if not st:
                    continue
                p_mw = st.get("mannwhitney_pvalue", float("nan"))
                p_ks = st.get("ks_pvalue", float("nan"))
                r_rb = st.get("rank_biserial_r", float("nan"))
                sig = "***" if p_mw < 0.001 else "**" if p_mw < 0.01 else "*" if p_mw < 0.05 else "ns"
                lines.append(f"    {col_label}: MW p={p_mw:.4g} ({sig})  KS p={p_ks:.4g}  r={r_rb}")
            lines.append("")

            # Вердикт
            cm_stats = pair_stats.get("combined_min", {})
            p_val = cm_stats.get("mannwhitney_pvalue", 1.0)
            total_pairs += 1
            single_median = df_pair[df_pair["mode"] == "single"]["combined_min"].median()
            double_median = df_pair[df_pair["mode"] == "double"]["combined_min"].median()
            if p_val < 0.05 and double_median < single_median:
                verdict = "Double ЗНАЧИМО лучше (p < 0.05, медиана ниже)"
                double_wins += 1
            elif p_val < 0.05:
                verdict = "Значимое различие, но double НЕ лучше"
            else:
                verdict = "Различие НЕ значимо (p >= 0.05)"
            lines.append(f"  Вердикт: {verdict}")
            lines.append("")

    # Общая оценка
    lines.append("=" * 72)
    lines.append("ОБЩАЯ ОЦЕНКА")
    lines.append("=" * 72)
    lines.append("")

    if total_pairs > 0:
        lines.append(f"Double значимо лучше в {double_wins}/{total_pairs} парах.")
    else:
        lines.append("Нет пар для сравнения.")

    # Проверка фармакологической значимости скоров
    all_combined = df["combined_min"].dropna()
    if not all_combined.empty:
        med = all_combined.median()
        frac_sub_zero = (all_combined <= 0.0).mean()
        lines.append("")
        lines.append(f"Медиана combined_min по всем молекулам: {med:.3f}")
        lines.append(f"Доля молекул с combined_min <= 0 (IC50 <= 1 мкМ): {frac_sub_zero:.1%}")
        if frac_sub_zero > 0.1:
            lines.append("Скоры в фармакологически осмысленном диапазоне.")
        else:
            lines.append("Мало молекул в фармакологически значимом диапазоне (IC50 <= 1 мкМ).")

    lines.append("")
    return "\n".join(lines)


@click.command()
@click.option("--input", "input_path", default="report/data/all_results.parquet", help="Parquet с результатами")
@click.option("--metrics", "metrics_path", default="report/metrics.json", help="Метрики хитов")
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
    """Сформировать текстовый отчёт с мини-аналитикой."""
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

    report = _build_report(df, metrics, stats, pairs_cfg, thresholds_cfg)

    out_path = PROJECT_ROOT / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    logger.info("Отчёт сохранён: {}", out_path)

    # Печать в stdout для видимости в пайплайне
    print(report)


if __name__ == "__main__":
    main()
