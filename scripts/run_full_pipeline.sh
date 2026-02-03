#!/usr/bin/env bash
# Полный пайплайн: генерация + скоринг + аналитика для всех 4 пар.
# Запуск: bash scripts/run_full_pipeline.sh 2>&1 | tee pipeline.log
set -euo pipefail

cd /mnt/data/SWW2026

PAIRS=(CDK2_ERK2 EGFR_HER2 COX1_COX2 AChE_BChE)
MODES=(single_a single_b double)
SEEDS=(42 137 256 512 1024)

echo "=============================================="
echo " ПОЛНЫЙ ПАЙПЛАЙН — $(date)"
echo "=============================================="

# ─── 1. Генерация FREED++ ───────────────────────
for pair in "${PAIRS[@]}"; do
  for mode in "${MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      csv="results/${pair}/${mode}/seed_${seed}/generated.csv"
      if [[ -f "$csv" ]] && [[ $(wc -l < "$csv") -gt 1 ]]; then
        n=$(($(wc -l < "$csv") - 1))
        echo "[SKIP] Генерация ${pair}/${mode}/seed_${seed} — уже есть ${n} молекул"
        continue
      fi
      echo ""
      echo "====== ГЕНЕРАЦИЯ: ${pair} / ${mode} / seed_${seed} — $(date) ======"
      uv run python scripts/generate.py --pair "$pair" --mode "$mode" --seed "$seed"
      echo "[DONE] Генерация ${pair}/${mode}/seed_${seed} — $(date)"
    done
  done
done

# ─── 2. Скоринг Boltz-2 ─────────────────────────
for pair in "${PAIRS[@]}"; do
  for mode in "${MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      joined="results/${pair}/${mode}/seed_${seed}/scored_joined.csv"
      if [[ -f "$joined" ]]; then
        echo "[SKIP] Скоринг ${pair}/${mode}/seed_${seed} — scored_joined.csv уже есть"
        continue
      fi
      echo ""
      echo "====== СКОРИНГ: ${pair} / ${mode} / seed_${seed} — $(date) ======"
      uv run python scripts/score.py --pair "$pair" --mode "$mode" --seed "$seed"
      echo "[DONE] Скоринг ${pair}/${mode}/seed_${seed} — $(date)"
    done
  done
done

# ─── 3. Аналитика ────────────────────────────────
echo ""
echo "====== АГРЕГАЦИЯ — $(date) ======"
uv run python scripts/aggregate.py

echo "====== МЕТРИКИ — $(date) ======"
uv run python scripts/compute_metrics.py

echo "====== ГРАФИКИ — $(date) ======"
uv run python scripts/plot_results.py

echo "====== РЕНДЕР МОЛЕКУЛ — $(date) ======"
uv run python scripts/render_molecules.py

echo "====== СВОДКА — $(date) ======"
uv run python scripts/summarize.py

echo ""
echo "=============================================="
echo " ПАЙПЛАЙН ЗАВЕРШЁН — $(date)"
echo "=============================================="
echo " Отчёт: report/summary.txt"
echo " Графики: report/figures/"
echo "=============================================="
