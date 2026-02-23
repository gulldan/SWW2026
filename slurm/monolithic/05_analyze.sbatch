#!/usr/bin/env bash
#SBATCH --job-name=dd-analyze
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

echo "[$(date -u)] START analyze (s5)  job=$SLURM_JOB_ID"
uv run python cli/s5_analyze/aggregate.py
uv run python cli/s5_analyze/compute_metrics.py
uv run python cli/s5_analyze/threshold_sensitivity.py
uv run python cli/s5_analyze/proxy_correlation.py
uv run python cli/s5_analyze/summarize.py
echo "[$(date -u)] DONE  analyze (s5)  job=$SLURM_JOB_ID"
