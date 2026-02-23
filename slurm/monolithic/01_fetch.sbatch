#!/usr/bin/env bash
#SBATCH --job-name=dd-fetch
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

echo "[$(date -u)] START fetch  job=$SLURM_JOB_ID"
uv run python cli/s0_fetch/fetch_all.py --skip-baseline
echo "[$(date -u)] DONE  fetch  job=$SLURM_JOB_ID"
