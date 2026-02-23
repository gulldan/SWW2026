#!/usr/bin/env bash
#SBATCH --job-name=dd-gnina
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

echo "[$(date -u)] START gnina cross-validation  job=$SLURM_JOB_ID"
uv run python cli/pipeline/run_stages.py --stages gnina --resume
echo "[$(date -u)] DONE  gnina cross-validation  job=$SLURM_JOB_ID"
