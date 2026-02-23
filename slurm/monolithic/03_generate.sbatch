#!/usr/bin/env bash
#SBATCH --job-name=dd-generate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=7-00:00:00
#SBATCH --output=%x_%j.out
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

# Detect GPU count from SLURM allocation
N_GPUS="${SLURM_GPUS_ON_NODE:-$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)}"

echo "[$(date -u)] START generate (s3)  job=$SLURM_JOB_ID  gpus=$N_GPUS"
uv run python cli/pipeline/run_stages.py --stages generate --skip-preflight
echo "[$(date -u)] DONE  generate (s3)  job=$SLURM_JOB_ID"
