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

# Detect GPU count from effective visibility inside allocation
N_GPUS=$(uv run python - <<'PY'
from dualdocking.pipeline.core import detect_gpu_count
print(detect_gpu_count())
PY
)

echo "[$(date -u)] START generate (s3)  job=$SLURM_JOB_ID  gpus=$N_GPUS"
uv run python cli/pipeline/run_stages.py --stages generate --skip-preflight --parallel "$N_GPUS" --resume
echo "[$(date -u)] DONE  generate (s3)  job=$SLURM_JOB_ID"
