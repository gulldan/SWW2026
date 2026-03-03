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

# Detect GPU count from effective visibility inside allocation
N_GPUS=$(uv run python - <<'PY'
from dualdocking.pipeline.core import detect_gpu_count
print(detect_gpu_count())
PY
)

echo "[$(date -u)] START gnina cross-validation  job=$SLURM_JOB_ID  gpus=$N_GPUS"
uv run python cli/pipeline/run_stages.py --stages gnina --parallel "$N_GPUS" --resume
echo "[$(date -u)] DONE  gnina cross-validation  job=$SLURM_JOB_ID"
