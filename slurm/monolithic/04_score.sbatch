#!/usr/bin/env bash
#SBATCH --job-name=dd-score
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

# Resolve scorer backend from configs/experiment.yaml
SCORER_BACKEND=$(uv run python - <<'PY'
from dualdocking.config import load_experiment
cfg = load_experiment("configs").get("scorer", {})
print(str(cfg.get("backend", "cli")).lower())
PY
)

DD_USE_NIM_MODE="${DD_USE_NIM:-auto}"
if [[ "$DD_USE_NIM_MODE" == "auto" ]]; then
    if [[ "$SCORER_BACKEND" == "nim" ]]; then
        DD_USE_NIM_MODE=1
    else
        DD_USE_NIM_MODE=0
    fi
fi

# Optional: start NIM containers for Boltz-2 scoring
if [[ "$DD_USE_NIM_MODE" == "1" ]]; then
    source "$DD_ROOT/slurm/nim/lifecycle.sh"
    nim_start_all "$N_GPUS"
    trap 'nim_stop_all' EXIT
elif [[ "$SCORER_BACKEND" == "nim" ]]; then
    echo "[$(date -u)] WARN score backend=nim but DD_USE_NIM=0; expecting external NIM service(s)"
fi

echo "[$(date -u)] START score (s4)  job=$SLURM_JOB_ID  gpus=$N_GPUS  scorer_backend=$SCORER_BACKEND  use_nim=$DD_USE_NIM_MODE"
uv run python cli/pipeline/run_stages.py --stages score --skip-preflight --parallel "$N_GPUS" --resume
echo "[$(date -u)] DONE  score (s4)  job=$SLURM_JOB_ID"
