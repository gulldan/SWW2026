#!/usr/bin/env bash
#SBATCH --job-name=dd-score
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=%x_%A_%a.out
# --array is set at submit time by submit.sh (e.g. --array=0-149%8)
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

# Resolve (pair, mode, seed) from array index
eval "$(uv run python slurm/task_matrix.py --index "$SLURM_ARRAY_TASK_ID" --format env)"

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

echo "[$(date -u)] START score  task=$SLURM_ARRAY_TASK_ID  pair=$TASK_PAIR  mode=$TASK_MODE  seed=$TASK_SEED  job=$SLURM_JOB_ID  scorer_backend=$SCORER_BACKEND  use_nim=$DD_USE_NIM_MODE"

# Optional: start NIM container for Boltz-2 scoring
if [[ "$DD_USE_NIM_MODE" == "1" ]]; then
    source "$DD_ROOT/slurm/nim/lifecycle.sh"
    nim_start_single
    trap 'nim_stop_single' EXIT
elif [[ "$SCORER_BACKEND" == "nim" ]]; then
    echo "[$(date -u)] WARN score backend=nim but DD_USE_NIM=0; expecting external NIM service"
fi

# Score molecules with Boltz-2
uv run python cli/s4_score/score.py \
    --pair "$TASK_PAIR" \
    --mode "$TASK_MODE" \
    --seed "$TASK_SEED"

echo "[$(date -u)] DONE  score  task=$SLURM_ARRAY_TASK_ID  pair=$TASK_PAIR  mode=$TASK_MODE  seed=$TASK_SEED"
