#!/usr/bin/env bash
#SBATCH --job-name=dd-task
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=%x_%A_%a.out
# --array is set at submit time by submit.sh (e.g. --array=0-179%8)
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

# Resolve (pair, mode, seed) from array index
eval "$(uv run python slurm/task_matrix.py --index "$SLURM_ARRAY_TASK_ID" --format env)"

echo "[$(date -u)] START task=$SLURM_ARRAY_TASK_ID  pair=$TASK_PAIR  mode=$TASK_MODE  seed=$TASK_SEED  job=$SLURM_JOB_ID"

# Optional: start NIM container for Boltz-2 scoring
if [[ "${DD_USE_NIM:-0}" == "1" ]]; then
    source "$DD_ROOT/slurm/nim/lifecycle.sh"
    nim_start_single
    trap 'nim_stop_single' EXIT
fi

# Generate molecules
uv run python cli/s3_generate/generate.py \
    --pair "$TASK_PAIR" \
    --mode "$TASK_MODE" \
    --seed "$TASK_SEED"

# Score molecules with Boltz-2
uv run python cli/s4_score/score.py \
    --pair "$TASK_PAIR" \
    --mode "$TASK_MODE" \
    --seed "$TASK_SEED"

echo "[$(date -u)] DONE  task=$SLURM_ARRAY_TASK_ID  pair=$TASK_PAIR  mode=$TASK_MODE  seed=$TASK_SEED"
