#!/usr/bin/env bash
#SBATCH --job-name=dd-gnina
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=%x_%A_%a.out
# --array is set at submit time by submit.sh (e.g. --array=0-179%8)
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

# Resolve (pair, mode, seed) from array index
eval "$(uv run python slurm/task_matrix.py --index "$SLURM_ARRAY_TASK_ID" --format env)"

echo "[$(date -u)] START gnina  task=$SLURM_ARRAY_TASK_ID  pair=$TASK_PAIR  mode=$TASK_MODE  seed=$TASK_SEED"
uv run python cli/s4_score/score_gnina.py \
    --pair "$TASK_PAIR" \
    --mode "$TASK_MODE" \
    --seed "$TASK_SEED"
echo "[$(date -u)] DONE  gnina  task=$SLURM_ARRAY_TASK_ID"
