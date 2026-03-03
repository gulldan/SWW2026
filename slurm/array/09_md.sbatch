#!/usr/bin/env bash
#SBATCH --job-name=dd-md
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --output=%x_%A_%a.out
# --array is set at submit time by submit.sh (e.g. --array=0-9%4)
# Each array element handles one active pair from configs/pairs.yaml.
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

# Resolve pair from array index (canonical order from task matrix)
eval "$(uv run python slurm/task_matrix.py --pair-index "$SLURM_ARRAY_TASK_ID" --format env)"

SIM_TIME_ARG=""
if [[ -n "${DD_MD_SIM_TIME:-}" ]]; then
    SIM_TIME_ARG="--simulation-time $DD_MD_SIM_TIME"
fi

echo "[$(date -u)] START md  task=$SLURM_ARRAY_TASK_ID  pair=$TASK_PAIR  job=$SLURM_JOB_ID"
uv run python cli/s9_md/run_md.py --pair "$TASK_PAIR" --docker $SIM_TIME_ARG
echo "[$(date -u)] DONE  md  task=$SLURM_ARRAY_TASK_ID  pair=$TASK_PAIR"
