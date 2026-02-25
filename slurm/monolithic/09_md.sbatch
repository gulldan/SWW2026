#!/usr/bin/env bash
#SBATCH --job-name=dd-md
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.out
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

# MD validation for all active pairs (top-3 molecules, 50 ns default).
# Pair order is resolved dynamically from task_matrix.py (configs/pairs.yaml).
# Override pair set: DD_MD_PAIRS="EGFR_HER2 AChE_BChE"
mapfile -t PAIRS < <(uv run python slurm/task_matrix.py --pairs | awk 'NR>1 {print $2}')
if [[ -n "${DD_MD_PAIRS:-}" ]]; then
    read -r -a PAIRS <<< "$DD_MD_PAIRS"
fi

SIM_TIME_ARG=""
if [[ -n "${DD_MD_SIM_TIME:-}" ]]; then
    SIM_TIME_ARG="--simulation-time $DD_MD_SIM_TIME"
fi

echo "[$(date -u)] START md validation  job=$SLURM_JOB_ID  pairs=${PAIRS[*]}"
for pair in "${PAIRS[@]}"; do
    echo "[$(date -u)]   MD $pair"
    uv run python cli/s9_md/run_md.py --pair "$pair" --docker $SIM_TIME_ARG || {
        echo "[$(date -u)]   WARN: MD failed for $pair (continuing)"
    }
done
echo "[$(date -u)] DONE  md validation  job=$SLURM_JOB_ID"
