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
# --array is set at submit time by submit.sh (e.g. --array=0-11%4)
# Each array element handles one pair.
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

# Active pairs in canonical order (must match task_matrix.py pair order)
PAIRS=(CDK2_ERK2 EGFR_HER2 AChE_BChE BACE1_GSK3B AChE_BACE1 GSK3B_CDK5 BACE1_CDK5 AChE_MAOB MEK1_MEK2 FGFR1_FGFR2 EZH2_EZH1 BRD4_BRD2)
PAIR="${PAIRS[$SLURM_ARRAY_TASK_ID]}"

SIM_TIME_ARG=""
if [[ -n "${DD_MD_SIM_TIME:-}" ]]; then
    SIM_TIME_ARG="--simulation-time $DD_MD_SIM_TIME"
fi

echo "[$(date -u)] START md  task=$SLURM_ARRAY_TASK_ID  pair=$PAIR  job=$SLURM_JOB_ID"
uv run python cli/s9_md/run_md.py --pair "$PAIR" --docker $SIM_TIME_ARG
echo "[$(date -u)] DONE  md  task=$SLURM_ARRAY_TASK_ID  pair=$PAIR"
