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
# Override: DD_MD_PAIRS="CDK2_ERK2 AChE_BChE" DD_MD_SIM_TIME=10

PAIRS="${DD_MD_PAIRS:-CDK2_ERK2 AChE_BChE BACE1_GSK3B AChE_BACE1 GSK3B_CDK5 EGFR_HER2 BACE1_CDK5 AChE_MAOB MEK1_MEK2 FGFR1_FGFR2 EZH2_EZH1 BRD4_BRD2}"
SIM_TIME_ARG=""
if [[ -n "${DD_MD_SIM_TIME:-}" ]]; then
    SIM_TIME_ARG="--simulation-time $DD_MD_SIM_TIME"
fi

echo "[$(date -u)] START md validation  job=$SLURM_JOB_ID  pairs=$PAIRS"
for pair in $PAIRS; do
    echo "[$(date -u)]   MD $pair"
    uv run python cli/s9_md/run_md.py --pair "$pair" --docker $SIM_TIME_ARG || {
        echo "[$(date -u)]   WARN: MD failed for $pair (continuing)"
    }
done
echo "[$(date -u)] DONE  md validation  job=$SLURM_JOB_ID"
