#!/usr/bin/env bash
#SBATCH --job-name=dd-audit
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

echo "[$(date -u)] START audit (s7)  job=$SLURM_JOB_ID"
uv run python cli/s7_audit/claim_audit.py
echo "[$(date -u)] DONE  audit (s7)  job=$SLURM_JOB_ID"
