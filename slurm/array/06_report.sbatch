#!/usr/bin/env bash
#SBATCH --job-name=dd-report
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

echo "[$(date -u)] START report (s6)  job=$SLURM_JOB_ID"
uv run python cli/s6_report/plot_results.py
uv run python cli/s6_report/render_molecules.py

if [[ "${DD_LATEX_MODE:-off}" != "off" ]]; then
    uv run python cli/s6_report/generate_latex_report.py --mode "${DD_LATEX_MODE}"
fi
echo "[$(date -u)] DONE  report (s6)  job=$SLURM_JOB_ID"
