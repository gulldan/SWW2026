#!/usr/bin/env bash
#SBATCH --job-name=dd-calibrate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

echo "[$(date -u)] START calibrate  job=$SLURM_JOB_ID"
uv run python cli/s2_calibrate/dock_calibration.py
uv run python cli/s2_calibrate/compute_normalization.py
echo "[$(date -u)] DONE  calibrate  job=$SLURM_JOB_ID"
