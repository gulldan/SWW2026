#!/usr/bin/env bash
# Common environment setup for all SLURM jobs.
# Source this file at the top of every sbatch script:
#   source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
set -euo pipefail

# ── Project root (one level above slurm/) ──────────────────────────
export DD_ROOT="${DD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$DD_ROOT"

# ── HPC module system (uncomment / adapt for your cluster) ─────────
# module purge
# module load cuda/12.4
# module load apptainer/1.3

# ── Python (uv must be on PATH) ────────────────────────────────────
export UV_PROJECT_ENVIRONMENT="$DD_ROOT/.venv"

# ── Experiment run ID (shared across all jobs in a chain) ──────────
export EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:-slurm_$(date -u +%Y%m%dT%H%M%SZ)}"

# ── Log directory ───────────────────────────────────────────────────
export DD_LOG_DIR="${DD_LOG_DIR:-$DD_ROOT/slurm/logs/$EXPERIMENT_RUN_ID}"
mkdir -p "$DD_LOG_DIR"

# ── Container runtime (docker or apptainer) ─────────────────────────
export DD_CONTAINER_RUNTIME="${DD_CONTAINER_RUNTIME:-docker}"

# ── Boltz-2 NIM lifecycle mode ───────────────────────────────────────
# auto: follow scorer.backend in configs/experiment.yaml
# 0: do not manage local NIM containers in sbatch jobs
# 1: start/stop local NIM containers in score stages
export DD_USE_NIM="${DD_USE_NIM:-auto}"
