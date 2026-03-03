#!/bin/bash
set -e

# ── Activate Python venv (must be first — GMXRC may alter PATH) ────
export VIRTUAL_ENV=/opt/venv
export PATH="/opt/venv/bin:$PATH"

# ── Source GROMACS environment ──────────────────────────────────────
source /opt/gromacs/bin/GMXRC

# ── Add conda AmberTools to PATH (after GMXRC, before venv) ────────
export PATH="/opt/micromamba/envs/ambertools/bin:$PATH"
export AMBERHOME="/opt/micromamba/envs/ambertools"

# ── Ensure venv python is still first in PATH ──────────────────────
export PATH="/opt/venv/bin:$PATH"

# ── Make project packages + conda packages (pdbfixer, openmm) importable ──
CONDA_ENV=/opt/micromamba/envs/ambertools
CONDA_SP=$(find "${CONDA_ENV}/lib" -maxdepth 1 -name 'python3.*' -type d 2>/dev/null | head -1)/site-packages
export PYTHONPATH="/workspace:${CONDA_SP}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export OPENMM_DEFAULT_PLATFORM=CPU

# ── Mode 0: Pass-through for shell/sleep/absolute paths ────────────
case "$1" in
    bash|sh|sleep|/*)
        exec "$@"
        ;;
esac

# ── Default: run MD validation CLI ──────────────────────────────────
exec python /workspace/cli/s9_md/run_md.py "$@"
