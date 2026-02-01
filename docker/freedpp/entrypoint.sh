#!/bin/bash
set -e

# --- Mode 1: PDB → PDBQT conversion via OpenBabel ---
if [ "$1" = "prepare-receptor" ]; then
    if [ -z "$2" ] || [ -z "$3" ]; then
        echo "Usage: prepare-receptor <input.pdb> <output.pdbqt>" >&2
        exit 1
    fi
    obabel -ipdb "$2" -opdbqt -O "$3" -h
    exit 0
fi

# --- Mode 2: FREED++ training / sampling ---

# If dual-mode is requested, add the patch to PYTHONPATH
if [ -n "$DUAL_RECEPTOR_B" ]; then
    export PYTHONPATH="/opt:$PYTHONPATH"
    echo "[entrypoint] Dual-target mode: DUAL_RECEPTOR_B=$DUAL_RECEPTOR_B"
fi

cd /opt/freed
exec python -m ffreed.main "$@"
