#!/bin/bash
set -e

# --- Mode 0: Pass-through for shell/sleep/absolute paths (for docker exec usage) ---
case "$1" in
    bash|sh|sleep|/*)
        exec "$@"
        ;;
esac

# --- Mode 1: PDB → PDBQT conversion via Meeko (RDKit backend) ---
if [ "$1" = "prepare-receptor" ]; then
    if [ -z "$2" ] || [ -z "$3" ]; then
        echo "Usage: prepare-receptor <input.pdb> <output.pdbqt>" >&2
        exit 1
    fi
    out_base="${3%.pdbqt}"
    fixed_pdb="/tmp/receptor_fixed.pdb"
    /opt/micromamba/envs/unidock/bin/python /opt/pdbfixer_add_h.py --pdb "$2" --out "$fixed_pdb"
    mk_prepare_receptor.py --pdb "$fixed_pdb" -o "$out_base" --skip_gpf
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
