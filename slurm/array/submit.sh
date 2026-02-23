#!/usr/bin/env bash
# Submit array-based SLURM dependency chain (multi-node, 1 GPU per task).
#
# Stages:
#   fetch (s0+s1) -> calibrate (s2) -> array[gen+score] (s3+s4) -> analyze (s5) -> report (s6)
#                                                                                -> audit  (s7)
#                                                                -> array[gnina] (optional)
#                                                                -> array[md]    (optional, per-pair)
#
# Usage:
#   slurm/array/submit.sh                         # core pipeline only
#   slurm/array/submit.sh --skip-fetch            # skip fetch stage
#   slurm/array/submit.sh --max-concurrent=4      # limit concurrent array tasks
#   slurm/array/submit.sh --with-gnina            # add GNINA cross-validation
#   slurm/array/submit.sh --with-md               # add MD validation (12 array tasks, per pair)
#   slurm/array/submit.sh --with-gnina --with-md  # add both
#
# Retry failed tasks:
#   sbatch --array=3,17,42 slurm/array/03_generate_score.sbatch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

SKIP_FETCH=0
MAX_CONCURRENT=8
WITH_GNINA=0
WITH_MD=0
for arg in "$@"; do
    case "$arg" in
        --skip-fetch) SKIP_FETCH=1 ;;
        --max-concurrent=*) MAX_CONCURRENT="${arg#*=}" ;;
        --with-gnina) WITH_GNINA=1 ;;
        --with-md)    WITH_MD=1 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# Get total task count from task_matrix.py
TASK_COUNT=$(uv run python slurm/task_matrix.py --count)
MAX_INDEX=$((TASK_COUNT - 1))
# Number of active pairs (for MD array)
N_PAIRS=12

echo "=== DualDocking SLURM chain (array) ==="
echo "  Project root   : $DD_ROOT"
echo "  Run ID         : $EXPERIMENT_RUN_ID"
echo "  Log directory  : $DD_LOG_DIR"
echo "  Total tasks    : $TASK_COUNT"
echo "  Max concurrent : $MAX_CONCURRENT"
echo "  GNINA          : $([ "$WITH_GNINA" == 1 ] && echo yes || echo no)"
echo "  MD             : $([ "$WITH_MD" == 1 ] && echo yes || echo no)"
echo ""

DEP=""

# Stage s0+s1: Fetch + Prepare
if [[ "$SKIP_FETCH" == "0" ]]; then
    FETCH_JOB=$(sbatch --parsable --export=ALL "$SCRIPT_DIR/01_fetch.sbatch")
    echo "Submitted fetch (s0+s1)    : $FETCH_JOB"
    DEP="--dependency=afterok:$FETCH_JOB"
fi

# Stage s2: Calibrate
CAL_JOB=$(sbatch --parsable --export=ALL $DEP "$SCRIPT_DIR/02_calibrate.sbatch")
echo "Submitted calibrate (s2)   : $CAL_JOB"

# Stage s3+s4: Generate + Score (array job)
GEN_JOB=$(sbatch --parsable --export=ALL \
    --dependency=afterok:"$CAL_JOB" \
    --array="0-${MAX_INDEX}%${MAX_CONCURRENT}" \
    "$SCRIPT_DIR/03_generate_score.sbatch")
echo "Submitted gen+score (s3+s4): $GEN_JOB (array 0-${MAX_INDEX}%${MAX_CONCURRENT})"

# Stage s5: Analyze (runs after ALL array elements complete)
ANA_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:"$GEN_JOB" "$SCRIPT_DIR/04_analyze.sbatch")
echo "Submitted analyze (s5)     : $ANA_JOB"

# Stage s6: Report
REP_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:"$ANA_JOB" "$SCRIPT_DIR/05_report.sbatch")
echo "Submitted report (s6)      : $REP_JOB"

# Stage s7: Audit (depends on analyze, parallel with report)
AUD_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:"$ANA_JOB" "$SCRIPT_DIR/06_audit.sbatch")
echo "Submitted audit (s7)       : $AUD_JOB"

ALL_JOBS="${FETCH_JOB:+$FETCH_JOB,}${CAL_JOB},${GEN_JOB},${ANA_JOB},${REP_JOB},${AUD_JOB}"

# Optional: GNINA cross-validation (array job, same task matrix, depends on gen+score)
GNINA_JOB=""
if [[ "$WITH_GNINA" == "1" ]]; then
    GNINA_JOB=$(sbatch --parsable --export=ALL \
        --dependency=afterok:"$GEN_JOB" \
        --array="0-${MAX_INDEX}%${MAX_CONCURRENT}" \
        "$SCRIPT_DIR/07_gnina.sbatch")
    echo "Submitted gnina            : $GNINA_JOB (array 0-${MAX_INDEX}%${MAX_CONCURRENT})"
    ALL_JOBS="${ALL_JOBS},${GNINA_JOB}"
fi

# Optional: MD validation (array job, 1 per pair, depends on gen+score)
MD_JOB=""
if [[ "$WITH_MD" == "1" ]]; then
    MD_MAX=$((N_PAIRS - 1))
    MD_JOB=$(sbatch --parsable --export=ALL \
        --dependency=afterok:"$GEN_JOB" \
        --array="0-${MD_MAX}%4" \
        "$SCRIPT_DIR/08_md.sbatch")
    echo "Submitted md               : $MD_JOB (array 0-${MD_MAX}%4)"
    ALL_JOBS="${ALL_JOBS},${MD_JOB}"
fi

echo ""
echo "=== Dependency chain ==="
echo "  ... -> gen+score[$GEN_JOB] -> analyze($ANA_JOB) -> report($REP_JOB)"
echo "                                                   -> audit($AUD_JOB)"
[[ -n "$GNINA_JOB" ]] && echo "                              -> gnina[$GNINA_JOB]"
[[ -n "$MD_JOB" ]]    && echo "                              -> md[$MD_JOB]"
echo ""
echo "Monitor all    : squeue -u \$USER -j $ALL_JOBS"
echo "Monitor array  : squeue -j $GEN_JOB"
echo "Task matrix    : uv run python slurm/task_matrix.py --list"
echo "Retry failures : sbatch --array=<indices> slurm/array/03_generate_score.sbatch"
echo "Cancel all     : scancel $ALL_JOBS"
