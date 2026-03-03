#!/usr/bin/env bash
# Submit monolithic SLURM dependency chain (single node, N GPUs).
#
# Stages:
#   00_fetch (s0) -> 01_prepare (s1) -> 02_calibrate (s2) -> 03_generate (s3) -> 04_score (s4) -> 05_analyze (s5) -> 06_report (s6)
#                                                                                                                  -> 07_audit  (s7)
#                                                                               -> 08_gnina (s8, optional)
#                                                                               -> 09_md    (s9, optional)
#
# Usage:
#   slurm/monolithic/submit.sh                        # core pipeline only
#   slurm/monolithic/submit.sh --skip-fetch           # skip fetch + prepare stages
#   slurm/monolithic/submit.sh --with-gnina           # add GNINA cross-validation
#   slurm/monolithic/submit.sh --with-md              # add MD validation
#   slurm/monolithic/submit.sh --with-gnina --with-md # add both
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

SKIP_FETCH=0
WITH_GNINA=0
WITH_MD=0
for arg in "$@"; do
    case "$arg" in
        --skip-fetch) SKIP_FETCH=1 ;;
        --with-gnina) WITH_GNINA=1 ;;
        --with-md)    WITH_MD=1 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

echo "=== DualDocking SLURM chain (monolithic) ==="
echo "  Project root : $DD_ROOT"
echo "  Run ID       : $EXPERIMENT_RUN_ID"
echo "  Log directory: $DD_LOG_DIR"
echo "  GNINA        : $([ "$WITH_GNINA" == 1 ] && echo yes || echo no)"
echo "  MD           : $([ "$WITH_MD" == 1 ] && echo yes || echo no)"
echo ""

DEP=""

# Stage s0: Fetch
if [[ "$SKIP_FETCH" == "0" ]]; then
    FETCH_JOB=$(sbatch --parsable --export=ALL "$SCRIPT_DIR/00_fetch.sbatch")
    echo "Submitted fetch (s0)       : $FETCH_JOB"
    DEP="--dependency=afterok:$FETCH_JOB"
fi

# Stage s1: Prepare
if [[ "$SKIP_FETCH" == "0" ]]; then
    PREP_JOB=$(sbatch --parsable --export=ALL $DEP "$SCRIPT_DIR/01_prepare.sbatch")
    echo "Submitted prepare (s1)     : $PREP_JOB"
    DEP="--dependency=afterok:$PREP_JOB"
fi

# Stage s2: Calibrate
CAL_JOB=$(sbatch --parsable --export=ALL $DEP "$SCRIPT_DIR/02_calibrate.sbatch")
echo "Submitted calibrate (s2)   : $CAL_JOB"

# Stage s3: Generate
GEN_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:"$CAL_JOB" "$SCRIPT_DIR/03_generate.sbatch")
echo "Submitted generate (s3)    : $GEN_JOB"

# Stage s4: Score
SCORE_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:"$GEN_JOB" "$SCRIPT_DIR/04_score.sbatch")
echo "Submitted score (s4)       : $SCORE_JOB"

# Stage s5: Analyze
ANA_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:"$SCORE_JOB" "$SCRIPT_DIR/05_analyze.sbatch")
echo "Submitted analyze (s5)     : $ANA_JOB"

# Stage s6: Report
REP_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:"$ANA_JOB" "$SCRIPT_DIR/06_report.sbatch")
echo "Submitted report (s6)      : $REP_JOB"

# Stage s7: Audit (depends on analyze, parallel with report)
AUD_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:"$ANA_JOB" "$SCRIPT_DIR/07_audit.sbatch")
echo "Submitted audit (s7)       : $AUD_JOB"

ALL_JOBS="${FETCH_JOB:+$FETCH_JOB,}${PREP_JOB:+$PREP_JOB,}${CAL_JOB},${GEN_JOB},${SCORE_JOB},${ANA_JOB},${REP_JOB},${AUD_JOB}"

# Optional: GNINA cross-validation (depends on score, parallel with analyze)
GNINA_JOB=""
if [[ "$WITH_GNINA" == "1" ]]; then
    GNINA_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:"$SCORE_JOB" "$SCRIPT_DIR/08_gnina.sbatch")
    echo "Submitted gnina (s8)       : $GNINA_JOB"
    ALL_JOBS="${ALL_JOBS},${GNINA_JOB}"
fi

# Optional: MD validation (depends on score, parallel with analyze)
MD_JOB=""
if [[ "$WITH_MD" == "1" ]]; then
    MD_JOB=$(sbatch --parsable --export=ALL --dependency=afterok:"$SCORE_JOB" "$SCRIPT_DIR/09_md.sbatch")
    echo "Submitted md (s9)          : $MD_JOB"
    ALL_JOBS="${ALL_JOBS},${MD_JOB}"
fi

echo ""
echo "=== Dependency chain ==="
echo "  ... -> generate($GEN_JOB) -> score($SCORE_JOB) -> analyze($ANA_JOB) -> report($REP_JOB)"
echo "                                                                       -> audit($AUD_JOB)"
[[ -n "$GNINA_JOB" ]] && echo "                                                  -> gnina($GNINA_JOB)"
[[ -n "$MD_JOB" ]]    && echo "                                                  -> md($MD_JOB)"
echo ""
echo "Monitor: squeue -u \$USER -j $ALL_JOBS"
echo "Cancel:  scancel $ALL_JOBS"
