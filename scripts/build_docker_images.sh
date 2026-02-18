#!/usr/bin/env bash
# Build all project Docker images sequentially.
#
# Images (in build order):
#   1. dualdocking/freedpp:latest  — FREED++ generation (required)
#   2. dualdocking/gnina:latest    — GNINA scoring     (optional)
#   3. dualdocking/md:latest       — MD validation      (optional)
#
# By default existing images are skipped.  Use --force to rebuild.
#
# Usage:
#   ./scripts/build_docker_images.sh                    # build all, skip existing
#   ./scripts/build_docker_images.sh --force            # force-rebuild all
#   ./scripts/build_docker_images.sh --only freedpp     # build only freedpp
#   ./scripts/build_docker_images.sh --no-md            # skip md image
#   ./scripts/build_docker_images.sh --cuda-arch 80     # A100 instead of RTX 4090
#   ./scripts/build_docker_images.sh --dry-run          # preview commands
set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root (parent of scripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DRY_RUN=0
FORCE=0
CUDA_ARCH="89"
ONLY=""
SKIP_FREEDPP=0
SKIP_GNINA=0
SKIP_MD=0

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=1;   shift ;;
        --force)      FORCE=1;     shift ;;
        --cuda-arch)  CUDA_ARCH="$2"; shift 2 ;;
        --only)       ONLY="$2";   shift 2 ;;
        --no-freedpp) SKIP_FREEDPP=1; shift ;;
        --no-gnina)   SKIP_GNINA=1;  shift ;;
        --no-md)      SKIP_MD=1;     shift ;;
        -h|--help)
            awk '/^#/{sub(/^# ?/,""); print} /^set /{exit}' "$0" | tail -n +2
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'  # no colour

log()   { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[SKIP]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }

image_exists() {
    docker image inspect "$1" >/dev/null 2>&1
}

# ---------------------------------------------------------------------------
# Image table:  key | tag | dockerfile | build-args | timeout
# ---------------------------------------------------------------------------
KEYS=(   freedpp                          gnina                          md )
TAGS=(   dualdocking/freedpp:latest       dualdocking/gnina:latest       dualdocking/md:latest )
DFILES=( docker/freedpp/Dockerfile        docker/gnina/Dockerfile        docker/md/Dockerfile )
BARGS=( ""                               ""                             "CUDA_ARCHITECTURES=$CUDA_ARCH" )
TOUTS=(  3600                             3600                           7200 )

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" -eq 0 ]]; then
    if ! docker info >/dev/null 2>&1; then
        fail "Docker is not available. Install Docker and ensure the daemon is running."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Build loop
# ---------------------------------------------------------------------------
BUILT=()
SKIPPED=()
FAILED=()
TOTAL_T0=$SECONDS

for i in "${!KEYS[@]}"; do
    key="${KEYS[$i]}"
    tag="${TAGS[$i]}"
    dockerfile="$REPO_ROOT/${DFILES[$i]}"
    build_args="${BARGS[$i]}"
    timeout="${TOUTS[$i]}"

    # --only filter
    if [[ -n "$ONLY" && "$key" != "$ONLY" ]]; then
        SKIPPED+=("$key")
        continue
    fi

    # --no-<key> filter
    case "$key" in
        freedpp) [[ "$SKIP_FREEDPP" -eq 1 ]] && { warn "$key — skipped (--no-freedpp)"; SKIPPED+=("$key"); continue; } ;;
        gnina)   [[ "$SKIP_GNINA"   -eq 1 ]] && { warn "$key — skipped (--no-gnina)";   SKIPPED+=("$key"); continue; } ;;
        md)      [[ "$SKIP_MD"      -eq 1 ]] && { warn "$key — skipped (--no-md)";       SKIPPED+=("$key"); continue; } ;;
    esac

    # Skip existing (unless --force)
    if [[ "$FORCE" -eq 0 ]] && image_exists "$tag"; then
        warn "$key — image exists: $tag (use --force to rebuild)"
        SKIPPED+=("$key")
        continue
    fi

    # Check Dockerfile exists
    if [[ ! -f "$dockerfile" ]]; then
        fail "$key — Dockerfile not found: $dockerfile"
        FAILED+=("$key")
        break
    fi

    # Build command
    cmd=(docker build -f "$dockerfile" -t "$tag")
    if [[ -n "$build_args" ]]; then
        cmd+=(--build-arg "$build_args")
    fi
    cmd+=("$REPO_ROOT")

    log "$key — ${cmd[*]}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        BUILT+=("$key")
        continue
    fi

    t0=$SECONDS
    if timeout "$timeout" "${cmd[@]}"; then
        elapsed=$(( SECONDS - t0 ))
        ok "$key — built $tag in ${elapsed}s"
        BUILT+=("$key")
    else
        rc=$?
        elapsed=$(( SECONDS - t0 ))
        if [[ $rc -eq 124 ]]; then
            fail "$key — TIMEOUT after ${timeout}s"
        else
            fail "$key — FAILED (rc=$rc) after ${elapsed}s"
        fi
        FAILED+=("$key")
        break
    fi
done

TOTAL_ELAPSED=$(( SECONDS - TOTAL_T0 ))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
log "--- Docker Build Summary ---"
for k in "${BUILT[@]:-}";   do [[ -n "$k" ]] && ok   "  $k"; done
for k in "${SKIPPED[@]:-}"; do [[ -n "$k" ]] && warn  "  $k"; done
for k in "${FAILED[@]:-}";  do [[ -n "$k" ]] && fail  "  $k"; done
log "Total time: ${TOTAL_ELAPSED}s"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    fail "Docker builds FAILED: ${FAILED[*]}"
    exit 1
fi
