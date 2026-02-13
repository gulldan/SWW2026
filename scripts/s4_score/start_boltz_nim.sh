#!/usr/bin/env bash
set -euo pipefail

# Start NIM Boltz-2 instances — one per GPU.
#
# Usage:
#   ./scripts/s4_score/start_boltz_nim.sh          # auto-detect GPUs
#   ./scripts/s4_score/start_boltz_nim.sh 2        # explicitly 2 GPUs
#   ./scripts/s4_score/start_boltz_nim.sh stop     # stop all instances
#
# GPU 0 → port 8000 (container boltz2_nim_gpu0)
# GPU 1 → port 8001 (container boltz2_nim_gpu1)
# ...
#
# Requirements:
#   - Docker with nvidia runtime
#   - Image nvcr.io/nim/mit/boltz2:1.5.0 (or docker load from boltz2_1.5.0.tar)
#   - Model cache in output/cache/boltz_nim/ (prepare via docker/boltz2/boltz2nim.sh)

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
NIM_IMAGE="nvcr.io/nim/mit/boltz2:1.5.0"
NIM_CACHE="$PROJECT_DIR/output/cache/boltz_nim"
BASE_PORT=8000
CONTAINER_PREFIX="boltz2_nim_gpu"

# --- Stop command ---
if [[ "${1:-}" == "stop" ]]; then
    info "Stopping all NIM Boltz-2 containers..."
    for cid in $(docker ps -q --filter "name=${CONTAINER_PREFIX}"); do
        name=$(docker inspect --format '{{.Name}}' "$cid" | sed 's|/||')
        docker stop "$name" 2>/dev/null && ok "Stopped $name"
    done
    exit 0
fi

# --- Detect GPU count ---
if [[ -n "${1:-}" ]]; then
    N_GPUS="$1"
else
    N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    N_GPUS=$(echo "$N_GPUS" | tr -d ' ')
fi

if [[ "$N_GPUS" -lt 1 ]]; then
    error "No GPUs detected"
    exit 1
fi

info "Starting ${BOLD}${N_GPUS}${NC} NIM Boltz-2 instance(s)"
info "Image: $NIM_IMAGE"
info "Cache: $NIM_CACHE"

# Ensure cache dir exists
mkdir -p "$NIM_CACHE"

# --- Start one container per GPU ---
for gpu_id in $(seq 0 $((N_GPUS - 1))); do
    port=$((BASE_PORT + gpu_id))
    name="${CONTAINER_PREFIX}${gpu_id}"

    # Stop existing container with same name
    if docker ps -q --filter "name=$name" | grep -q .; then
        info "Container $name already running on port $port"
        continue
    fi
    docker rm "$name" 2>/dev/null || true

    info "Starting $name — GPU $gpu_id, port $port"
    docker run -d \
        --name "$name" \
        --gpus "\"device=$gpu_id\"" \
        --shm-size=16G \
        -e NGC_API_KEY="${NGC_API_KEY:-}" \
        -e NIM_BOLTZ_STRUCTURE_OPTIMIZED_BACKEND=trt \
        -v "$NIM_CACHE":/opt/nim/.cache \
        -p "${port}:8000" \
        "$NIM_IMAGE"

    ok "$name started (GPU $gpu_id → localhost:$port)"
done

# --- Wait for health ---
info "Waiting for NIM instances to become ready..."
for gpu_id in $(seq 0 $((N_GPUS - 1))); do
    port=$((BASE_PORT + gpu_id))
    name="${CONTAINER_PREFIX}${gpu_id}"
    info "Checking $name (localhost:$port)..."

    retries=0
    max_retries=120  # 10 min max
    until curl -sf "http://localhost:${port}/v1/health/ready" 2>/dev/null | grep -q "true"; do
        retries=$((retries + 1))
        if [[ $retries -ge $max_retries ]]; then
            error "$name did not become ready after ${max_retries}×5s"
            error "Check logs: docker logs $name"
            exit 1
        fi
        sleep 5
    done
    ok "$name ready (localhost:$port)"
done

echo ""
echo -e "${GREEN}${BOLD}All $N_GPUS NIM instance(s) ready.${NC}"
echo ""
echo "NIM URLs for experiment.yaml:"
for gpu_id in $(seq 0 $((N_GPUS - 1))); do
    port=$((BASE_PORT + gpu_id))
    echo "  - http://localhost:$port   # GPU $gpu_id"
done
echo ""
echo "To stop:  ./scripts/s4_score/start_boltz_nim.sh stop"
