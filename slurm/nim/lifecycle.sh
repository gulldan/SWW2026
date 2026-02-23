#!/usr/bin/env bash
# NIM Boltz-2 container lifecycle helpers.
#
# Source this file from sbatch scripts when DD_USE_NIM=1:
#   source "$DD_ROOT/slurm/nim/lifecycle.sh"
#
# Monolithic (multi-GPU):
#   nim_start_all <n_gpus>   # start N containers on ports 8000..8000+N-1
#   nim_stop_all             # stop all containers started by this job
#
# Array (single GPU):
#   nim_start_single         # start 1 container on port 8000
#   nim_stop_single          # stop the container

NIM_IMAGE="${NIM_IMAGE:-nvcr.io/nim/mit/boltz2:1.5.0}"
NIM_CACHE="${NIM_CACHE:-${DD_ROOT:-.}/docker/boltz2/cache}"
NIM_SHM_SIZE="${NIM_SHM_SIZE:-16G}"
NIM_HEALTH_TIMEOUT="${NIM_HEALTH_TIMEOUT:-300}"  # seconds
NIM_BASE_PORT="${NIM_BASE_PORT:-8000}"

# Track container IDs for cleanup
_NIM_CONTAINERS=()

_nim_wait_healthy() {
    # Wait for a NIM container to become healthy.
    # Args: $1 = container name/id, $2 = port
    local container="$1" port="$2"
    local elapsed=0
    echo "[NIM] Waiting for $container on port $port (timeout ${NIM_HEALTH_TIMEOUT}s)..."
    while (( elapsed < NIM_HEALTH_TIMEOUT )); do
        if curl -sf "http://localhost:${port}/v1/health/ready" >/dev/null 2>&1; then
            echo "[NIM] $container healthy after ${elapsed}s"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "[NIM] ERROR: $container not healthy after ${NIM_HEALTH_TIMEOUT}s" >&2
    return 1
}

nim_start_single() {
    # Start a single NIM container on port NIM_BASE_PORT using GPU 0.
    local port="${NIM_BASE_PORT}"
    local name="nim-boltz2-${SLURM_JOB_ID:-$$}"

    mkdir -p "$NIM_CACHE"

    docker run -d \
        --name "$name" \
        --gpus '"device=0"' \
        --shm-size="$NIM_SHM_SIZE" \
        -e NGC_API_KEY="${NGC_API_KEY:-}" \
        -e NIM_BOLTZ_STRUCTURE_OPTIMIZED_BACKEND=torch \
        -e NIM_BOLTZ_AFFINITY_OPTIMIZED_BACKEND=torch \
        -v "${NIM_CACHE}:/opt/nim/.cache" \
        -p "${port}:8000" \
        "$NIM_IMAGE"

    _NIM_CONTAINERS+=("$name")
    _nim_wait_healthy "$name" "$port"

    export BOLTZ_BACKEND=nim
    export BOLTZ_NIM_URL="http://localhost:${port}"
    echo "[NIM] Exported BOLTZ_BACKEND=nim BOLTZ_NIM_URL=http://localhost:${port}"
}

nim_stop_single() {
    # Stop the single NIM container.
    for name in "${_NIM_CONTAINERS[@]}"; do
        echo "[NIM] Stopping $name..."
        docker stop "$name" 2>/dev/null || true
        docker rm "$name" 2>/dev/null || true
    done
    _NIM_CONTAINERS=()
}

nim_start_all() {
    # Start N NIM containers (one per GPU) on consecutive ports.
    # Args: $1 = number of GPUs
    local n_gpus="${1:-1}"
    mkdir -p "$NIM_CACHE"

    local urls=()
    for (( i=0; i<n_gpus; i++ )); do
        local port=$((NIM_BASE_PORT + i))
        local name="nim-boltz2-${SLURM_JOB_ID:-$$}-gpu${i}"

        docker run -d \
            --name "$name" \
            --gpus "\"device=${i}\"" \
            --shm-size="$NIM_SHM_SIZE" \
            -e NGC_API_KEY="${NGC_API_KEY:-}" \
            -e NIM_BOLTZ_STRUCTURE_OPTIMIZED_BACKEND=torch \
            -e NIM_BOLTZ_AFFINITY_OPTIMIZED_BACKEND=torch \
            -v "${NIM_CACHE}:/opt/nim/.cache" \
            -p "${port}:8000" \
            "$NIM_IMAGE"

        _NIM_CONTAINERS+=("$name")
        urls+=("http://localhost:${port}")
    done

    # Wait for all containers to become healthy
    for (( i=0; i<n_gpus; i++ )); do
        local port=$((NIM_BASE_PORT + i))
        local name="nim-boltz2-${SLURM_JOB_ID:-$$}-gpu${i}"
        _nim_wait_healthy "$name" "$port"
    done

    local nim_urls
    nim_urls=$(IFS=,; echo "${urls[*]}")
    export BOLTZ_BACKEND=nim
    export BOLTZ_NIM_URLS="$nim_urls"
    echo "[NIM] Exported BOLTZ_BACKEND=nim BOLTZ_NIM_URLS=$nim_urls"
}

nim_stop_all() {
    # Stop all NIM containers started by this job.
    nim_stop_single  # same logic — iterates _NIM_CONTAINERS
}
