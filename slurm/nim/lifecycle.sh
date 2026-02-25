#!/usr/bin/env bash
# NIM Boltz-2 container lifecycle helpers.
#
# Source this file from sbatch scripts when DD_USE_NIM=1:
#   source "$DD_ROOT/slurm/nim/lifecycle.sh"
#
# Monolithic (multi-GPU):
#   nim_start_all <n_gpus>   # start N containers (one per GPU)
#   nim_stop_all             # stop all containers started by this job
#
# Array (single GPU):
#   nim_start_single         # start 1 container on one visible GPU
#   nim_stop_single          # stop the container

NIM_IMAGE="${NIM_IMAGE:-nvcr.io/nim/mit/boltz2:1.5.0}"
NIM_CACHE="${NIM_CACHE:-${DD_ROOT:-.}/docker/boltz2/cache}"
NIM_SHM_SIZE="${NIM_SHM_SIZE:-16G}"
NIM_HEALTH_TIMEOUT="${NIM_HEALTH_TIMEOUT:-300}"  # seconds
NIM_BASE_PORT="${NIM_BASE_PORT:-8000}"           # used when NIM_DYNAMIC_PORTS=0
NIM_DYNAMIC_PORTS="${NIM_DYNAMIC_PORTS:-1}"      # 1 = random free host ports

# Track container IDs for cleanup
_NIM_CONTAINERS=()
_NIM_PORTS=()

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

_nim_gpu_token() {
    # Resolve the physical GPU token for local index $1 from CUDA_VISIBLE_DEVICES.
    # Falls back to the local index when CUDA visibility is not constrained.
    local local_idx="$1"
    local cvd="${CUDA_VISIBLE_DEVICES:-}"

    if [[ -z "$cvd" || "${cvd,,}" == "all" || "$cvd" == "-1" ]]; then
        echo "$local_idx"
        return
    fi

    local IFS=','
    # shellcheck disable=SC2206
    local tokens=($cvd)
    if (( local_idx >= 0 && local_idx < ${#tokens[@]} )); then
        echo "${tokens[$local_idx]}"
        return
    fi
    echo "$local_idx"
}

_nim_host_port() {
    # Resolve mapped host port for container's 8000/tcp.
    local name="$1"
    local binding
    binding="$(docker port "$name" 8000/tcp 2>/dev/null | tail -n 1 || true)"
    if [[ -z "$binding" ]]; then
        return 1
    fi
    echo "${binding##*:}"
}

_nim_publish_arg() {
    # Return docker -p argument for NIM container.
    # Uses random free ports by default to avoid collisions across jobs/nodes.
    local requested_port="$1"
    if [[ "$NIM_DYNAMIC_PORTS" == "1" ]]; then
        echo "127.0.0.1::8000"
    else
        echo "127.0.0.1:${requested_port}:8000"
    fi
}

nim_start_single() {
    # Start a single NIM container on the first visible GPU.
    local local_gpu=0
    local gpu_token
    gpu_token="$(_nim_gpu_token "$local_gpu")"
    local name="nim-boltz2-${SLURM_JOB_ID:-$$}-${SLURM_ARRAY_TASK_ID:-0}"
    local publish
    publish="$(_nim_publish_arg "$NIM_BASE_PORT")"

    mkdir -p "$NIM_CACHE"

    docker run -d \
        --name "$name" \
        --gpus "device=${gpu_token}" \
        --shm-size="$NIM_SHM_SIZE" \
        -e NGC_API_KEY="${NGC_API_KEY:-}" \
        -e NIM_BOLTZ_STRUCTURE_OPTIMIZED_BACKEND=torch \
        -e NIM_BOLTZ_AFFINITY_OPTIMIZED_BACKEND=torch \
        -v "${NIM_CACHE}:/opt/nim/.cache" \
        -p "${publish}" \
        "$NIM_IMAGE"

    local port
    port="$(_nim_host_port "$name")"
    if [[ -z "$port" ]]; then
        echo "[NIM] ERROR: failed to resolve host port for $name" >&2
        return 1
    fi

    _NIM_CONTAINERS+=("$name")
    _NIM_PORTS+=("$port")
    _nim_wait_healthy "$name" "$port"

    export BOLTZ_BACKEND=nim
    export BOLTZ_NIM_URL="http://localhost:${port}"
    export BOLTZ_NIM_URLS="http://localhost:${port}"
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
    _NIM_PORTS=()
}

nim_start_all() {
    # Start N NIM containers (one per visible GPU index).
    # Args: $1 = number of GPUs
    local n_gpus="${1:-1}"
    mkdir -p "$NIM_CACHE"

    local urls=()
    for (( i=0; i<n_gpus; i++ )); do
        local requested_port=$((NIM_BASE_PORT + i))
        local name="nim-boltz2-${SLURM_JOB_ID:-$$}-gpu${i}"
        local publish
        publish="$(_nim_publish_arg "$requested_port")"
        local gpu_token
        gpu_token="$(_nim_gpu_token "$i")"

        docker run -d \
            --name "$name" \
            --gpus "device=${gpu_token}" \
            --shm-size="$NIM_SHM_SIZE" \
            -e NGC_API_KEY="${NGC_API_KEY:-}" \
            -e NIM_BOLTZ_STRUCTURE_OPTIMIZED_BACKEND=torch \
            -e NIM_BOLTZ_AFFINITY_OPTIMIZED_BACKEND=torch \
            -v "${NIM_CACHE}:/opt/nim/.cache" \
            -p "${publish}" \
            "$NIM_IMAGE"

        local port
        port="$(_nim_host_port "$name")"
        if [[ -z "$port" ]]; then
            echo "[NIM] ERROR: failed to resolve host port for $name" >&2
            return 1
        fi

        _NIM_CONTAINERS+=("$name")
        _NIM_PORTS+=("$port")
        urls+=("http://localhost:${port}")
    done

    # Wait for all containers to become healthy
    for (( i=0; i<n_gpus; i++ )); do
        local port="${_NIM_PORTS[$i]}"
        local name="${_NIM_CONTAINERS[$i]}"
        _nim_wait_healthy "$name" "$port"
    done

    local nim_urls
    nim_urls=$(IFS=,; echo "${urls[*]}")
    export BOLTZ_BACKEND=nim
    export BOLTZ_NIM_URLS="$nim_urls"
    export BOLTZ_NIM_URL="${urls[0]}"
    echo "[NIM] Exported BOLTZ_BACKEND=nim BOLTZ_NIM_URLS=$nim_urls"
}

nim_stop_all() {
    # Stop all NIM containers started by this job.
    nim_stop_single  # same logic — iterates _NIM_CONTAINERS
}
