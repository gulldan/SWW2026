#!/usr/bin/env bash
# install_md_tools.sh — Install MD validation prerequisites (GROMACS, acpype, OpenBabel)
#
# Usage:
#   bash cli/s0_fetch/install_md_tools.sh [--prefix DIR] [--cuda-arch SM] [--no-gpu] [--with-mmpbsa]
#
# Options:
#   --prefix DIR       Install GROMACS here (default: ~/.local/gromacs)
#   --cuda-arch SM     CUDA compute capability (default: auto-detect)
#   --no-gpu           Build GROMACS without GPU support
#   --with-mmpbsa      Also install gmx_MMPBSA via pip
#
# Prerequisites (must already be installed):
#   - cmake >= 3.28, g++ >= 11, make
#   - CUDA toolkit >= 12.1 (unless --no-gpu)
#   - uv (Python package manager)
#   - obabel (OpenBabel CLI — install via system package manager)
#   - arpack (libarpack — required by AmberTools sqm for AM1-BCC charges)
#
# Example:
#   bash cli/s0_fetch/install_md_tools.sh --prefix /opt/gromacs --cuda-arch 89
#   source ~/.local/gromacs/bin/GMXRC  # or /opt/gromacs/bin/GMXRC

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────
GROMACS_VERSION="2026.0"
GROMACS_URL="https://ftp.gromacs.org/gromacs/gromacs-${GROMACS_VERSION}.tar.gz"
PREFIX="${HOME}/.local/gromacs"
CUDA_ARCH=""
GPU=1
WITH_MMPBSA=0
BUILD_DIR="/tmp/gromacs-build-$$"

# ── Parse arguments ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)     PREFIX="$2"; shift 2 ;;
        --cuda-arch)  CUDA_ARCH="$2"; shift 2 ;;
        --no-gpu)     GPU=0; shift ;;
        --with-mmpbsa) WITH_MMPBSA=1; shift ;;
        -h|--help)
            head -20 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

log() { echo "[install_md_tools] $*"; }
err() { echo "[install_md_tools] ERROR: $*" >&2; exit 1; }

# ── Check prerequisites ──────────────────────────────────────────────
log "Checking prerequisites..."

command -v cmake >/dev/null || err "cmake not found. Install: sudo pacman -S cmake (Arch) or sudo apt install cmake (Debian)"
command -v g++ >/dev/null   || err "g++ not found. Install: sudo pacman -S gcc (Arch) or sudo apt install g++ (Debian)"
command -v make >/dev/null  || err "make not found. Install: sudo pacman -S make (Arch) or sudo apt install make (Debian)"
command -v uv >/dev/null    || err "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
command -v obabel >/dev/null || err "obabel not found. Install: sudo pacman -S openbabel (Arch) or sudo apt install openbabel (Debian)"

# arpack is required by AmberTools sqm (semi-empirical QM for AM1-BCC charges)
if ! ldconfig -p 2>/dev/null | grep libarpack >/dev/null; then
    if [[ -f /etc/arch-release ]]; then
        err "libarpack not found. Install: sudo pacman -S arpack"
    elif [[ -f /etc/debian_version ]]; then
        err "libarpack not found. Install: sudo apt install libarpack2-dev"
    else
        err "libarpack not found. Install arpack from your package manager."
    fi
fi

if [[ $GPU -eq 1 ]]; then
    command -v nvcc >/dev/null || err "nvcc not found. Install CUDA toolkit >= 12.1 or use --no-gpu"
fi

CMAKE_VER=$(cmake --version | head -1 | grep -oP '\d+\.\d+')
log "cmake $CMAKE_VER, g++ $(g++ -dumpversion)"

# ── Auto-detect CUDA architecture ────────────────────────────────────
if [[ $GPU -eq 1 && -z "$CUDA_ARCH" ]]; then
    if command -v nvidia-smi >/dev/null; then
        # Query compute capability from nvidia-smi
        CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits --id=0 2>/dev/null | tr -d '.' | head -1)
        if [[ -n "$CUDA_ARCH" ]]; then
            log "Auto-detected CUDA architecture: sm_${CUDA_ARCH}"
        fi
    fi
    if [[ -z "$CUDA_ARCH" ]]; then
        log "Could not auto-detect CUDA architecture, letting cmake decide"
    fi
fi

# ── Step 1: Install acpype (includes antechamber) ────────────────────
log "Installing acpype via uv pip..."
uv pip install acpype 2>&1 | tail -3

# acpype bundles AmberTools binaries but they need wrapper scripts to set
# AMBERHOME correctly. Create wrappers in the venv bin directory.
VENV_BIN="$(uv run python -c 'import sys; print(sys.prefix)')"/bin
AMBER_DIR="$(uv run python -c 'import acpype, pathlib; print(pathlib.Path(acpype.__file__).parent)')"
if [[ "$(uname)" == "Darwin" ]]; then
    AMBER_DIR="${AMBER_DIR}/amber_macos"
else
    AMBER_DIR="${AMBER_DIR}/amber_linux"
fi

for bin in antechamber parmchk2 sqm am1bcc tleap; do
    cat > "${VENV_BIN}/${bin}" << WRAPPER
#!/bin/bash
source "${AMBER_DIR}/amber.sh"
exec "${AMBER_DIR}/bin/wrapped_progs/${bin}" "\$@"
WRAPPER
    chmod +x "${VENV_BIN}/${bin}"
done
log "AmberTools wrappers created in ${VENV_BIN}"
log "acpype installed: $(uv run acpype --version 2>&1 | head -1 || echo 'ok')"

# ── Step 2: Install gmx_MMPBSA (optional) ───────────────────────────
if [[ $WITH_MMPBSA -eq 1 ]]; then
    log "Installing gmx_MMPBSA via uv pip..."
    uv pip install gmx-MMPBSA 2>&1 | tail -3
fi

# ── Step 3: Check if GROMACS already installed ───────────────────────
if [[ -x "${PREFIX}/bin/gmx" ]]; then
    EXISTING_VER=$("${PREFIX}/bin/gmx" --version 2>&1 | grep "GROMACS version" | head -1 || true)
    log "GROMACS already installed at ${PREFIX}: ${EXISTING_VER}"
    log "To reinstall, remove ${PREFIX} first."
    log ""
    log "Done. Source the GROMACS environment:"
    log "  source ${PREFIX}/bin/GMXRC"
    exit 0
fi

# ── Step 4: Download GROMACS ─────────────────────────────────────────
TARBALL="/tmp/gromacs-${GROMACS_VERSION}.tar.gz"
if [[ -f "$TARBALL" ]]; then
    log "Using cached tarball: $TARBALL"
else
    log "Downloading GROMACS ${GROMACS_VERSION}..."
    wget -q --show-progress "$GROMACS_URL" -O "$TARBALL"
fi

# ── Step 5: Extract and configure ────────────────────────────────────
log "Extracting..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
tar xzf "$TARBALL" -C "$BUILD_DIR" --strip-components=1

cd "$BUILD_DIR"
mkdir -p build && cd build

CMAKE_ARGS=(
    -DCMAKE_INSTALL_PREFIX="$PREFIX"
    -DGMX_BUILD_OWN_FFTW=ON
)

if [[ $GPU -eq 1 ]]; then
    CMAKE_ARGS+=(-DGMX_GPU=CUDA)
    # Find CUDA toolkit
    if [[ -d /opt/cuda ]]; then
        CMAKE_ARGS+=(-DCUDAToolkit_ROOT=/opt/cuda)
    elif [[ -d /usr/local/cuda ]]; then
        CMAKE_ARGS+=(-DCUDAToolkit_ROOT=/usr/local/cuda)
    fi
    if [[ -n "$CUDA_ARCH" ]]; then
        CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
    fi
else
    log "Building without GPU support (--no-gpu)"
fi

log "Configuring GROMACS ${GROMACS_VERSION}..."
cmake .. "${CMAKE_ARGS[@]}" 2>&1 | tail -5

# ── Step 6: Build and install ────────────────────────────────────────
NPROC=$(nproc 2>/dev/null || echo 4)
log "Building with ${NPROC} cores (this takes 5-15 minutes)..."
make -j"$NPROC" 2>&1 | tail -3

log "Installing to ${PREFIX}..."
make install 2>&1 | tail -3

# ── Step 7: Clean up ────────────────────────────────────────────────
rm -rf "$BUILD_DIR"
log "Build directory cleaned up."

# ── Step 8: Verify ──────────────────────────────────────────────────
log ""
log "============================================"
log "  Installation complete!"
log "============================================"
log ""
log "Add to your shell profile (~/.bashrc or ~/.zshrc):"
log "  source ${PREFIX}/bin/GMXRC"
log ""
log "Then verify:"
log "  gmx --version"
log "  acpype --help"
log "  obabel --version"
if [[ $WITH_MMPBSA -eq 1 ]]; then
    log "  gmx_MMPBSA --help"
fi
log ""
log "Run MD validation:"
log "  uv run python cli/s6_md/run_md.py --pair CDK2_ERK2 --dry-run"
