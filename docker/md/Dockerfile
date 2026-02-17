# MD validation Docker image
# GROMACS 2026.0 (CUDA) + AmberTools (conda-forge) + acpype + OpenBabel
# Build (from repo root): docker build -f docker/md/Dockerfile -t dualdocking/md:latest .
# Requires: NVIDIA Container Toolkit for --gpus all

# ── Stage 1: Build GROMACS from source with CUDA ─────────────────────
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS gromacs-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ wget ca-certificates gpg \
    && rm -rf /var/lib/apt/lists/*

# CMake >= 3.28 required by GROMACS 2026.0 (Ubuntu 22.04 ships 3.22)
RUN wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc \
        | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main" \
        > /etc/apt/sources.list.d/kitware.list \
    && apt-get update && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*

# Download and build GROMACS 2026.0
ARG GROMACS_VERSION=2026.0
ARG CUDA_ARCHITECTURES=89
RUN wget -q "https://ftp.gromacs.org/gromacs/gromacs-${GROMACS_VERSION}.tar.gz" -O /tmp/gromacs.tar.gz \
    && mkdir /tmp/gromacs-src && tar -xzf /tmp/gromacs.tar.gz -C /tmp/gromacs-src --strip-components=1 \
    && mkdir /tmp/gromacs-build && cd /tmp/gromacs-build \
    && cmake /tmp/gromacs-src \
        -DGMX_GPU=CUDA \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DGMX_BUILD_OWN_FFTW=ON \
        -DCMAKE_INSTALL_PREFIX=/opt/gromacs \
        -DGMX_DEFAULT_SUFFIX=OFF \
        -DGMX_BINARY_SUFFIX="" \
        -DGMX_LIBS_SUFFIX="" \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf /tmp/gromacs*


# ── Stage 2: Runtime image ───────────────────────────────────────────
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget ca-certificates bzip2 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy GROMACS from builder
COPY --from=gromacs-builder /opt/gromacs /opt/gromacs

# Install micromamba for conda-forge packages
RUN wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /usr/local bin/micromamba

# AmberTools + acpype + OpenBabel + PDBFixer via conda-forge (pinned for reproducibility)
RUN MAMBA_NO_LOW_SPEED_LIMIT=1 micromamba create -n ambertools \
        -c conda-forge \
        "ambertools=24.8" "openbabel=3.1.1" "acpype=2023.10.27" \
        pdbfixer \
        --yes --root-prefix /opt/micromamba --retry-clean-cache \
    && rm -rf /opt/micromamba/pkgs

# Symlink conda tools to /usr/local/bin for easy PATH access
RUN for tool in antechamber parmchk2 sqm tleap acpype obabel; do \
        ln -sf "/opt/micromamba/envs/ambertools/bin/$tool" "/usr/local/bin/$tool"; \
    done

# ── uv + Python venv ────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy

RUN uv python install 3.12 \
    && uv venv --python 3.12 /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Python deps (subset needed by md.py — pinned for reproducibility)
RUN uv pip install --no-cache \
        "rdkit==2025.9.4" \
        "numpy==2.4.2" \
        "pandas==3.0.0" \
        "pyarrow==23.0.0" \
        "pyyaml==6.0.3" \
        "scipy==1.17.0" \
        "loguru==0.7.3" \
        "click==8.3.1"

# ── Entrypoint ──────────────────────────────────────────────────────
COPY docker/md/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/opt/entrypoint.sh"]
