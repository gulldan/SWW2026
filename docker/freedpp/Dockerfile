# FREED++ Docker image
# Isolates legacy environment (PyTorch 1.12, DGL 0.9, RDKit).
# Build: docker build -t dualdocking/freedpp:latest docker/freedpp/
# Requires: NVIDIA Container Toolkit for --gpus all

FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# ── System dependencies ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git build-essential libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

# ── uv ───────────────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy

# ── Python 3.10 + venv ──────────────────────────────────────────────
# DGL 0.9.1 has cp310 wheels on PyPI; PyTorch 1.12.1 supports 3.10.
RUN uv python install 3.10
RUN uv venv --python 3.10 /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── PyTorch 1.12 + CUDA 11.3 ────────────────────────────────────────
RUN uv pip install --no-cache \
        torch==1.12.1+cu113 \
        torchvision==0.13.1+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113

# ── DGL 0.9.1 + CUDA 11.3 ──────────────────────────────────────────────
RUN UV_HTTP_TIMEOUT=1200 uv pip install --no-cache dgl-cu113==0.9.1 -f https://data.dgl.ai/wheels/repo.html

# ── Remaining dependencies ───────────────────────────────────────────
RUN UV_HTTP_TIMEOUT=1200 uv pip install --no-cache \
        rdkit-pypi==2022.9.5 \
        numpy==1.23.5 \
        pandas==1.5.3 \
        scipy==1.10.1 \
        networkx==2.8.8 \
        tqdm \
        pyyaml \
        gym==0.26.2 \
        crem==0.2.9 \
        meeko==0.5.1

# moses (molsets) for metrics – skip pomegranate (Cython build fails on 3.10,
# and FREED never calls ScaffoldConstrainedGenerator which needs it).
RUN UV_HTTP_TIMEOUT=1200 uv pip install --no-cache --no-deps molsets==0.3.1 \
    && UV_HTTP_TIMEOUT=1200 uv pip install --no-cache fcd_torch>=1.0.5 matplotlib seaborn \
    && UV_HTTP_TIMEOUT=1200 uv pip install --no-cache tensorboardX

# ── Clone FFREED ─────────────────────────────────────────────────────
RUN git clone https://github.com/AIRI-Institute/FFREED.git /opt/freed \
    && cd /opt/freed \
    && git checkout $(git rev-parse HEAD)

# Ensure QuickVina2 binary is executable
RUN chmod +x /opt/freed/ffreed/env/qvina02 2>/dev/null || true

# ── Uni-Dock (GPU-accelerated docking) ────────────────────────────────
# CLI-compatible with Vina: --receptor, --ligand, --center_x/y/z, etc.
# unidock is only available via conda-forge, not PyPI.
RUN wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /usr/local bin/micromamba \
    && micromamba create -n unidock -c conda-forge unidock --yes --root-prefix /opt/micromamba \
    && micromamba install -n unidock -c conda-forge openmm pdbfixer --yes --root-prefix /opt/micromamba \
    && ln -sf /opt/micromamba/envs/unidock/bin/unidock /usr/local/bin/unidock \
    && rm -rf /opt/micromamba/pkgs

# ── Data files (not shipped with FFREED repo) ─────────────────────────
# Fragment library: 91 CReM fragments from ZINC-250K (AITRICS/FREED motifs_91.txt → JSON)
# Alert filters: PAINS / SureChEMBL / Glaxo rules (PatWalters/rd_filters)
COPY data/zinc_crem.json data/alert_collections.csv /opt/freed/data/

# ── Entrypoint & dual-target patch ──────────────────────────────────
COPY entrypoint.sh /opt/entrypoint.sh
COPY dual_reward_patch.py /opt/dual_reward_patch.py
COPY pdbfixer_add_h.py /opt/pdbfixer_add_h.py
RUN chmod +x /opt/entrypoint.sh

WORKDIR /opt/freed

ENTRYPOINT ["/opt/entrypoint.sh"]
