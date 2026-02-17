# FREED++ Docker image
# Environment: Python 3.12, PyTorch 2.4, DGL 2.4, CUDA 12.4, RDKit (H100/sm_90 native).
# Build (from repo root): docker build -f docker/freedpp/Dockerfile -t dualdocking/freedpp:latest .
# Requires: NVIDIA Container Toolkit for --gpus all

FROM nvidia/cuda:12.4.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# ── System dependencies ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git build-essential libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

# ── uv ───────────────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy

# ── Python 3.12 + venv ──────────────────────────────────────────────
RUN uv python install 3.12
RUN uv venv --python 3.12 /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── PyTorch 2.4 + CUDA 12.4 (sm_90 / H100 native) ─────────────────
RUN uv pip install --no-cache \
        torch==2.4.0 \
        torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cu124

# ── DGL 0.9.1 + CUDA 12.4 ──────────────────────────────────────────────
RUN UV_HTTP_TIMEOUT=1200 uv pip install --no-cache dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

# ── Remaining dependencies ───────────────────────────────────────────
RUN UV_HTTP_TIMEOUT=1200 uv pip install --no-cache \
        rdkit \
        numpy==1.26.4 \
        "pandas==2.2.3" \
        scipy==1.14.1 \
        networkx \
        tqdm \
        pyyaml \
        gymnasium==0.29.1 \
        crem \
        "meeko==0.5.1"

RUN UV_HTTP_TIMEOUT=1200 uv pip install --no-cache --no-deps molsets==0.3.1 \
    && UV_HTTP_TIMEOUT=1200 uv pip install --no-cache fcd_torch>=1.0.5 matplotlib seaborn \
    && UV_HTTP_TIMEOUT=1200 uv pip install --no-cache tensorboardX
# 4) moses/sascorer: rdkit.six removed in modern RDKit (was a Python 2/3 shim)
RUN sed -i 's/from rdkit.six import iteritems/iteritems = lambda d: d.items()/' \
    /opt/venv/lib/python3.12/site-packages/moses/metrics/SA_Score/sascorer.py
# 5) moses/utils: DataFrame.append removed in pandas 2.x → pd.concat
RUN sed -i 's/_mcf\.append(_pains, sort=True)/pd.concat([_mcf, _pains], sort=True)/' \
    /opt/venv/lib/python3.12/site-packages/moses/metrics/utils.py

# ── Clone FFREED ─────────────────────────────────────────────────────
RUN git clone https://github.com/AIRI-Institute/FFREED.git /opt/freed \
    && cd /opt/freed \
    && git checkout $(git rev-parse HEAD)

# ── Patch FREED for modern deps ──────────────────────────────────────
# 1) DGL 2.x: copy_src(src=, out=) → copy_u(u, out) with positional args
RUN sed -i "s/fn\.copy_src(src='\([^']*\)', out='\([^']*\)')/fn.copy_u('\1', '\2')/g" /opt/freed/ffreed/train/nn/gcn.py
# 2) gymnasium: gym 0.26 replaced by gymnasium 0.29 (numpy 1.24+ broke np.bool)
RUN find /opt/freed -name '*.py' -exec sed -i \
        -e 's/^import gym$/import gymnasium as gym/' \
        -e 's/^from gym\b/from gymnasium/' {} +
# 3) numpy 2.x compat: np.bool → bool (replay_buffer.py)
RUN find /opt/freed -name '*.py' -exec sed -i \
        -e 's/np\.bool\b/bool/g' \
        -e 's/np\.int\b/int/g' \
        -e 's/np\.float\b/float/g' {} +

# Ensure QuickVina2 binary is executable
RUN chmod +x /opt/freed/ffreed/env/qvina02 2>/dev/null || true

# ── Uni-Dock (GPU-accelerated docking) ────────────────────────────────
# CLI-совместим с AutoDock Vina: --receptor, --ligand, --center_x/y/z, etc.
# unidock is only available via conda-forge, not PyPI.
RUN wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /usr/local bin/micromamba \
    && MAMBA_NO_LOW_SPEED_LIMIT=1 micromamba create -n unidock -c conda-forge unidock --yes --root-prefix /opt/micromamba --retry-clean-cache \
    && MAMBA_NO_LOW_SPEED_LIMIT=1 micromamba install -n unidock -c conda-forge openmm pdbfixer --yes --root-prefix /opt/micromamba \
    && ln -sf /opt/micromamba/envs/unidock/bin/unidock /usr/local/bin/unidock \
    && rm -rf /opt/micromamba/pkgs

# ── Data files (not shipped with FFREED repo) ─────────────────────────
# Fragment library: 91 CReM fragments from ZINC-250K (AITRICS/FREED motifs_91.txt → JSON)
# Alert filters: PAINS / SureChEMBL / Glaxo rules (PatWalters/rd_filters)
COPY docker/freedpp/data/zinc_crem.json docker/freedpp/data/alert_collections.csv /opt/freed/data/
# ── Entrypoint & dual-target patch ──────────────────────────────────
COPY docker/freedpp/entrypoint.sh /opt/entrypoint.sh
COPY dualdocking/data/mol_prep.py /opt/mol_prep.py
COPY dualdocking/reward/selectivity.py /opt/reward_selectivity.py
COPY configs/reward_modes.yaml /opt/reward_modes.yaml
COPY docker/freedpp/dual_reward_patch.py /opt/dual_reward_patch.py
COPY docker/freedpp/run_with_patch.py /opt/run_with_patch.py
COPY docker/freedpp/pdbfixer_add_h.py /opt/pdbfixer_add_h.py
RUN chmod +x /opt/entrypoint.sh

WORKDIR /opt/freed

ENTRYPOINT ["/opt/entrypoint.sh"]
