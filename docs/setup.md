# Setup

Detailed installation and data preparation instructions.

## Requirements

- Python >= 3.11 (tested on 3.12)
- [uv](https://docs.astral.sh/uv/) >= 0.9
- Docker >= 20.x with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) (`--gpus all`)
- NVIDIA GPU with >= 16 GB VRAM (RTX 4090 recommended)
- LaTeX distribution (optional, for PDF report): `pdflatex` + `bibtex`

## Python Environment

```bash
git clone <repo-url> && cd SWW2026

# Create venv and install all dependencies (including Boltz-2)
uv sync --group dev

# Verify
uv run pytest tests/ -v
uv run ruff check dualdocking/ cli/ docker/
```

Boltz-2 is installed from source via `pyproject.toml` (`boltz[cuda]` git dependency).
It is the default built-in scorer engine, but scoring can be replaced via
`configs/experiment.yaml` (`scorer.engine`).

## Docker (FREED++ Generator)

FREED++ requires Python 3.10 / PyTorch 1.12 / DGL 0.9 — incompatible with the main
environment, so it runs in an isolated Docker container.

```bash
# Fetch data files copied into the image
uv run python cli/s0_fetch/fetch_freed_fragments.py

# Build the image (from repo root)
docker build -f docker/freedpp/Dockerfile -t dualdocking/freedpp:latest .
```

Rebuild the image after any changes in `docker/freedpp/`.

The default built-in generator backend is `freedpp`, configurable via
`configs/experiment.yaml` (`generator.backend`).

## Calibration

Reward normalization in FREED++ requires per-target mean/std from a reference SMILES set.

```bash
# Download calibration SMILES (MOSES by default)
uv run python cli/s0_fetch/fetch_calibration_smiles.py

# Dock calibration set against all targets (GPU)
uv run python cli/s2_calibrate/dock_calibration.py

# Compute normalization.yaml
uv run python cli/s2_calibrate/compute_normalization.py
```

Parameters are in `configs/calibration.yaml`. Generated stats go to `configs/normalization.yaml`.

## Baseline ZINC (optional)

For baseline comparison with druglike molecules from ZINC:

```bash
# One-shot: fetch archive + convert to Parquet + sample 1000 SMILES
uv run python cli/s1_prepare/prepare_all.py \
  --fetch-baseline \
  --baseline-url "https://huggingface.co/datasets/Pixelatory/ZINC20-Druglike/resolve/main/zinc-druglike-cano.tar.xz?download=true" \
  --convert-baseline --sample-baseline
```

Or step by step:

```bash
uv run python cli/s0_fetch/fetch_zinc_baseline.py --url "https://<host>/zinc-druglike-cano.tar.xz"
uv run python cli/s1_prepare/convert_baseline_parquet.py \
  --input output/data/baseline/zinc-druglike-cano.tar.xz \
  --output output/data/baseline/zinc-druglike-cano.parquet --column smiles
uv run python cli/s1_prepare/sample_baseline.py \
  --input output/data/baseline/zinc-druglike-cano.parquet \
  --output output/data/baseline/zinc_druglike_baseline_1000_seed42.smi \
  --n 1000 --seed 42 --column smiles
```

## MD Validation Prerequisites (Optional)

Required only for `cli/s6_md/run_md.py` — post-hoc molecular dynamics
validation of top candidate molecules.

### Option A: Docker (Recommended)

The Docker image bundles GROMACS 2026.0 (CUDA), AmberTools, acpype, and OpenBabel
in a reproducible environment. No host installation required beyond Docker itself.

```bash
# Build the image (first time only, ~20-30 min — compiles GROMACS from source)
docker build -f docker/md/Dockerfile -t dualdocking/md:latest .

# Run MD via Docker (auto-builds image if missing)
uv run python cli/s6_md/run_md.py --pair CDK2_ERK2 --docker

# Dry-run (no build needed)
uv run python cli/s6_md/run_md.py --pair CDK2_ERK2 --docker --dry-run

# Custom image tag
uv run python cli/s6_md/run_md.py --pair CDK2_ERK2 --docker --docker-image myrepo/md:v2

# Override simulation time, single mode
uv run python cli/s6_md/run_md.py --pair CDK2_ERK2 --docker \
  --mode single_a --top-n 1 --simulation-time 10

# With MM-GBSA
uv run python cli/s6_md/run_md.py --pair CDK2_ERK2 --docker --mmpbsa
```

The `--docker` flag bind-mounts the project directories into the container:

| Host | Container | Access |
|------|-----------|--------|
| `output/` | `/workspace/output` | read-write |
| `configs/` | `/workspace/configs` | read-only |
| `dualdocking/` | `/workspace/dualdocking` | read-only |
| `cli/` | `/workspace/cli` | read-only |
| `pyproject.toml` | `/workspace/pyproject.toml` | read-only |

To rebuild the image with a different CUDA architecture (default `89` = RTX 4090):

```bash
docker build -f docker/md/Dockerfile -t dualdocking/md:latest \
  --build-arg CUDA_ARCHITECTURES=80 .   # e.g. 80 for A100
```

### Option B: Local Installation

Install all tools on the host system. This is more complex but avoids Docker overhead.

#### Tools Required

| Tool | Purpose | Version |
|------|---------|---------|
| [GROMACS](https://manual.gromacs.org/current/install-guide/index.html) | MD engine | >= 2024 (2026.x recommended) |
| [AmberTools](https://ambermd.org/AmberTools.php) | Antechamber (AM1-BCC charges) | >= 23 |
| [acpype](https://github.com/alanwilter/acpype) | AMBER → GROMACS topology | >= 2023 |
| [OpenBabel](https://openbabel.org/) | SDF → MOL2 conversion | >= 3.1 |
| [gmx_MMPBSA](https://valdes-tresanco-ms.github.io/gmx_MMPBSA/v1.6.1/installation/) | MM-GBSA binding ΔG (optional) | >= 1.6 |

#### GROMACS (Build from Source)

GROMACS must be built from source for GPU (CUDA) support.
Full guide: https://manual.gromacs.org/current/install-guide/index.html

Build dependencies (Arch Linux):

```bash
sudo pacman -S --needed cmake gcc gcc-fortran fftw hwloc
```

Build dependencies (Ubuntu/Debian):

```bash
sudo apt install cmake g++ gfortran libfftw3-dev libhwloc-dev
```

Build and install:

```bash
wget https://ftp.gromacs.org/gromacs/gromacs-2026.0.tar.gz
tar xzf gromacs-2026.0.tar.gz && cd gromacs-2026.0
mkdir build && cd build
cmake .. -DGMX_GPU=CUDA -DGMX_BUILD_OWN_FFTW=ON -DCMAKE_INSTALL_PREFIX=/opt/gromacs
make -j$(nproc) && sudo make install
source /opt/gromacs/bin/GMXRC
```

Add `source /opt/gromacs/bin/GMXRC` to your shell profile (`~/.bashrc` or `~/.zshrc`)
so that `gmx` is always in `PATH`.

For CPU-only builds, omit `-DGMX_GPU=CUDA`.

#### acpype + AmberTools

**conda-forge (recommended for non-Ubuntu systems):**

```bash
micromamba create -n ambertools -c conda-forge ambertools acpype openbabel
micromamba activate ambertools
```

**pip (Ubuntu x86_64 only — bundled binaries may be incompatible with other distros):**

```bash
uv pip install acpype
```

This provides both `acpype` and `antechamber` in a single install. Verify:

```bash
acpype --help
antechamber --help
```

If you need the full AmberTools suite (not just the binaries bundled with acpype),
download the source from https://ambermd.org/AmberTools.php (free, registration required)
and compile:

```bash
tar xjf AmberTools25.tar.bz2 && cd amber25_src
./configure --no-updates gnu
make -j$(nproc) install
source amber25/amber.sh
```

#### OpenBabel

On Arch Linux:

```bash
sudo pacman -S openbabel
```

On Ubuntu/Debian:

```bash
sudo apt install openbabel
```

To build from source (if system packages are unavailable or outdated):

```bash
git clone https://github.com/openbabel/openbabel.git && cd openbabel
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc) && sudo make install
```

#### gmx_MMPBSA (optional, pip)

Only needed when running with `--mmpbsa` flag for MM-GBSA binding free energy:

```bash
uv pip install gmx-MMPBSA
```

#### Verification

```bash
# All tools must be in PATH
gmx --version           # GROMACS >= 2024
antechamber --help      # AmberTools (antechamber for charge assignment)
acpype --help           # AMBER → GROMACS conversion
obabel --version        # OpenBabel >= 3.1

# Optional
gmx_MMPBSA --help      # MM-GBSA (only needed with --mmpbsa flag)
```

### Usage

```bash
# Docker mode (recommended — no local tools needed):
uv run python cli/s6_md/run_md.py --pair CDK2_ERK2 --docker

# Local mode (requires all tools in PATH):
uv run python cli/s6_md/run_md.py --pair CDK2_ERK2

# Dry-run (works in both modes)
uv run python cli/s6_md/run_md.py --pair CDK2_ERK2 --dry-run

# Custom simulation time, disable GPU
uv run python cli/s6_md/run_md.py --pair CDK2_ERK2 --simulation-time 100 --no-gpu

# With MM-GBSA binding free energy
uv run python cli/s6_md/run_md.py --pair CDK2_ERK2 --mmpbsa
```

See [io_contracts.md](io_contracts.md) (Stage 5c) for complete output specification.

## LaTeX (optional)

For PDF report generation. On Arch Linux:

```bash
sudo pacman -S --needed texlive-core texlive-latexextra texlive-bibtexextra
```
