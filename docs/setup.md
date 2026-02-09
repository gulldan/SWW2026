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
uv run ruff check src/ scripts/ docker/
```

Boltz-2 is installed from source via `pyproject.toml` (`boltz[cuda]` git dependency).

## Docker (FREED++ Generator)

FREED++ requires Python 3.10 / PyTorch 1.12 / DGL 0.9 — incompatible with the main
environment, so it runs in an isolated Docker container.

```bash
# Fetch data files copied into the image
uv run python scripts/fetch/fetch_freed_data.py

# Build the image (from repo root)
docker build -f docker/freedpp/Dockerfile -t dualdocking/freedpp:latest .
```

Rebuild the image after any changes in `docker/freedpp/`.

## Calibration

Reward normalization in FREED++ requires per-target mean/std from a reference SMILES set.

```bash
# Download calibration SMILES (MOSES by default)
uv run python scripts/fetch/fetch_calibration_sets.py

# Dock calibration set against all targets (GPU)
uv run python scripts/validation/dock_calibration.py

# Compute normalization.yaml
uv run python scripts/validation/calibrate_normalization.py
```

Parameters are in `configs/calibration.yaml`. Generated stats go to `configs/normalization.yaml`.

## Baseline ZINC (optional)

For baseline comparison with druglike molecules from ZINC:

```bash
# One-shot: fetch archive + convert to Parquet + sample 1000 SMILES
uv run python scripts/prep/prepare_data_dependencies.py \
  --fetch-baseline \
  --baseline-url "https://huggingface.co/datasets/Pixelatory/ZINC20-Druglike/resolve/main/zinc-druglike-cano.tar.xz?download=true" \
  --convert-baseline --sample-baseline
```

Or step by step:

```bash
uv run python scripts/fetch/fetch_baseline_zinc.py --url "https://<host>/zinc-druglike-cano.tar.xz"
uv run python scripts/prep/convert_baseline_to_parquet.py \
  --input data/baseline/zinc-druglike-cano.tar.xz \
  --output data/baseline/zinc-druglike-cano.parquet --column smiles
uv run python scripts/prep/sample_baseline_from_parquet.py \
  --input data/baseline/zinc-druglike-cano.parquet \
  --output data/baseline/zinc_druglike_baseline_1000_seed42.smi \
  --n 1000 --seed 42 --column smiles
```

## LaTeX (optional)

For PDF report generation. On Arch Linux:

```bash
sudo pacman -S --needed texlive-core texlive-latexextra texlive-bibtexextra
```
