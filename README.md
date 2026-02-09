# Dual-Target Docking

Comparing single-target vs dual-target RL molecule generation using **FREED++** (generator) and **Boltz-2** (scorer).

**Hypothesis:** RL optimized for worst-case dual-target docking (`max(d_A, d_B)`) produces molecules with higher dual hit-rate than single-target optimization.

## Requirements

- Python >= 3.11, [uv](https://docs.astral.sh/uv/) >= 0.9
- Docker >= 20.x + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- NVIDIA GPU >= 16 GB VRAM

## Quick Start

```bash
# 1. Install
git clone <repo-url> && cd SWW2026
uv sync --group dev

# 2. Build FREED++ Docker image
uv run python scripts/fetch/fetch_freed_data.py
docker build -f docker/freedpp/Dockerfile -t dualdocking/freedpp:latest .

# 3. Fetch all external data (internet required only here)
uv run python scripts/fetch/stage0_fetch.py

# 4. Calibrate normalization
uv run python scripts/validation/dock_calibration.py
uv run python scripts/validation/calibrate_normalization.py

# 5. Run full experiment (offline from here)
EXPERIMENT_RUN_ID=run01 uv run python scripts/run/run_full_experiment.py \
  --resume --parallel 2 --with-post --latex-mode draft
```

Results in `results/` and `report/`. See [Pipeline](docs/pipeline.md) for details.

### Smoke Test

```bash
uv run python scripts/run/generate.py --pair CDK2_ERK2 --mode single_a --epochs 2 --n-molecules 10
```

### Tests

```bash
uv run pytest tests/ -v
uv run ruff check src/ scripts/ docker/
```

## Project Structure

```
configs/              # pairs.yaml, metrics.yaml, experiment.yaml
src/dualdocking/      # Core library (scoring, generation, statistics, objectives)
scripts/              # CLI entry points (fetch, generate, score, analyze)
docker/freedpp/       # FREED++ Docker image + dual_reward_patch.py
tests/                # pytest suite (333 tests)
data/                 # Runtime data: targets/, calibration/, baseline/ (gitignored)
cache/                # Boltz-2 score cache (gitignored)
results/              # Experiment outputs (gitignored)
report/               # LaTeX manuscript, figures, metrics (partially generated)
docs/                 # Project documentation (see below)
```

## Documentation

| Document | Description |
|----------|-------------|
| [Setup](docs/setup.md) | Installation, Docker, calibration, baseline |
| [Pipeline](docs/pipeline.md) | Stage 0 -> generation -> scoring -> analysis -> report |
| [Architecture](docs/architecture.md) | Boltz-2, FREED++, caching, score conventions, troubleshooting |
| [Goal](docs/goal.md) | Hypothesis, notation, endpoints, success criteria |
| [Selection Criteria](docs/selection_criteria.md) | Target pair rationale, factorial design |
| [Pre-registration](docs/preregistration.md) | Frozen hypotheses and decision rules |
| [Statistical Analysis Plan](docs/statistical_analysis_plan.md) | Frozen SAP: inference, claim rules, code mapping |
| [Receptor QC](docs/receptor_qc.md) | Per-target structural quality control |
| [Doc Index](docs/README.md) | Source priority and conflict resolution |

## References

- **FREED++**: Telepov et al., TMLR 2023 — [arXiv:2401.09840](https://arxiv.org/abs/2401.09840)
- **Boltz-2**: Passaro et al., 2025 — [boltz-community](https://github.com/jwohlwend/boltz)
- **UniDock**: [dptech-corp/Uni-Dock](https://github.com/dptech-corp/Uni-Dock)
- **RDKit**: [rdkit.org](https://www.rdkit.org/)
