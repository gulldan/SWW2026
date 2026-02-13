# Dual-Target Docking

Single-target vs dual-target RL molecule generation using **FREED++** (generator) and **Boltz-2** (scorer).

**Hypothesis.** RL optimized for worst-case dual-target docking (`max(d_A, d_B)`) produces molecules with higher dual hit-rate than single-target optimization, evaluated by an orthogonal Boltz-2 affinity surrogate.

---

## Data Flow

```
Stage 0: Fetch                   Generation (FREED++)               Scoring (Boltz-2)
---------------------     ----------------------------      --------------------------
 PDB structures                                               Boltz-2 affinity (a_A, a_B)
 UniProt FASTA          ->  For each of 5 core pairs:    ->   per molecule, both targets
 ZINC calibration sets       3 modes x 5 seeds                 (log10 IC50-like uM)
 ChEMBL actives              - single_a: reward=f(d_A)
                              - single_b: reward=f(d_B)         |
                              - double:   reward=f(max(d_A,d_B))  v

                              UniDock GPU docking             Analysis & Report
                              (d_A, d_B in kcal/mol)     --------------------------
                                                           Dual hit-rate comparison
                                                           Bootstrap CI (m=5 seeds)
                                                           Cliff's delta effect size
                                                           LaTeX manuscript + figures
```

**75 runs total**: 5 core pairs x 3 modes x 5 seeds. Auxiliary pairs (3) for robustness checks.

---

## Project Structure

```
SWW2026/
|-- dualdocking/                Core Python library
|   |-- config.py               YAML config loading & validation (pairs, metrics, experiment)
|   |-- scoring/                Scoring & docking objectives
|   |   |-- objectives.py       Multi-objective scalarizations (min, harmonic, Tchebycheff, Pareto, ...)
|   |   |-- backends.py         Scoring backend registry (builtin boltz2 + external engines)
|   |   |-- boltz2.py           Boltz-2 affinity scorer with disk cache + NIM backend
|   |   |-- cache_io.py         Process-safe JSON cache (fcntl.flock + atomic temp-and-replace)
|   |   +-- binding_box.py      PDB binding-site extraction from co-crystallized ligands
|   |-- generation/             FREED++ molecule generation
|   |   |-- backends.py         Generation backend registry (builtin freedpp + external backends)
|   |   +-- freedpp.py          Docker subprocess wrapper (GenerationConfig, generate())
|   |-- analysis/               Post-hoc statistical analysis
|   |   |-- statistical.py      Hierarchical bootstrap, FDR correction, Cliff's delta
|   |   |-- fairness.py         Generation budget and duplication checks
|   |   |-- enrichment.py       AUROC, BEDROC, enrichment factor for docking validation
|   |   +-- diversity.py        Uniqueness, scaffold diversity, Tanimoto metrics
|   |-- data/                   Molecular data ingestion and cleaning
|   |   |-- chem_sanitize.py    SMILES validation, canonicalization & deduplication
|   |   |-- baseline.py         ZINC calibration baseline stability checks
|   |   +-- chembl_validation.py ChEMBL active retrieval, dual-active overlap, enrichment
|   +-- reporting/              Experiment provenance and manuscript
|       |-- run_manifest.py     Run lifecycle tracking (JSON manifest with atomic I/O)
|       +-- latex_report.py     Jinja2-to-LaTeX rendering and PDF compilation
|
|-- scripts/                    CLI entry points (organized by pipeline stage)
|   |-- s0_fetch/               Data download (PDB, FASTA, FREED fragments, ZINC, ChEMBL)
|   |-- s1_prepare/             Target preparation (PDB -> PDBQT, binding box, MSA)
|   |-- s2_calibrate/           Calibration and validation (dock calibration, normalization, redocking)
|   |-- s3_generate/            Molecule generation (configured backend; default FREED++)
|   |-- s4_score/               Affinity scoring (configured engine; default Boltz-2)
|   |-- s5_analyze/             Statistical analysis (aggregate, metrics, correlation)
|   |-- s6_report/              Visualization and manuscript (plots, molecules, LaTeX)
|   |-- s7_audit/               Post-hoc verification (claim traceability)
|   |-- pipeline/               End-to-end orchestrators (full experiment, pilot, post)
|   +-- smoke/                  Smoke tests (Boltz-2 health check, reward capping)
|
|-- configs/                    YAML configuration (pairs, metrics, experiment, normalization)
|-- tests/                      pytest suite (~400 tests)
|-- docker/freedpp/             FREED++ Docker image + dual_reward_patch.py
|-- docs/                       Project documentation (see Documentation Map below)
|-- report/                     LaTeX manuscript (chapters/, figures/, templates/)
+-- output/                     Runtime data (gitignored)
    |-- data/                   Target structures, calibration sets, baseline
    |-- cache/                  Score cache + MSA cache (backend-dependent)
    +-- results/                Experiment outputs
```

---

## Requirements

| Requirement | Minimum |
|-------------|---------|
| Python | >= 3.11 |
| [uv](https://docs.astral.sh/uv/) | >= 0.9 |
| Docker | >= 20.x with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) |
| GPU | NVIDIA, >= 16 GB VRAM (24 GB recommended for Boltz-2) |
| OS | Linux (fcntl-based file locking) |

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd SWW2026
uv sync --group dev

# 2. Build FREED++ Docker image
uv run python scripts/s0_fetch/fetch_freed_fragments.py
docker build -f docker/freedpp/Dockerfile -t dualdocking/freedpp:latest .

# 3. Fetch all external data (internet required only for this step)
uv run python scripts/s0_fetch/fetch_all.py

# 4. Calibrate normalization (per-target z-score parameters)
uv run python scripts/s2_calibrate/dock_calibration.py
uv run python scripts/s2_calibrate/compute_normalization.py

# 5. Run full experiment (offline from here)
EXPERIMENT_RUN_ID=run01 uv run python scripts/pipeline/run_full_experiment.py \
  --resume --parallel 1 --with-post --latex-mode draft
```

Results land in `output/results/` and `report/`. See [Pipeline](docs/pipeline.md) for stage-by-stage details.

### Smoke Test

Verify the setup works end-to-end with a minimal run:

```bash
# Generate 10 molecules for one pair, 2 epochs (takes ~2 minutes with GPU)
uv run python scripts/s3_generate/generate.py --pair CDK2_ERK2 --mode single_a --epochs 2 --n-molecules 10

# Boltz-2 health check
uv run dd-smoke
```

---

## CLI Commands

All commands are installed as console scripts via `uv sync`:

| Command | Description |
|---------|-------------|
| `dd-prepare` | Prepare target structures (download PDB, extract binding boxes, generate PDBQT) |
| `dd-generate` | Run molecule generation for a specific pair/mode (configured backend) |
| `dd-score` | Score generated molecules with the configured scoring engine (default Boltz-2) |
| `dd-aggregate` | Aggregate results across seeds and modes into summary tables |
| `dd-metrics` | Compute evaluation metrics (hit-rates, Cliff's delta, bootstrap CI) |
| `dd-summarize` | Generate text summaries of experiment results |
| `dd-plot` | Generate result plots (violin, heatmap, Pareto front) |
| `dd-render` | Render 2D molecule images for top hits |
| `dd-smoke` | Boltz-2 smoke test (health check and single-molecule scoring) |

Run any command with `--help` for usage details, e.g. `uv run dd-generate --help`.

## Python API (non-CLI)

You can run workflows directly from Python (not only from shell CLI):

```python
from dualdocking.api import run_full_experiment, run_post_pipeline, run_stages

# Full pipeline dry-run for one pair/seed
run_full_experiment(dry_run=True, pairs=["CDK2_ERK2"], seeds=[42], with_post=False)

# Run selected stages
run_stages(stages="analyze,report", dry_run=True, skip_preflight=True)

# Post-processing only
run_post_pipeline(dry_run=True, latex_mode="off")
```

## Backend Configuration

Generator and scorer are configurable in `configs/experiment.yaml`.

```yaml
generator:
  # Legacy key also supported: name: "FREED++"
  backend: freedpp  # built-in alias; or package.module[:callable]

scorer:
  # Legacy key also supported: name: "Boltz-2"
  engine: boltz2    # built-in alias; or package.module[:factory_or_object]
  backend: nim      # boltz2 transport: cli | nim
```

Default behavior remains unchanged:
- Generator: built-in `freedpp`
- Scorer: built-in `boltz2` (`cli` or `nim` based on config)

---

## Key Concepts

### Notation

The project uses two independent scoring systems with distinct roles:

| Symbol | Source | Units | Role |
|--------|--------|-------|------|
| `d_A`, `d_B` | UniDock GPU (docking) | kcal/mol, lower-is-better | **RL reward** -- drives molecule generation |
| `a_A`, `a_B` | Boltz-2 (learned affinity) | log10(IC50-like, uM), lower-is-better | **Evaluation surrogate** -- orthogonal assessment |

The two scorers are never mixed: FREED++ optimizes only `d_A, d_B`; Boltz-2 evaluates post-hoc.

### Score Convention

**lower_is_better = True** throughout the codebase. Docking scores are negative (more negative = better binding). Utility = -score when lower_is_better is True.

### Generation Modes

**Pre-registered (primary):**

- **single_a**: RL reward = `-z_A` — optimize docking to target A only
- **single_b**: RL reward = `-z_B` — optimize docking to target B only
- **double**: RL reward = `-max(z_A, z_B)` — worst-case over both targets

**Exploratory:**

- **linear_equal**: RL reward = `-(z_A + z_B)` — equally weighted
- **biased_a**: RL reward = `-(z_A + 0.3·z_B)` — target A priority
- **selective_a**: RL reward = `-(z_A - z_B)` — maximize selectivity for A

### Target Pairs

Defined in `configs/pairs.yaml`. Each pair has a `set` field:

| Set | Count | Purpose | Pairs |
|-----|-------|---------|-------|
| **core** | 5 | Primary hypothesis test | CDK2_ERK2, AChE_BChE, BACE1_GSK3B, AChE_BACE1, GSK3B_CDK5 |
| **auxiliary** | 3 | Robustness / sensitivity | EGFR_HER2, BACE1_CDK5, AChE_MAOB |
| **blocked** | 1 | Excluded (non-human PDB) | COX1_COX2 |

### Hit Definition

A molecule is a **hit** for target X if Boltz-2 affinity `a_X <= T_X`, where `T_A = T_B = 0.0` (= 1 uM). A **dual hit** satisfies both thresholds simultaneously. Thresholds are fixed pre-experiment in `configs/metrics.yaml`.

### Success Criteria

Cliff's delta > -0.47 (medium effect size) on at least 3 of 5 core pairs, comparing double vs best-of(single_a, single_b) dual hit-rates.

---

## Documentation Map

Read in the order shown for fastest onboarding.

| # | Document | Status | Description |
|---|----------|--------|-------------|
| 1 | [Goal](docs/goal.md) | Operational spec (read-only) | Hypothesis, notation, endpoints, success criteria -- single source of truth |
| 2 | [Setup](docs/setup.md) | Canonical | Installation, Docker build, calibration, baseline preparation |
| 3 | [Pipeline](docs/pipeline.md) | Canonical | Full experiment pipeline: Stage 0 through report generation |
| 4 | [Architecture](docs/architecture.md) | Reference | Backend architecture (generator/scorer), Boltz-2, FREED++, caching, score conventions, troubleshooting |
| 5 | [Selection Criteria](docs/selection_criteria.md) | Reference | Target pair rationale, factorial design, core/auxiliary split |
| 6 | [Pre-registration](docs/preregistration.md) | Frozen (v1.1) | Frozen hypotheses and decision rules |
| 7 | [Statistical Analysis Plan](docs/statistical_analysis_plan.md) | Frozen (v1.1) | Inference procedures, claim rules, code mapping |
| 8 | [Receptor QC](docs/receptor_qc.md) | Reference | Per-target structural quality control |
| 9 | [NIM Benchmark](docs/nim_benchmark.md) | Reference | Boltz-2 NIM backend benchmarks |
| 10 | [Doc Index](docs/README.md) | Living | Source priority and conflict resolution rules |

**Source priority** (higher wins in conflicts): goal.md > configs/*.yaml > dualdocking/ > selection_criteria.md > report/ > README.md.

---

## Testing

```bash
# Full test suite (~400 tests, parallelized)
uv run pytest tests/ -v

# Single test file
uv run pytest tests/scoring/test_objectives.py -v

# Linting (must pass with zero warnings)
uv run ruff check dualdocking/ scripts/ docker/

# Type checking (must pass with zero errors)
uv run ty check dualdocking/

# Auto-format
uv run ruff format dualdocking/ scripts/
```

---

## References

- **FREED++**: Telepov et al., TMLR 2023 -- [arXiv:2401.09840](https://arxiv.org/abs/2401.09840). RL-based molecular generation with docking reward.
- **Boltz-2**: Passaro et al., 2025 -- [bioRxiv (PMID 40667369)](https://github.com/jwohlwend/boltz). Structure-aware learned affinity predictor (Pearson 0.66 on 4-target FEP+).
- **UniDock**: Yu et al. -- [dptech-corp/Uni-Dock](https://github.com/dptech-corp/Uni-Dock). GPU-accelerated molecular docking (Vina scoring function).
- **RDKit**: [rdkit.org](https://www.rdkit.org/). Cheminformatics toolkit for molecular processing.
