# Dual-Target Docking

Single-target vs dual-target RL molecule generation using **FREED++** or **REINVENT4** (generators) and **Boltz-2** (scorer).

**Hypothesis.** RL optimized for worst-case dual-target docking (`max(d_A, d_B)`) produces molecules with higher dual hit-rate than single-target optimization, evaluated by an orthogonal Boltz-2 affinity surrogate.

---

## Data Flow

```
Stage 0: Fetch              Generation (FREED++ / REINVENT4)        Scoring (Boltz-2)
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

**75 runs total** (core): 5 core pairs x 3 modes x 5 seeds. Auxiliary pairs (7) for robustness checks.

---

## Project Structure

```
SWW2026/
|-- dualdocking/                Core Python library
|   |-- api.py                  Programmatic API (run_full_experiment, run_stages, run_post_pipeline)
|   |-- config.py               YAML config loading & validation (pairs, metrics, experiment)
|   |-- command_runner.py       In-process dispatch + subprocess fallback for CLI scripts
|   |-- script_dispatch.py      Click CLI entry point dispatcher
|   |-- scoring/                Scoring & docking objectives
|   |   |-- objectives.py       Multi-objective scalarizations (min, harmonic, Tchebycheff, Pareto, ...)
|   |   |-- backends.py         Scoring backend registry (builtin boltz2 + external engines)
|   |   |-- boltz2.py           Boltz-2 affinity scorer facade (CLI or NIM backend)
|   |   |-- boltz2_batch.py     Batch processing for Boltz-2 scoring
|   |   |-- boltz2_context.py   Execution context manager
|   |   |-- boltz2_helpers.py   Boltz-2 utility functions
|   |   |-- boltz2_nim_backend.py  NVIDIA NIM REST API client
|   |   |-- boltz2_scoring_api.py  Scoring orchestration
|   |   |-- boltz2_storage.py   Result storage & caching
|   |   |-- boltz2_subprocess.py  Subprocess management
|   |   |-- cache_io.py         Process-safe JSON cache (fcntl.flock + atomic temp-and-replace)
|   |   |-- binding_box.py      PDB binding-site extraction from co-crystallized ligands
|   |   |-- checkpoint.py       Scoring checkpoint recovery
|   |   |-- docking.py          UniDock Docker integration & PDBQT preparation
|   |   +-- gnina.py            GNINA CNN-based scoring (optional cross-validation)
|   |-- generation/             Molecule generation (multi-backend)
|   |   |-- backends.py         Generation backend registry (builtin freedpp/reinvent4 + external)
|   |   |-- freedpp/            FREED++ Docker-based generation subpackage
|   |   |   |-- core.py         Docker subprocess wrapper (GenerationConfig, generate())
|   |   |   |-- docker.py       Docker container management
|   |   |   |-- io.py           Receptor prep & file I/O helpers
|   |   |   +-- model.py        FREED++ model configuration
|   |   +-- reinvent4/          REINVENT4 native Python generation subpackage
|   |       |-- core.py         Pipeline orchestration (generate(), cache, workspace)
|   |       |-- scoring.py      Dual-target docking scoring, RewardStats tracking
|   |       +-- config.py       TOML config generation for REINVENT4 CLI
|   |-- analysis/               Post-hoc statistical analysis
|   |   |-- statistical.py      Hierarchical bootstrap, FDR correction, Cliff's delta
|   |   |-- fairness.py         Generation budget, duplication, and reward quality gate
|   |   |-- enrichment.py       AUROC, BEDROC, enrichment factor for docking validation
|   |   |-- diversity.py        Uniqueness, scaffold diversity, Tanimoto metrics
|   |   |-- learning_curves.py  Learning curve analysis for generative models
|   |   |-- mixed_effects.py    Linear mixed-effects modeling for paired targets
|   |   |-- novelty.py          Scaffold and similarity-based novelty scoring
|   |   |-- power_analysis.py   Statistical power and sample size calculations
|   |   |-- rliable_metrics.py  Rliable aggregate statistics across seeds
|   |   |-- wild_bootstrap.py   Wild bootstrap for time-series data
|   |   |-- calibration/        Baseline calibration & validation
|   |   |   |-- docking.py      UniDock integration for redocking & enrichment
|   |   |   |-- enrichment.py   Enrichment benchmark (AUROC/BEDROC)
|   |   |   |-- normalization.py  Z-score normalization & stability checks
|   |   |   |-- power.py        Power analysis for statistical tests
|   |   |   +-- redocking.py    Redocking validation (known ligand pose prediction)
|   |   +-- metrics/            Modular metric computation
|   |       |-- aggregation.py  Concatenate scored results across pairs/modes/seeds
|   |       |-- common.py       Shared metric utilities
|   |       |-- computation.py  Primary performance metrics (hit-rate, selectivity)
|   |       |-- correlation.py  Correlation analysis between surrogates
|   |       +-- sensitivity.py  Threshold sensitivity analysis
|   |-- data/                   Molecular data ingestion and cleaning
|   |   |-- chem_sanitize.py    SMILES validation, canonicalization & deduplication
|   |   |-- baseline.py         ZINC calibration baseline stability checks
|   |   |-- chembl_validation.py  ChEMBL active retrieval, dual-active overlap, enrichment
|   |   |-- filters.py          Molecular property filters (Lipinski, PAINS, etc.)
|   |   |-- http_utils.py       HTTP retry logic & error handling
|   |   |-- io_utils.py         Parquet/CSV I/O helpers
|   |   |-- mol_prep.py         SMILES→3D→PDBQT molecule preparation
|   |   |-- fetch/              External data source integration
|   |   |   |-- docker.py       Docker container management for data tools
|   |   |   |-- msa.py          Multiple sequence alignment fetching
|   |   |   |-- pdb.py          PDB structure & UniProt FASTA download
|   |   |   +-- smiles.py       SMILES file fetch (ZINC, ChEMBL, FREED fragments)
|   |   +-- prep/               Data preparation pipelines
|   |       |-- baseline.py     ZINC baseline splitting & sampling
|   |       +-- targets.py      Target structure preparation
|   |-- pipeline/               Orchestration & workflow management
|   |   |-- checks.py           Pre-flight validation (environment, dependencies)
|   |   |-- config.py           Pipeline stage configuration
|   |   |-- core.py             Logging, subprocess execution, environment management
|   |   |-- executor.py         Stage execution coordinator
|   |   |-- runner.py           End-to-end pipeline runner
|   |   +-- stages/             Atomic pipeline stages (fetch, prepare)
|   |-- reporting/              Experiment provenance and manuscript
|   |   |-- run_manifest.py     Run lifecycle tracking (JSON manifest with atomic I/O)
|   |   |-- latex_report.py     Jinja2-to-LaTeX rendering and PDF compilation
|   |   +-- plots/              Plotting & figure generation
|   |       |-- distributions.py  Hit-rate, potency, diversity distributions
|   |       +-- molecules.py    Molecule structure visualization (2D depiction)
|   |-- reward/                 Reward functions & scalarization
|   |   +-- selectivity.py      Multi-objective reward modes (single_a, double, biased, etc.)
|   |-- utils/                  Utilities
|   |   |-- hardware.py         GPU/CPU detection & resource checking
|   |   |-- hash_utils.py       Deterministic hashing for caching
|   |   |-- io_atomic.py        Atomic file I/O (prevent corruption)
|   |   |-- master_config.py    Master configuration merging
|   |   |-- naming.py           Canonical naming conventions
|   |   +-- nim.py              NVIDIA NIM API utilities
|   +-- validation/             Post-hoc validation
|       |-- audit.py            Claim traceability audit
|       |-- md_config.py        MD configuration dataclasses, constants, GPU detection
|       |-- md_docker.py        GROMACS Docker management
|       |-- md_mdp.py           MDP file writers (EM, NVT, NPT, production)
|       |-- md_setup.py         System preparation (PDB repair, ligand parameterization, topology)
|       |-- md_analysis.py      Trajectory analysis (RMSD, RMSF, Rg, H-bonds, convergence)
|       +-- md_pipeline.py      Orchestration (22 GROMACS commands, convergence assessment)
|
|-- cli/                        CLI entry points (organized by pipeline stage)
|   |-- s0_fetch/               Data download (PDB, FASTA, FREED fragments, ZINC, ChEMBL, GNINA)
|   |-- s1_prepare/             Target preparation (PDB -> PDBQT, binding box, MSA, baseline)
|   |-- s2_calibrate/           Calibration and validation (docking, normalization, redocking, enrichment)
|   |-- s3_generate/            Molecule generation (configured backend; default FREED++)
|   |-- s4_score/               Affinity scoring (configured engine; default Boltz-2) + GNINA
|   |-- s5_analyze/             Statistical analysis (aggregate, metrics, correlation, sensitivity)
|   |-- s6_report/              Visualization and manuscript (plots, molecules, LaTeX)
|   |-- s6_md/                  MD validation (GROMACS, post-hoc binding stability)
|   |-- s7_audit/               Post-hoc verification (claim traceability, duplication check)
|   |-- pipeline/               End-to-end orchestrators (full experiment, pilot, post, stages)
|   +-- smoke/                  Smoke tests (Boltz-2 health check, reward capping)
|
|-- configs/                    YAML configuration
|   |-- pairs.yaml              Target pairs (5 core + 7 auxiliary + 1 blocked)
|   |-- metrics.yaml            Metric definitions (hit-rate thresholds, effect size)
|   |-- experiment.yaml         Global experiment config (generator, scorer, seeds, paths)
|   |-- reward_modes.yaml       Reward scalarization definitions
|   |-- calibration.yaml        UniDock calibration parameters
|   |-- normalization.yaml      Z-score normalization (generated by calibration)
|   |-- filters.yaml            Molecular property filters
|   |-- master_config.yaml      Master configuration (merges pairs + experiment)
|   +-- pairs_single.yaml       Single-target baseline pairs
|-- tests/                      pytest suite (~1160 tests)
|-- docker/                     Container definitions
|   |-- freedpp/                FREED++ image (Dockerfile + dual_reward_patch modules + data/)
|   |-- gnina/                  GNINA CNN scoring image
|   |-- boltz2/                 Boltz-2 NIM launch script
|   +-- md/                     GROMACS MD simulation image
|-- scripts/                    Standalone runnable scripts (mirrors cli/ structure)
|-- docs/                       Project documentation (see Documentation Map below)
|-- report/                     LaTeX manuscript (chapters/, figures/, templates/)
+-- output/                     Runtime data (gitignored)
    |-- data/                   Target structures, calibration sets, baseline
    |-- cache/                  Score cache + MSA cache + FREED++ model cache
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

## Container Images

Build all images at once (recommended):

```bash
./scripts/build_docker_images.sh                    # build all, skip existing
./scripts/build_docker_images.sh --force            # force-rebuild all
./scripts/build_docker_images.sh --only freedpp     # build only freedpp
./scripts/build_docker_images.sh --dry-run          # preview commands
```

Or build individually:

```bash
# Required for generation (FREED++)
docker build -f docker/freedpp/Dockerfile -t dualdocking/freedpp:latest .

# Optional: custom GNINA image (for docker-based GNINA scoring script)
docker build -f docker/gnina/Dockerfile -t dualdocking/gnina:latest .

# Optional: MD validation runtime (required only for --docker MD runs)
docker build -f docker/md/Dockerfile -t dualdocking/md:latest .
```

Notes:
- `cli/s0_fetch/fetch_all.py` builds `dualdocking/freedpp:latest` automatically.
- `cli/s0_fetch/fetch_all.py` downloads GNINA binary to `output/bin/gnina`; it does not build `dualdocking/gnina:latest`.
- MD image is only needed for `cli/s6_md/run_md.py --docker`.

Detailed setup and troubleshooting: [docs/setup.md](docs/setup.md).

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd SWW2026
uv sync --group dev

# 2. Build FREED++ Docker image
uv run python cli/s0_fetch/fetch_freed_fragments.py
docker build -f docker/freedpp/Dockerfile -t dualdocking/freedpp:latest .

# 3. Fetch all external data (internet required only for this step)
uv run python cli/s0_fetch/fetch_all.py

# 4. Calibrate normalization (per-target z-score parameters)
uv run python cli/s2_calibrate/dock_calibration.py
uv run python cli/s2_calibrate/compute_normalization.py

# 5. Run full experiment (offline from here)
EXPERIMENT_RUN_ID=18022026 uv run python cli/pipeline/run_full_experiment.py \
  --resume --parallel 1 --with-post --latex-mode draft

# Run specific pairs only (pair filter applies to all stages including analysis)
EXPERIMENT_RUN_ID=18022026 uv run python cli/pipeline/run_full_experiment.py \
  --pair EGFR_HER2 --pair MEK1_MEK2 --seeds 42 \
  --resume --parallel 2 --with-post --latex-mode draft
```

GPU routing note:
- `--parallel` should match the number of GPUs you expose via `CUDA_VISIBLE_DEVICES`.
- On shared servers, prefer explicit masks (example: `CUDA_VISIBLE_DEVICES=1,3 --parallel 2`).
- Detailed GPU affinity rules and FREED++ runtime env vars: [docs/setup.md](docs/setup.md) and [docs/pipeline.md](docs/pipeline.md).

Results land in `output/results/` and `report/`. See [Pipeline](docs/pipeline.md) for stage-by-stage details.

### MD Validation (Optional)

Post-hoc molecular dynamics validation of top candidates. Requires
GROMACS, AmberTools, acpype, and OpenBabel — see [Setup](docs/setup.md)
for installation.

```bash
uv run python cli/s6_md/run_md.py --pair CDK2_ERK2 --dry-run   # preview
uv run python cli/s6_md/run_md.py --pair CDK2_ERK2              # run (50 ns, auto-GPU)
```

### Smoke Test

Verify the setup works end-to-end with a minimal run:

```bash
# Quick smoke test: 10 epochs, 1 seed, 1 pair (takes ~15-20 min with GPU)
FREEDPP_EPOCHS=10 EXPERIMENT_RUN_ID=smoke uv run python cli/pipeline/run_full_experiment.py \
  --pair CDK2_ERK2 --seeds 42 --resume --parallel 1 --with-post --latex-mode draft

# Generate 10 molecules for one pair, 2 epochs (takes ~2 minutes with GPU)
uv run python cli/s3_generate/generate.py --pair CDK2_ERK2 --mode single_a --epochs 2 --n-molecules 10

# Boltz-2 health check
uv run dd-smoke
```

Environment variables for generation backends:

| Variable | Default | Purpose |
|----------|---------|---------|
| `FREEDPP_EPOCHS` | from config (100) | Override FREED++ epoch count (e.g. `10` for smoke tests) |
| `FREEDPP_NUM_SUB_PROC` | auto-detect | Override FREED++ concurrent docking subprocesses |
| `REINVENT4_N_STEPS` | from config (500) | Override REINVENT4 RL step count |

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
  backend: freedpp  # built-in: freedpp | reinvent4; or package.module[:callable]
  freedpp:
    # Reward-mode config passthrough into container (model-dependent)
    pass_reward_modes_to_container: true
    reward_modes_path_in_container: /opt/reward_modes.yaml
  reinvent4:
    algorithm: dap
    batch_size: 128
    n_steps: 500

scorer:
  # Legacy key also supported: name: "Boltz-2"
  engine: boltz2    # built-in alias; or package.module[:factory_or_object]
  backend: nim      # boltz2 transport: cli | nim
```

Default behavior remains unchanged:
- Generator: built-in `freedpp` (Docker); alternative `reinvent4` (native Python)
- Scorer: built-in `boltz2` (`cli` or `nim` based on config)
- FREED++ reward config path passthrough: enabled (`/opt/reward_modes.yaml`)

Both generators write `reward_metadata.json` into their workspace directory. The quality gate (`fairness.py:validate_reward_quality`) resolves the newest metadata when both workspaces exist.

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
- **selective_weighted**: RL reward = `-(z_A - lambda_off * z_B)` — configurable off-target penalty

All reward formulas are centralized in `dualdocking/reward/selectivity.py`,
with built-in declarations in `configs/reward_modes.yaml`. The registry can be
extended via `register_reward_mode(...)` or environment plugin loading
(`DUALDOCKING_REWARD_PLUGINS`).

Default mode parameters in `configs/reward_modes.yaml`:

- `biased_a.weight_b = 0.3`
- `selective_weighted.lambda_off = 1.0`
- all other built-in modes have empty `default_params` (no tunable parameters)

For schema, loading priority, and plugin mechanics, see [Reward Modes](docs/reward_modes.md).

### Target Pairs

Defined in `configs/pairs.yaml`. Each pair has a `set` field:

| Set | Count | Purpose | Pairs |
|-----|-------|---------|-------|
| **core** | 5 | Primary hypothesis test | CDK2_ERK2, AChE_BChE, BACE1_GSK3B, AChE_BACE1, GSK3B_CDK5 |
| **auxiliary** | 7 | Robustness / sensitivity | EGFR_HER2, BACE1_CDK5, AChE_MAOB, MEK1_MEK2, FGFR1_FGFR2, EZH2_EZH1, BRD4_BRD2 |
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
| 10 | [Reward Modes](docs/reward_modes.md) | Reference | Declarative reward registry, YAML schema, plugin extension |
| 11 | [I/O Contracts](docs/io_contracts.md) | Reference | File paths, Parquet schemas, data flow between stages |
| 12 | [Novelty Report](docs/novelty_report.md) | Reference | Gap analysis vs prior work, novelty assessment |
| 13 | [Pipeline Granularity](docs/tasks_pipeline_granularity.md) | Living | Pipeline stage refactoring plan (9→17 stages) |
| 14 | [Doc Index](docs/README.md) | Living | Source priority and conflict resolution rules |

**Source priority** (higher wins in conflicts): goal.md > configs/*.yaml > dualdocking/ > selection_criteria.md > report/ > README.md.

---

## Testing

```bash
# Full test suite (~1160 tests, parallelized)
uv run pytest tests/ -v

# Single test file
uv run pytest tests/scoring/test_objectives.py -v

# Linting (must pass with zero warnings)
uv run ruff check dualdocking/ cli/ docker/

# Type checking (must pass with zero errors)
uv run ty check dualdocking/

# Auto-format
uv run ruff format dualdocking/ cli/
```

---

## References

- **FREED++**: Telepov et al., TMLR 2023 -- [arXiv:2401.09840](https://arxiv.org/abs/2401.09840). RL-based molecular generation with docking reward.
- **REINVENT4**: Loeffler et al., J Cheminformatics 2024, [PMID 38383444](https://pubmed.ncbi.nlm.nih.gov/38383444/). SMILES-based RL with DAP loss and curriculum learning.
- **Boltz-2**: Passaro et al., 2025 -- [bioRxiv (PMID 40667369)](https://github.com/jwohlwend/boltz). Structure-aware learned affinity predictor (Pearson 0.66 on 4-target FEP+).
- **UniDock**: Yu et al. -- [dptech-corp/Uni-Dock](https://github.com/dptech-corp/Uni-Dock). GPU-accelerated molecular docking (Vina scoring function).
- **RDKit**: [rdkit.org](https://www.rdkit.org/). Cheminformatics toolkit for molecular processing.
