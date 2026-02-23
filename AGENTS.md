# AGENTS.md

## Scientific Rigor — READ FIRST

This is a **scientific research repository** (computational chemistry / drug discovery). Every action must meet academic standards.

### Critical & Destructive Analysis

- **Default posture: skepticism.** Do not accept claims, metrics, or design choices at face value. Actively look for flaws, confounders, and alternative explanations.
- When reviewing or writing code, docs, or configs — ask: *"What could make this result wrong?"* Check for data leakage, selection bias, overfitting to the evaluation surrogate, and violation of statistical assumptions.
- If a limitation is found, **document it explicitly** (in `docs/`, `report/chapters/06_limitations.tex`, or inline comments) rather than silently working around it.
- Challenge thresholds, sample sizes, and effect-size interpretations. m=5 bootstrap seeds is a known weakness — never present it as statistically robust without the caveat.

### Citation Requirement

- **Every empirical claim must be backed by a peer-reviewed source** (journal article, peer-reviewed conference, or well-established preprint with >50 citations).
- When adding or modifying claims in `report/`, `docs/goal.md`, or `docs/selection_criteria.md`, verify the citation exists in `report/references.bib` and the stated facts match the source.
- Use MCP tools (PubMed, bioRxiv, ChEMBL, HuggingFace papers) to **verify facts at the source** — do not rely on memory or parametric knowledge for numbers, author lists, or conclusions.
- If a claim cannot be substantiated, flag it with `% TODO: citation needed` in LaTeX or `<!-- TODO: citation needed -->` in Markdown.
- Distinguish clearly between: (a) established facts with citations, (b) methodological choices with justification, and (c) assumptions that remain unvalidated.

### Document Synchronization

All documents describe the **same experiment** and must stay consistent. When changing any parameter, threshold, pair list, or design decision:

| Change in...            | Must also update...                                                              |
|-------------------------|----------------------------------------------------------------------------------|
| `configs/pairs.yaml`    | `docs/selection_criteria.md`, `report/chapters/04_experimental_protocol.tex`     |
| `configs/metrics.yaml`  | `docs/goal.md` (thresholds), `report/chapters/04_experimental_protocol.tex`      |
| `configs/experiment.yaml` | `docs/goal.md`, `report/chapters/03_methods_notes.tex`                         |
| `configs/reward_modes.yaml` | `docs/reward_modes.md`, `docs/architecture.md`, `README.md`                 |
| `dualdocking/**/*.py`   | Corresponding `tests/test_*.py`, `docs/goal.md` if behavior changes            |
| `docs/goal.md`          | `report/chapters/` (methods, protocol, limitations)                              |
| `report/references.bib` | All `\cite{}` in `report/chapters/*.tex`                                         |
| Install/Docker/pipeline | `docs/setup.md`, `docs/pipeline.md`                                             |
| SLURM scripts/resources | `docs/pipeline.md` (SLURM section), `docs/setup.md` (SLURM section)             |
| Scoring/caching/FREED++ | `docs/architecture.md`                                                           |

**Before finishing any task**, verify cross-document consistency for the entities you touched. If you change a number in one place and the same number appears elsewhere, update all occurrences or flag the discrepancy.

### `docs/goal.md` — Protected Master Document

`docs/goal.md` is the **single source of truth** for the entire project. It defines:

- Hypothesis and research question
- Notation (`d_A`/`d_B`, `a_A`/`a_B`)
- Endpoint hierarchy (primary, secondary, sensitivity)
- Success criteria (≥3/5 core pairs, Cliff's delta > −0.47)
- Compute parity constraints
- Statistical analysis plan (two-level bootstrap, FDR)

**This file is READ-ONLY without explicit user permission.** Do not edit, rewrite, or reformat `docs/goal.md` — even to fix typos or formatting. If you find an inconsistency between `docs/goal.md` and another document, the other document is wrong. Ask the user before proposing changes to `goal.md`.

All other documents (`report/`, `docs/selection_criteria.md`, `README.md`, `configs/`) must conform to `docs/goal.md`, not the other way around.

### Source Priority (conflicts → higher wins)

1. **`docs/goal.md`** — supreme authority: hypothesis, endpoints, success criteria, notation. **Read-only without user approval.**
2. `configs/*.yaml` — ground truth for runtime parameters (must agree with goal.md)
3. `dualdocking/` (all subpackages) — ground truth for computation logic (must implement what goal.md specifies)
4. `docs/selection_criteria.md` — ground truth for pair selection rationale
5. `report/chapters/*.tex` — narrative must reflect the above, never contradict
6. `README.md` — operational guide, may simplify but must not misrepresent

---

## Build & Test Commands

```bash
# Install all dependencies (boltz is a pinned git dep, included automatically)
uv sync --group dev

# Run full test suite (~1160 tests)
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/scoring/test_objectives.py -v

# Linting (must pass with zero warnings)
uv run ruff check dualdocking/ cli/ docker/ scripts/

# Type checking (must pass with zero errors)
uv run ty check dualdocking/

# Auto-format
uv run ruff format dualdocking/ cli/

# Docker build (FREED++ generator)
docker build -f docker/freedpp/Dockerfile -t dualdocking/freedpp:latest .
```

## Project Architecture

```
dualdocking/                  # Core library (at project root, no src/ wrapper)
  api.py                      #   Programmatic API (run_full_experiment, run_stages, run_post_pipeline)
  config.py                   #   YAML config loading & validation
  command_runner.py           #   In-process dispatch + subprocess fallback for CLI scripts
  script_dispatch.py          #   Click CLI entry point dispatcher (avoids extra interpreters)
  scoring/                    #   Scoring & docking subpackage
    objectives.py             #     Multi-objective functions (min, harmonic, Pareto, etc.)
    backends.py               #     Scorer registry (builtin boltz2 + external engines)
    boltz2.py                 #     Boltz-2 affinity scorer facade (CLI or NIM backend)
    boltz2_batch.py           #     Batch processing for Boltz-2 scoring
    boltz2_context.py         #     Boltz-2 execution context manager
    boltz2_helpers.py         #     Boltz-2 utility functions
    boltz2_nim_backend.py     #     NVIDIA NIM Boltz-2 REST API client
    boltz2_scoring_api.py     #     Boltz-2 scoring orchestration
    boltz2_storage.py         #     Boltz-2 result storage & caching
    boltz2_subprocess.py      #     Boltz-2 subprocess management
    cache_io.py               #     Atomic disk I/O (fcntl + temp-and-replace)
    binding_box.py            #     PDB binding site extraction
    checkpoint.py             #     Scoring checkpoint recovery
    docking.py                #     UniDock Docker integration & PDBQT preparation
    gnina.py                  #     GNINA CNN-based scoring (optional cross-validation)
  generation/                 #   Molecule generation subpackage
    backends.py               #     Generation backend registry (builtin freedpp/reinvent4 + external)
    freedpp/                  #     FREED++ Docker-based generation subpackage
      core.py                 #       Docker subprocess wrapper (GenerationConfig, generate())
      docker.py               #       Docker container management
      io.py                   #       Receptor prep & file I/O helpers
      model.py                #       FREED++ model configuration
    reinvent4/                #     REINVENT4 native Python generation subpackage
      core.py                 #       Pipeline orchestration (generate(), cache, workspace)
      scoring.py              #       Dual-target docking scoring + RewardStats tracking
      config.py               #       TOML config generation for REINVENT4 CLI
  analysis/                   #   Statistical analysis subpackage
    statistical.py            #     Bootstrap, FDR, Cliff's delta
    fairness.py               #     Fairness & quality gate (multi-workspace reward metadata)
    enrichment.py             #     Enrichment analysis
    diversity.py              #     Chemical diversity metrics
    learning_curves.py        #     Learning curve analysis for generative models
    mixed_effects.py          #     Linear mixed-effects modeling for paired targets
    novelty.py                #     Scaffold and similarity-based novelty scoring
    power_analysis.py         #     Statistical power and sample size calculations
    rliable_metrics.py        #     Rliable aggregate statistics across seeds
    wild_bootstrap.py         #     Wild bootstrap for time-series data
    calibration/              #     Baseline calibration & validation
      docking.py              #       UniDock integration for redocking & enrichment
      enrichment.py           #       Enrichment benchmark (AUROC/BEDROC)
      normalization.py        #       Z-score normalization & stability checks
      power.py                #       Power analysis for statistical tests
      redocking.py            #       Redocking validation (known ligand pose prediction)
    metrics/                  #     Modular metric computation
      aggregation.py          #       Concatenate scored results across pairs/modes/seeds
      common.py               #       Shared metric utilities
      computation.py          #       Primary performance metrics (hit-rate, selectivity)
      correlation.py          #       Correlation analysis between surrogates
      sensitivity.py          #       Threshold sensitivity analysis
  data/                       #   Data handling subpackage
    chem_sanitize.py          #     SMILES validation & dedup
    baseline.py               #     Baseline comparison
    chembl_validation.py      #     ChEMBL target validation
    filters.py                #     Molecular property filters (Lipinski, PAINS, etc.)
    http_utils.py             #     HTTP retry logic & error handling
    io_utils.py               #     Parquet/CSV I/O helpers
    mol_prep.py               #     SMILES→3D→PDBQT molecule preparation
    fetch/                    #     External data source integration
      docker.py               #       Docker container management for data tools
      msa.py                  #       Multiple sequence alignment fetching
      pdb.py                  #       PDB structure & UniProt FASTA download
      smiles.py               #       SMILES file fetch (ZINC, ChEMBL, FREED fragments)
    prep/                     #     Data preparation pipelines
      baseline.py             #       ZINC baseline splitting & sampling
      targets.py              #       Target structure preparation
  pipeline/                   #   Orchestration & workflow management
    checks.py                 #     Pre-flight validation (environment, dependencies)
    config.py                 #     Pipeline stage configuration
    core.py                   #     Logging, subprocess execution, environment management
    executor.py               #     Stage execution coordinator
    runner.py                 #     End-to-end pipeline runner
    stages/                   #     Atomic pipeline stages
      fetch.py                #       Data fetching orchestration
      prepare.py              #       Data preparation orchestration
  reporting/                  #   Reporting subpackage
    run_manifest.py           #     Lifecycle tracking for experiment runs
    latex_report.py           #     LaTeX report generation
    plots/                    #     Plotting & figure generation
      distributions.py        #       Hit-rate, potency, diversity distributions
      molecules.py            #       Molecule structure visualization (2D depiction)
  reward/                     #   Reward functions & scalarization
    selectivity.py            #     Multi-objective reward modes (single_a, double, biased, etc.)
  utils/                      #   Utilities
    hardware.py               #     GPU/CPU detection & resource checking
    hash_utils.py             #     Deterministic hashing for caching
    io_atomic.py              #     Atomic file I/O (prevent corruption)
    master_config.py          #     Master configuration merging
    naming.py                 #     Canonical naming conventions
    nim.py                    #     NVIDIA NIM API utilities
  validation/                 #   Post-hoc validation subpackage
    audit.py                  #     Claim traceability audit
    md_config.py              #     MD configuration dataclasses, constants, GPU detection
    md_docker.py              #     GROMACS Docker management
    md_mdp.py                 #     MDP file writers (EM, NVT, NPT, production)
    md_setup.py               #     System preparation (PDB repair, ligand parameterization, topology)
    md_analysis.py            #     Trajectory analysis (RMSD, RMSF, Rg, H-bonds, convergence)
    md_pipeline.py            #     Orchestration (22 GROMACS commands, convergence KS test)
cli/                          # CLI entry points (organized by pipeline stage)
  s0_fetch/                   #   Data download (PDB, FASTA, FREED fragments, ZINC, ChEMBL, GNINA)
  s1_prepare/                 #   Target preparation (PDB -> PDBQT, binding box, MSA, baseline)
  s2_calibrate/               #   Calibration and validation (docking, normalization, redocking, enrichment)
  s3_generate/                #   Molecule generation (configured backend; default FREED++)
  s4_score/                   #   Affinity scoring (configured engine; default Boltz-2)
  s5_analyze/                 #   Statistical analysis (aggregate, metrics, correlation, sensitivity)
  s6_report/                  #   Visualization and manuscript (plots, molecules, LaTeX)
  s7_audit/                   #   Post-hoc verification (claim traceability, duplication check)
  s8_gnina/                   #   GNINA cross-validation scoring
  s9_md/                      #   MD validation (GROMACS, post-hoc binding stability)
  pipeline/                   #   End-to-end orchestrators (full experiment, pilot, post, stages)
  smoke/                      #   Smoke tests (Boltz-2 health check, reward capping)
tests/                        # pytest suite (~1160 tests), monkeypatch-heavy
configs/                      # YAML configuration
  pairs.yaml                  #   Target pairs (5 core + 7 auxiliary + 1 blocked)
  metrics.yaml                #   Metric definitions (hit-rate thresholds, effect size)
  experiment.yaml             #   Global experiment config (generator, scorer, seeds, paths)
  reward_modes.yaml           #   Reward scalarization definitions
  calibration.yaml            #   UniDock calibration parameters
  normalization.yaml          #   Z-score normalization (generated by calibration)
  filters.yaml                #   Molecular property filters
  master_config.yaml          #   Master configuration (merges pairs + experiment)
  pairs_single.yaml           #   Single-target baseline pairs
docker/                       # Container definitions
  freedpp/                    #   FREED++ image (Dockerfile + dual_reward_patch modules + data/)
  gnina/                      #   GNINA CNN scoring image
  boltz2/                     #   Boltz-2 NIM launch script
  md/                         #   GROMACS MD simulation image
slurm/                        # SLURM HPC integration (zero Python code changes)
  common.sh                   #   Shared environment setup (modules, paths, run ID)
  task_matrix.py              #   Array index -> (pair, mode, seed) mapper
  monolithic/                 #   Strategy A: single node, N GPUs (submit.sh + 6 sbatch)
  array/                      #   Strategy B: multi-node array jobs (submit.sh + 6 sbatch)
  nim/                        #   NIM Boltz-2 container lifecycle (start/stop/health)
docs/                         # Project documentation (14 files, see Documentation Map in README)
report/                       # LaTeX manuscript (chapters/, figures/, templates/)
scripts/                      # Standalone runnable scripts (mirrors cli/ structure)
output/                       # Runtime data (gitignored)
  data/                       #   Target structures, calibration, baseline
  cache/                      #   Boltz-2 score cache, MSA cache, FREED++ model cache
  results/                    #   Experiment outputs
```

### Data Flow

Stage 0 (fetch) → Generation (FREED++ or REINVENT4, 3 modes × 5 core pairs × 5 seeds = 75 primary runs) → Scoring (Boltz-2) → Aggregation → Metrics → Report

## Code Style & Conventions

- **Python >=3.11**, package manager `uv`, build system `hatchling`
- **Ruff**: `select = ["ALL"]` with curated ignores, line-length 130, Google docstrings, double quotes
- **Type checker**: `ty` in strict mode (`dualdocking/` only; tests and cli excluded)
- **lower_is_better = True everywhere** — docking scores are negative (more negative = better binding). Utility = −score.
- All objective functions convert scores to utilities before aggregation
- Tchebycheff is the exception: works directly in score-space (abs is symmetric)

### Naming

- `d_A`, `d_B` — UniDock docking scores (kcal/mol, RL reward signal)
- `a_A`, `a_B` — Boltz-2 affinity predictions (log10 IC50-like μM, evaluation surrogate)
- Target pairs: `CDK2_ERK2`, `AChE_BChE`, etc. — see `configs/pairs.yaml`
- Sets: `core` (5 pairs, primary analysis), `auxiliary` (7), `blocked` (1, COX1_COX2)

### Cache System

Boltz-2 scores use a disk cache segmented by target: `output/cache/boltz_scores/{target}.{ctx[:8]}.json`

- Cache key v2: `SHA256(target || smi || ctx_hash || seq_hash || msa_hash)`
- Context hash includes: Boltz version, CLI args, accelerator, Docker digest
- I/O: `fcntl.flock` for locking, atomic write via temp file + `os.replace`
- Module-level `_CONTEXT_HASH` is lazily cached; tests must reset it via `monkeypatch.setattr(mod, "_CONTEXT_HASH", None)`

### NIM Boltz-2 Backend

`scoring/boltz2.py` supports two backends controlled by the `BOLTZ_BACKEND` env var:

- **`cli`** (default) — runs Boltz-2 as a subprocess via the `boltz` CLI
- **`nim`** — calls a persistent NVIDIA NIM container via REST API (`BOLTZ_NIM_URL` env var, default `http://localhost:8000`)

NIM and CLI backends produce separate cache contexts (different context hashes) so their cached scores never collide. The NIM path skips MSA (the server handles it internally).

## Testing Patterns

- Tests use `monkeypatch` extensively to override module-level paths (`CACHE_DIR`, `DATA_DIR`)
- **FakeDockingVina**: real class with `_scores: list[float]` for integration tests through `_dual_call`
- **dual_reward_patch**: auto-applies on import — tests must mock `ffreed` BEFORE import and clean `sys.modules` AFTER
- **REINVENT4 `_reward_stats`**: module-level singleton; tests must call `_reward_stats.reset()` (autouse fixture) to prevent cross-contamination
- `load_pairs()` validates schema (requires `target_a.id`, `target_b.id`) — test fixtures must provide full structure
- Harmonic/geometric means use shift for positive-domain requirement

## Gotchas

- `_build_freedpp_args` signature: `(config, pdbqt_a, center_a, size_a, commands)` — no workspace param
- `INVALID_Z_PENALTY` is validated `> 0 && finite` at module load; negative values would invert reward
- `REWARD_METADATA_PATH` needs `mkdir(parents=True)` — custom paths may have non-existent parents
- Built-in reward mode defaults live in `configs/reward_modes.yaml` (fallback in code if YAML missing)
- FREED++ reward-config passthrough to Docker is model-dependent and configurable via
  `generator.freedpp.pass_reward_modes_to_container` and
  `generator.freedpp.reward_modes_path_in_container` in `configs/experiment.yaml`
- `pareto_mask` uses O(n log n) lexsort + sweep; handles ties with `−sb` tiebreak
- Boltz-2 output is `log10(IC50 in μM)`, NOT pIC50
- **Quality gate workspace resolution**: `validate_reward_quality()` in `fairness.py` picks the newest `reward_metadata.json` when both `freedpp_workspace/` and `reinvent4_workspace/` exist (mtime-based); output dict includes a `"workspace"` key
- **REINVENT4 `RewardStats`**: `scoring.py` accumulates per-molecule outcomes via module-level `_reward_stats` singleton; `core.py:generate()` calls `reset()` before and `write_metadata()` after training
- **REINVENT4 plugin check**: `_install_scoring_plugin()` raises `ImportError` (fail-fast) if the plugin is not importable

## Security & Boundaries

- **Never commit** `output/`, `.env`, or credential files (all gitignored)
- Docker container runs as non-root; GPU passthrough via `--gpus all`
- `scoring/cache_io.py` uses `fcntl.flock` — safe for concurrent single-machine access, NOT for NFS
- External API calls (ChEMBL, PDB) are only in `cli/` and `dualdocking/data/chembl_validation.py`, never in core scoring/generation

## Experiment Design (Context for Changes)

- **Hypothesis**: RL with worst-case dual-target reward (`max(d_A, d_B)`) produces higher dual hit-rates than single-target baselines
- **Success criteria**: Cliff's delta > −0.47 on ≥3/5 core pairs
- **Bootstrap**: m=5 seeds (acknowledged limitation; primary evidence is effect size + cross-pair consistency)
- **Thresholds**: `T_A = T_B = 0.0` (1 μM) fixed pre-experiment in `configs/metrics.yaml`
- **Scalarizations**: only `max(d_A, d_B)` in primary analysis; others are offline sensitivity
