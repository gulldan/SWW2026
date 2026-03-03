# Architecture

Technical details of scoring, generation, and data flow.

## Score Conventions

**Lower is better** everywhere:

| Score | Source | Units | Role |
|-------|--------|-------|------|
| `d_A`, `d_B` | UniDock GPU | kcal/mol | **RL reward** — drives molecule generation |
| `a_A`, `a_B` | Boltz-2 | log10(IC50-like, uM) | **Evaluation surrogate** — orthogonal post-hoc assessment |

**Important:** The two scorers are never mixed. FREED++ optimizes only docking scores
`d_A, d_B` (UniDock); Boltz-2 evaluates generated molecules independently after generation
is complete. Boltz-2 is **not** used in the RL reward loop.

- IC50 = 1 nM = 0.001 uM -> log10 = -3 (strong binder)
- IC50 = 1 uM = 1.0 uM -> log10 = 0 (threshold)
- `combined_min` = worst-case = `max(score_A, score_B)` for lower-is-better

Boltz-2 output is **log10(IC50 in uM)**, NOT pIC50.
Convert: `pIC50 = 6 - log10(IC50_uM)`.

Full notation: see `docs/goal.md` (Notation section).

## Scoring Backends

`cli/s4_score/score.py` resolves scorer implementation via
`dualdocking.scoring.backends.resolve_scoring_backend`.

- Built-in engine: `boltz2`
- External engines: `scorer.engine: "package.module"` or `scorer.engine: "package.module:factory_or_object"`

For built-in `boltz2`, `scorer.backend` selects transport (`cli` or `nim`).

### Built-in Boltz-2 Engine

Boltz-2 has no stable Python API. Built-in integration uses CLI (`boltz predict`)
or NIM REST backend.

Example config:

```yaml
scorer:
  engine: boltz2
  backend: nim  # cli | nim
  nim:
    urls:
      - http://localhost:8000
      - http://localhost:8001
```

### Input YAML

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: "MENFQKVEKIGE..."
      msa: output/cache/boltz_msa/CDK2.csv
  - ligand:
      id: B
      smiles: 'CC(=O)Oc1ccccc1C(=O)O'
properties:
  - affinity:
      binder: B
```

### CLI

```bash
# Offline scoring (MSA pre-computed):
uv run boltz predict input.yaml \
  --out_dir output/cache/boltz_runs/{target_id}/{hash} \
  --model boltz2 --accelerator gpu \
  --recycling_steps 3 --sampling_steps 50 --diffusion_samples 1 \
  --sampling_steps_affinity 50 --diffusion_samples_affinity 1 --override

# Online MSA warmup (for prepare_targets.py --prepare-all):
uv run boltz predict warmup.yaml \
  --out_dir output/cache/boltz_msa/{target_id} \
  --model boltz2 --accelerator gpu --use_msa_server
```

### Error Handling

| Problem | Protection |
|---------|-----------|
| Timeout | 600 sec per molecule, subprocess timeout |
| GPU OOM | "CUDA out of memory" detection -> NaN |
| Large ligands | <= 128 atoms check before call |
| Invalid SMILES | RDKit validation -> NaN |
| No network | MSA pre-computed, network not needed |
| MSA server down | Retry with backoff (3 attempts, prepare only) |
| Interruption | Incremental cache after each molecule |

### Cache

Disk cache segmented by target: `output/cache/boltz_scores/{target}.{ctx[:8]}.json`.

- Key: `SHA256(target || smi || ctx_hash || seq_hash || msa_hash)`
- Context includes: Boltz version, CLI args, accelerator, Docker digest
- I/O: `fcntl.flock` + atomic write (temp file + `os.replace`)
- Safe for concurrent single-machine access, NOT for NFS

## Generation Backends

`cli/s3_generate/generate.py` resolves generator implementation via
`dualdocking.generation.backends.resolve_generation_backend`.

- Built-in backend: `freedpp`
- External backends: `generator.backend: "package.module"` or `generator.backend: "package.module:callable"`

### Built-in FREED++ Backend

FREED++ (Telepov et al., TMLR 2023) is an RL molecule generator (SAC + fragment assembly).
Runs in Docker due to dependency conflicts (Python 3.10, PyTorch 1.12, DGL 0.9).

### Data Flow

```
cli/s3_generate/generate.py
  -> generation.backends.resolve_generation_backend(...)
  -> builtin generation.freedpp.generate(config)  # default
      1. _ensure_docker_image()       # build if absent
      2. _prepare_receptor()          # PDB -> PDBQT + binding box
      3. _run_freedpp()               # docker run --gpus all ... python main.py
      4. _parse_output_smiles()       # CSV from mounted volume
      -> list[str] SMILES
```

### Dual Reward Patch (inside Docker)

The FREED++ container applies a monkey-patch (`dual_reward_patch`) on import
that replaces `DockingVina.docking` and `DockingVina.__call__` to support
dual-target reward modes. The patch is split into four helper modules plus a
thin facade:

| Module | Responsibility |
|--------|----------------|
| `dual_reward_patch_deps.py` | Compatibility imports: tries `dualdocking.*` first, falls back to flat copies under `/opt`. |
| `dual_reward_patch_state.py` | Runtime state (counters, normalization, context dataclasses), z-score capping, PDBQT parsing, metadata writer. |
| `dual_reward_patch_gpu.py` | GPU-batch docking via `unidock --gpu_batch`: ligand prep pool, chunked execution, result cache. |
| `dual_reward_patch_hooks.py` | DockingVina monkey-patches: `_build_rdkit_docking`, `_build_dual_call`, CacheAndPool GPU hook. |
| `dual_reward_patch.py` | Facade: auto-applies the patch on import, delegates attribute lookups to submodules via `__getattr__`. |

**Entry point:** `run_with_patch.py` conditionally imports `dual_reward_patch`
when `REWARD_MODE` is set. The import triggers `_apply_dual_patch()`, which
loads normalization stats, resolves the reward mode, and patches DockingVina.

**Testing:** Tests (and the smoke script) mock `ffreed`, clear all five
module names from `sys.modules`, and reimport to get fresh state per test.

### Modes

Reward scalarizations are centralized in `dualdocking/reward/selectivity.py`.
FREED++ and other generators should resolve modes through this shared registry
instead of hard-coding formulas per backend.

Built-in modes are declared in `configs/reward_modes.yaml` (declarative layer).
At runtime the registry resolves the config path in this order:

1. `DUALDOCKING_REWARD_MODES_PATH`
2. `<module_dir>/reward_modes.yaml` (used in Docker fallback at `/opt/reward_modes.yaml`)
3. `<repo_root>/configs/reward_modes.yaml`
4. `<cwd>/configs/reward_modes.yaml`

For FREED++ Docker runs, exporting `DUALDOCKING_REWARD_MODES_PATH` is
configurable via `generator.freedpp.pass_reward_modes_to_container` and
`generator.freedpp.reward_modes_path_in_container` in `configs/experiment.yaml`.

**Pre-registered (primary hypothesis):**

| Mode | Evaluator output | Description |
|------|-----------------|-------------|
| `single_a` | `z_A` | Optimize for target A only |
| `single_b` | `z_B` | Optimize for target B only |
| `double` | `max(z_A, z_B)` | Worst-case over both targets |

**Exploratory (not pre-registered):**

| Mode | Evaluator output | Description |
|------|-----------------|-------------|
| `linear_equal` | `z_A + z_B` | Equally weighted linear combination |
| `biased_a` | `z_A + 0.3·z_B` | Target A priority |
| `selective_a` | `z_A - z_B` | Maximize selectivity for A over B |
| `selective_weighted` | `z_A - λ_off·z_B` | Configurable selectivity penalty |

Where `z_X = (d_X - mu_X) / sigma_X` (per-target z-score from calibration).
Z-scores are clamped to ±10 (`MAX_Z_ABS`) to prevent corrupt docking scores from
dominating the RL gradient.

**Sign convention.** Evaluators in `dualdocking/reward/selectivity.py` follow
**lower-is-better** (matching the docking score convention). Each generation
backend converts to its own RL reward convention:

- **REINVENT4** negates explicitly: `rewards[i] = -mode.evaluate(...)` (see
  `dualdocking/generation/reinvent4/scoring.py:391-393`). REINVENT4 DAP
  expects higher = better.
- **FREED++** returns the evaluator output directly from the patched
  `DockingVina.__call__` (see `docker/freedpp/dual_reward_patch_hooks.py:265`).
  This preserves the original DockingVina convention (docking scores are
  negative kcal/mol, lower = better); FREED++ SAC is designed to work with
  lower-is-better rewards natively.

Custom modes can be added without touching FREED++ internals:

- Python API: `register_reward_mode(...)`
- Plugin autoload: `DUALDOCKING_REWARD_PLUGINS="package.module[:register_reward_modes]"`

### Model Cache

Trained models saved in `output/cache/freedpp_models/{hash}/`.
Hash includes: reward mode, targets, epochs, steps_per_epoch, seed, etc.
Rerun with same params uses `--commands "sample"` (minutes vs hours).

## MD Validation Backend

`dualdocking/validation/` implements a complete GROMACS MD pipeline for
post-hoc binding stability validation of top candidate molecules. The code
is split into focused modules:

- `md_config.py` — dataclasses (`MDConfig`, `MDResult`), constants, `_run()` helper, GPU detection
- `md_mdp.py` — MDP file writers for all simulation phases
- `md_setup.py` — system preparation (PDB repair, ligand parameterization, topology, solvation)
- `md_analysis.py` — trajectory analysis (RMSD, RMSF, Rg, H-bonds, convergence, MM-GBSA)
- `md_pipeline.py` — orchestration (`run_md_validation`, `run_single_molecule_md`, `select_top_molecules`)

### Pipeline Architecture

22 GROMACS commands in 3 phases:

1. **Setup (14 commands)**: pdb2gmx → editconf → solvate → grompp/genion →
   EM → make_ndx → NVT equilibration → NPT equilibration → production MD
2. **Trajectory preprocessing (3 commands)**: PBC fix → center → least-squares fit
3. **Analysis (5 commands)**: backbone RMSD, ligand RMSD, per-residue RMSF,
   radius of gyration, H-bonds

### MDP Strategy (Modern Protocol)

| Phase | Barostat | Thermostat | Key Parameters |
|-------|----------|------------|----------------|
| NPT equilibration | **C-rescale** (τ=1.0 ps) | V-rescale (τ=0.1 ps) | Correct NPT ensemble (Bernetti & Bussi 2020) |
| Production | **Parrinello-Rahman** (τ=2.0 ps) | V-rescale (τ=1.0 ps) | Correct pressure fluctuations |

Force field: **amber14sb** (Maier et al. 2015, built into GROMACS 2026).
All MDPs use Verlet cutoff scheme, PME electrostatics (fourierspacing=0.12),
**1.0 nm** cutoffs (AMBER standard), LINCS constraints (order 4),
EnerPres dispersion correction. Thermostat coupling groups:
`Protein_LIG` / `Water_and_ions`. Equilibration uses position restraints
for both protein (`-DPOSRES`) and ligand (`-DPOSRES_LIG`).

### Stdin Handling

All interactive GROMACS tools (`genion`, `rms`, `hbond`, `trjconv`,
`make_ndx`) receive proper stdin via `subprocess.run(input=...)`. This
prevents the commands from hanging waiting for user input.

### GPU Auto-Detection

When `MDConfig.gpu_auto=True` (default), `nvidia-smi` is probed. If an NVIDIA
GPU is found, all `mdrun` calls get `-nb gpu -pme gpu -bonded gpu -update gpu`.

### Convergence Assessment

Backbone RMSD is split into halves; a 2-sample KS test determines if the
simulation has equilibrated. Converged = `p > 0.05 AND |mean_diff| < 0.5 Å`.

### Ligand Parameterization

RDKit (ETKDGv3 + MMFF) → OpenBabel (SDF→MOL2) → Antechamber (AM1-BCC/GAFF2) →
Parmchk2 → Acpype (AMBER→GROMACS) → `gmx genrestr` (ligand position restraints →
`posre_lig.itp`). No `--gen3d` in obabel (RDKit already did 3D).

## Organism Gate

`load_pairs()` blocks non-human targets (`organism != "Homo sapiens"`).
COX1_COX2 is blocked (COX1 = Ovis aries, COX2 = Mus musculus).
Bypass: `DUALDOCKING_ALLOW_NON_HUMAN=true`.

## SLURM Integration

For HPC clusters, `slurm/` provides shell-only integration with **zero changes
to core computation modules**. Two strategies:

- **Monolithic** (`slurm/monolithic/`): single node, N GPUs — runs stage
  runner with per-task parallelism (`run_stages.py --parallel N`) for s3/s4
- **Array** (`slurm/array/`): multi-node — each array element runs one
  (pair, mode, seed) task via `generate.py` + `score.py`

`slurm/task_matrix.py` maps `SLURM_ARRAY_TASK_ID` to (pair, mode, seed) using
the same `PipelineConfig.generate_tasks()` as the Python orchestrator. This
guarantees identical task ordering.

Both strategies submit the full SLURM dependency chain:
fetch → prepare → calibrate → generate → score → analyze → report/audit,
with optional gnina/md branches after score.

See [`pipeline.md`](pipeline.md#slurm-cluster-execution) for usage.

## Run Manifest

Each `run_full_experiment.py` invocation creates
`report/run_manifest_{run_id}.json` tracking parameters,
task status, timestamps, and config hashes.

## Process Boundaries

Internal pipeline entry points under `cli/` are invoked in-process through
`dualdocking.command_runner` + `dualdocking.script_dispatch` when possible.
This avoids spawning extra Python interpreters between stages.

`subprocess` is retained only where an external executable is required:
- Docker (`docker`)
- Boltz CLI (`boltz`) for built-in CLI scoring backend
- UniDock and related calibration binaries
- LaTeX (`pdflatex`, `bibtex`)
- Host probes (`nvidia-smi`, `git`)

## Offline Principle

All external data is fetched once during `cli/s0_fetch/fetch_all.py`.
After that, the entire pipeline (generation, scoring, analysis) runs offline.

| Data | Location | Source |
|------|----------|--------|
| Protein sequences | `output/data/targets/{id}.fasta` | UniProt |
| PDB structures | `output/data/targets/{pair}/{A,B}/{pdb_id}.pdb` | RCSB |
| MSA | `output/cache/boltz_msa/{id}.csv` | ColabFold API |
| Boltz checkpoints | `~/.boltz/` | Auto on first run |
| Score cache | `output/cache/boltz_scores/{target}.{ctx}.json` | Computed |
| NIM model cache | NIM container internal cache | NIM container |
| FREED++ models | `output/cache/freedpp_models/{key}/` | Generation |

## Troubleshooting

### FREED++: repeated `|z_a| > 10` / `|z_b| > 10`

Causes:
- Docker image not rebuilt after changes in `docker/freedpp/*`
- Stale model cache from old reward backend

Fix:

```bash
# 1. Rebuild image
docker build -f docker/freedpp/Dockerfile -t dualdocking/freedpp:latest .

# 2. Verify image exists
docker image inspect dualdocking/freedpp:latest --format '{{.Id}}'

# 3. Disable model cache for diagnostics
# configs/experiment.yaml -> generator.freedpp.use_model_cache: false
```

### Result Structure

```
results/
├── {pair}/
│   ├── single_a/        # generated.parquet, scores_A.parquet, scores_B.parquet, scored_joined.parquet
│   ├── single_b/
│   └── double/

report/
├── chapters/            # Editable LaTeX sections (.tex)
├── data/                # Aggregated data (generated)
├── figures/             # Generated plots
├── metrics.json         # Hit-rates per pair/mode
├── stats.json           # Bootstrap results, FDR, effect sizes
└── summary.txt          # Text analytics report
```
