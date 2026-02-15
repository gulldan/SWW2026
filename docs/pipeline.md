# Experiment Pipeline

> **Data format**: Pipeline outputs use Parquet (`.parquet`) as the default tabular
> format. CSV fallback is supported for backward compatibility — all read functions
> try `.parquet` first, then `.csv`.
>
> **I/O contracts**: For complete file paths, schemas, and data flow between stages
> see [docs/io_contracts.md](io_contracts.md).

All stages after Stage 0 run **fully offline** (no internet required).

## Stage 0: Fetch All External Data

Single entry point for everything that needs internet:

```bash
uv run python cli/s0_fetch/fetch_all.py
```

This downloads PDB structures, FASTA sequences, MSA data, calibration SMILES,
ChEMBL actives, GNINA binary, builds the Docker image, and prepares PDBQT
receptors. All HTTP downloads use exponential-backoff retry
(`dualdocking.data.http_utils`). Supports flags:

| Flag | Effect |
|------|--------|
| `--baseline-url URL` | Also download ZINC baseline archive |
| `--skip-docker-build` | Skip Docker image build |
| `--skip-chembl` | Skip ChEMBL actives fetch (Step 7) |
| `--skip-gnina` | Skip GNINA binary download (Step 8) |
| `--verify-only` | Check completeness without downloading |
| `--dry-run` | Print commands without executing |

### What gets fetched

| Step | What | Destination |
|------|------|-------------|
| Python packages | PyPI, GitHub | `.venv/` |
| FREED++ data | GitHub | `docker/freedpp/data/` |
| Docker image | NVIDIA, PyPI, conda | Docker cache |
| Calibration SMILES | HuggingFace | `output/data/calibration/` |
| Baseline ZINC | User URL (optional) | `output/data/baseline/` |
| PDB structures | RCSB | `output/data/targets/` |
| FASTA sequences | UniProt | `output/data/targets/` |
| MSA | ColabFold API | `output/cache/boltz_msa/` |
| PDBQT receptors | Local Docker | `output/data/targets/` |
| ChEMBL actives | ChEMBL API | `output/data/chembl/` |
| GNINA binary | GitHub releases | `output/bin/gnina` |
| Fetch manifest | — (generated) | `output/data/targets/fetch_manifest.json` |

## Stage 0.5: Calibration & Validation (offline, GPU)

```bash
uv run python cli/s2_calibrate/dock_calibration.py
uv run python cli/s2_calibrate/compute_normalization.py
uv run python cli/s2_calibrate/validate_redocking.py         # SCI-04: RMSD check
uv run python cli/s2_calibrate/enrichment_benchmark.py        # SCI-05: AUROC/BEDROC
```

## Stage 1 + 2: Full Experiment (offline)

Orchestrated run — generation, scoring, and analysis:

```bash
# Multi-GPU (requires 2+ GPUs):
EXPERIMENT_RUN_ID=run01 uv run python cli/pipeline/run_full_experiment.py \
  --resume --parallel 2 --with-post --latex-mode draft

# Single GPU (24 GB, e.g. RTX 4090):
EXPERIMENT_RUN_ID=run01 uv run python cli/pipeline/run_full_experiment.py \
  --resume --parallel 1 --with-post --latex-mode draft
```

> **Note:** `--parallel 2` requires 2+ GPUs. On a single GPU, FREED++ (576 MiB) and
> Boltz-2 (15.5 GB) compete for VRAM, causing OOM failures. Use `--parallel 1`.

Or split into stages:

```bash
# Stage 1: generate + score (all pairs x modes x seeds)
EXPERIMENT_RUN_ID=run01 uv run python cli/pipeline/run_full_experiment.py --resume --parallel 1

# Stage 2: aggregate + metrics + plots + report
EXPERIMENT_RUN_ID=run01 uv run python cli/pipeline/run_post_pipeline.py --latex-mode draft
```

### Filtering

```bash
# Single pair, subset of seeds
uv run python cli/pipeline/run_full_experiment.py --pair CDK2_ERK2 --seeds 42,137 --resume --with-post
```

## Stage Runner

Run individual stages or subsets with preflight input validation:

```bash
# List available stages
uv run python cli/pipeline/run_stages.py --list-stages

# Run only analysis + report
uv run python cli/pipeline/run_stages.py --stages analyze,report

# Run scoring for a specific pair
uv run python cli/pipeline/run_stages.py --stages score --pair CDK2_ERK2 --seeds 42

# Run everything
uv run python cli/pipeline/run_stages.py --stages all

# Dry run (show what would execute)
uv run python cli/pipeline/run_stages.py --stages all --dry-run
```

Stages and dependencies:

| Stage | Requires | Produces |
|-------|----------|----------|
| `fetch` | Internet | PDB, FASTA, SMILES, Docker image, ChEMBL actives, GNINA binary |
| `calibrate` | fetch outputs | `normalization.yaml`, calibration scores |
| `generate` | calibrate outputs | `generated.parquet` per run |
| `score` | generated molecules | `scored_joined.parquet` per run (configured scorer engine; default `boltz2`) |
| `analyze` | scored results | `all_results.parquet`, `metrics.json`, `stats.json` |
| `report` | analysis outputs | Figures, LaTeX report |
| `audit` | analysis outputs | `claim_audit.json` |
| `gnina` | scored results, GNINA binary | GNINA score Parquet per run |
| `md` | scored results, PDB structures | MD trajectories, RMSD/RMSF/Rg/energy CSVs, convergence assessment |

### Backend Selection

Pipeline backend choice is controlled by `configs/experiment.yaml`:

```yaml
generator:
  backend: freedpp  # default; or package.module[:callable]

scorer:
  engine: boltz2    # default; or package.module[:factory_or_object]
  backend: nim      # only for built-in boltz2: cli | nim
```

Legacy keys `generator.name` and `scorer.name` are still supported for backward compatibility.

### Execution Model

Pipeline orchestrators (`run_full_experiment.py`, `run_post_pipeline.py`,
`run_stages.py`, `prepare_all.py`, `fetch_all.py`) execute internal project
scripts through `dualdocking.command_runner.run_project_command`, which
dispatches known `cli/*.py` entry points in-process via
`dualdocking.script_dispatch`.

`subprocess` is still used at external boundaries only:
- Docker CLI (`docker build`, `docker run`, `docker image inspect`)
- Boltz CLI (`boltz predict`) when scorer backend is `cli`
- UniDock/other external docking binaries
- LaTeX toolchain (`pdflatex`, `bibtex`)
- System probes (`nvidia-smi`, `git`)

## Programmatic Run (Python API)

For notebook/service integration you can run the same workflows directly
from Python, without shell CLI wrappers:

```python
from dualdocking.api import run_full_experiment, run_post_pipeline, run_stages

run_full_experiment(dry_run=True, pairs=["CDK2_ERK2"], seeds=[42], with_post=False)
run_stages(stages="analyze,report", dry_run=True, skip_preflight=True)
run_post_pipeline(dry_run=True, latex_mode="off")
```

## Pilot Run (SCI-09)

Quick end-to-end check (1 pair x 3 modes x 2 seeds) before committing to full matrix:

```bash
uv run python cli/pipeline/run_pilot.py                       # CDK2_ERK2, seeds 42+137
uv run python cli/pipeline/run_pilot.py --pair AChE_BChE       # Custom pair
uv run python cli/pipeline/run_pilot.py --dry-run              # Dry run
uv run python cli/pipeline/run_pilot.py --verify-only          # Check existing artifacts
```

## Manual Step-by-Step (single pair)

For debugging or understanding the pipeline:

```bash
# Generate (3 modes)
uv run python cli/s3_generate/generate.py --pair CDK2_ERK2 --mode single_a
uv run python cli/s3_generate/generate.py --pair CDK2_ERK2 --mode single_b
uv run python cli/s3_generate/generate.py --pair CDK2_ERK2 --mode double

# Score
uv run python cli/s4_score/score.py --pair CDK2_ERK2 --mode single_a
uv run python cli/s4_score/score.py --pair CDK2_ERK2 --mode single_b
uv run python cli/s4_score/score.py --pair CDK2_ERK2 --mode double

# Score custom molecules (skip generation)
uv run python cli/s4_score/score.py --pair CDK2_ERK2 --mode double \
  --molecules my_compounds.csv

# Analyze
uv run python cli/s5_analyze/aggregate.py
uv run python cli/s5_analyze/compute_metrics.py
uv run python cli/s6_report/plot_results.py
uv run python cli/s6_report/render_molecules.py
uv run python cli/s5_analyze/summarize.py
```

## Full Matrix (all pairs)

```bash
for pair in CDK2_ERK2 AChE_BChE BACE1_GSK3B AChE_BACE1 GSK3B_CDK5 EGFR_HER2 BACE1_CDK5 AChE_MAOB; do
  for mode in single_a single_b double linear_equal biased_a selective_a; do
    uv run python cli/s3_generate/generate.py --pair $pair --mode $mode
    uv run python cli/s4_score/score.py --pair $pair --mode $mode
  done
done

uv run python cli/s5_analyze/aggregate.py
uv run python cli/s5_analyze/compute_metrics.py
uv run python cli/s6_report/plot_results.py
uv run python cli/s6_report/render_molecules.py
uv run python cli/s5_analyze/summarize.py
```

## Validation Scripts

| Script | SAP ref | What it checks |
|--------|---------|----------------|
| `s2_calibrate/validate_redocking.py` | SCI-04 | Co-crystal re-docking RMSD <= 2.0 A |
| `s2_calibrate/enrichment_benchmark.py` | SCI-05 | AUROC > 0.6, BEDROC > 0.3 per target |
| `s5_analyze/threshold_sensitivity.py` | SCI-06 | Robustness at T = -1, 0, +1 (Parquet output) |
| `s5_analyze/proxy_correlation.py` | SCI-07 | Spearman rho (docking vs Boltz-2) (Parquet output) |
| `s7_audit/claim_audit.py` | SCI-10 | Claim traceability (8 checks) |

## MD Validation (Optional, Post-Hoc)

> **Status: NOT YET VERIFIED.** The Docker image builds and the CLI runs, but
> no end-to-end MD run has completed successfully yet. The container has been
> observed to exit silently without producing output. Treat this stage as
> work-in-progress until the first successful `md_result.json` is produced.

Molecular dynamics validation for top candidate molecules. Not part of the
primary endpoint — used to strengthen Q1 publication credibility.

Requires GROMACS 2026.0, AmberTools, acpype, and OpenBabel. The recommended
approach is `--docker`, which bundles all tools in a container.
See [setup.md](setup.md) for full installation options.

### Prerequisites

1. **Docker image** `dualdocking/md:latest` must be built (one-time, ~20-30 min):
   ```bash
   docker build -f docker/md/Dockerfile -t dualdocking/md:latest .
   ```
   Check if already built: `docker images dualdocking/md`

2. **Scored results** must exist in `output/results/{pair}/{mode}/seed_{N}/scored_joined.parquet`.
   These are produced by Stage 1+2 (generate + score).

3. **PDB structures** must exist in `output/data/targets/{pair}/{A,B}/{pdb_id}.pdb`.
   Produced by Stage 0 (fetch).

### Docker Mode — Step by Step

```bash
# 1. Check what molecules would be selected (no Docker needed)
uv run python cli/s6_md/run_md.py --pair CDK2_ERK2 --dry-run

# 2. Quick smoke test: 1 molecule, 1 target mode, 1 ns simulation
#    (~5-10 min on RTX 4090 with GPU acceleration)
uv run python cli/s6_md/run_md.py --pair CDK2_ERK2 --docker \
  --mode single_a --top-n 1 --simulation-time 1

# 3. Full run: top-3 molecules, all modes, default 50 ns
#    (~2-4 hours per molecule×target on RTX 4090)
uv run python cli/s6_md/run_md.py --pair CDK2_ERK2 --docker

# 4. Shorter simulation (10 ns) for faster turnaround
uv run python cli/s6_md/run_md.py --pair CDK2_ERK2 --docker --simulation-time 10

# 5. With MM-GBSA binding free energy (requires gmx_MMPBSA in image)
uv run python cli/s6_md/run_md.py --pair CDK2_ERK2 --docker --mmpbsa

# 6. Disable GPU (CPU-only GROMACS)
uv run python cli/s6_md/run_md.py --pair CDK2_ERK2 --docker --no-gpu

# 7. Local mode (requires all tools in PATH — see setup.md Option B)
uv run python cli/s6_md/run_md.py --pair CDK2_ERK2
```

### What `--docker` Does Under the Hood

The CLI (`cli/s6_md/run_md.py --docker`) calls `dualdocking.validation.md_docker`:

1. **Builds image** if missing (`docker build -f docker/md/Dockerfile ...`)
2. **Runs container** with bind mounts:

   | Host | Container | Access |
   |------|-----------|--------|
   | `output/` | `/workspace/output` | read-write (results written here) |
   | `configs/` | `/workspace/configs` | read-only |
   | `dualdocking/` | `/workspace/dualdocking` | read-only |
   | `cli/` | `/workspace/cli` | read-only |
   | `pyproject.toml` | `/workspace/pyproject.toml` | read-only |

3. **Entrypoint** (`docker/md/entrypoint.sh`) sources GMXRC, sets up AmberTools/conda
   paths, then runs `python /workspace/cli/s6_md/run_md.py <args>` (without `--docker`).

The actual `docker run` command generated:
```bash
docker run --rm --gpus all \
  -v /mnt/data/SWW2026/output:/workspace/output \
  -v /mnt/data/SWW2026/configs:/workspace/configs:ro \
  -v /mnt/data/SWW2026/dualdocking:/workspace/dualdocking:ro \
  -v /mnt/data/SWW2026/cli:/workspace/cli:ro \
  -v /mnt/data/SWW2026/pyproject.toml:/workspace/pyproject.toml:ro \
  dualdocking/md:latest \
  --pair CDK2_ERK2 --top-n 3 \
  --config configs/pairs.yaml --experiment-config configs/experiment.yaml
```

### Running Docker Manually (for Debugging)

If the Python wrapper fails silently, run the container interactively:

```bash
# Interactive shell inside the MD container
docker run --rm -it --gpus all \
  -v /mnt/data/SWW2026/output:/workspace/output \
  -v /mnt/data/SWW2026/configs:/workspace/configs:ro \
  -v /mnt/data/SWW2026/dualdocking:/workspace/dualdocking:ro \
  -v /mnt/data/SWW2026/cli:/workspace/cli:ro \
  -v /mnt/data/SWW2026/pyproject.toml:/workspace/pyproject.toml:ro \
  dualdocking/md:latest bash

# Inside container — verify tools work:
gmx --version
antechamber --help
obabel --version
python -c "from dualdocking.validation.md_config import MDConfig; print('OK')"

# Run MD manually (see full output/errors):
python /workspace/cli/s6_md/run_md.py --pair CDK2_ERK2 \
  --mode single_a --top-n 1 --simulation-time 1

# Or run with --dry-run to check molecule selection:
python /workspace/cli/s6_md/run_md.py --pair CDK2_ERK2 --dry-run
```

### Pipeline Per Molecule×Target

22 GROMACS commands per molecule×target:

| Phase | Commands | What |
|-------|----------|------|
| **Setup** (14) | pdb2gmx, editconf, solvate, grompp×4, genion, mdrun×3, make_ndx | PDB repair → topology → solvation → ions → EM → NVT → NPT → production MD |
| **Preprocessing** (3) | trjconv ×3 | PBC fix → center → least-squares fit |
| **Analysis** (5) | rms ×2, rmsf, gyrate, hbond | Backbone RMSD, ligand RMSD, RMSF, Rg, H-bonds |

MDP protocol: **amber14sb** (Maier et al. 2015) + TIP3P water, **C-rescale**
(Bernetti & Bussi 2020, NPT equil) → Parrinello-Rahman (production),
V-rescale thermostat at 310 K, PME electrostatics (**rcoulomb=1.0 nm**,
AMBER standard), LINCS h-bonds constraints, dt=2 fs. Thermostat coupling:
`Protein_LIG` / `Water_and_ions`. Position restraints on both protein and
ligand during equilibration (`-DPOSRES -DPOSRES_LIG`).

### Output

Results go to `output/md/{pair}/{mode}/mol_{N}/target_{A,B}/`:

| File | Content |
|------|---------|
| `rmsd_backbone.csv` | time_ns, rmsd_A |
| `rmsd_ligand.csv` | time_ns, rmsd_A |
| `rmsf.csv` | residue, rmsf_A |
| `radius_gyration.csv` | time_ps, rg_A |
| `hbonds.csv` | time_ns, n_hbonds |
| `trajectory.xtc` | Fitted trajectory (PBC-corrected, centered) |
| `md_result.json` | All summary metrics (21 fields) |
| `mmpbsa.csv` | MM-GBSA output (only with `--mmpbsa`) |

Global summary: `output/md/md_summary.parquet` (appended after each run).

**Resumability**: If `md_result.json` exists for a molecule×target, that run is
skipped. To re-run, delete the specific `target_{A,B}/` directory.

### Timing Estimates (RTX 4090)

| Simulation | Time/molecule×target | Total for top-3 × 3 modes × 2 targets |
|------------|---------------------|---------------------------------------|
| 1 ns (smoke) | ~5-10 min | ~1.5-3 hours |
| 10 ns | ~30-60 min | ~9-18 hours |
| 50 ns (default) | ~2-4 hours | ~36-72 hours |

### Troubleshooting

- **Container exits silently**: Run interactively with `docker run -it ... bash`
  and execute `run_md.py` manually to see errors. Docker output now streams
  to the terminal (no longer captured silently).
- **`pdb2gmx` fails on missing atoms**: PDBFixer repairs are automatic but
  require `pdbfixer` + `openmm` (bundled in Docker image via conda-forge).
- **Docker output owned by root**: Container runs as root; output files in
  `output/md/` will be owned by root. Fix with `sudo chown -R $(id -u):$(id -g) output/md/`.
- **acpype/antechamber fails**: Some FREED++ molecules have unusual chemistry.
  The pipeline logs the error and moves to the next molecule.

See [io_contracts.md](io_contracts.md) (Stage 5c) for complete I/O specification.

## LaTeX Report

```bash
# Draft (.tex only)
uv run python cli/s6_report/generate_latex_report.py --mode draft \
  --chapters-dir report/chapters --chapters-manifest report/chapters/manifest.yaml

# Full PDF compilation
uv run python cli/s6_report/generate_latex_report.py --mode full
```

Output: `report/manuscript.tex`, `report/manuscript.pdf`.

## Logs

- Stage 1: `report/experiment_<run_id>_<utc>.log`
- Stage 2: `report/post_<run_id>_<utc>.log`
- Run manifests: `report/run_manifest_<run_id>.json`

## Data Migration (CSV to Parquet)

If you have existing results in CSV format, migrate them to Parquet:

```bash
# Preview what would be converted
uv run python cli/migrate_csv_to_parquet.py --dry-run

# Convert with verification, keep CSV originals
uv run python cli/migrate_csv_to_parquet.py --verify --keep-csv

# Convert and rename CSVs to .csv.bak
uv run python cli/migrate_csv_to_parquet.py --verify --remove-csv
```

Migrated files: `generated.parquet`, `scored_joined.parquet`, `scores_A/B.parquet`,
calibration scores, enrichment active scores, and report data files.
Not migrated: Boltz MSA cache, FREED++ training CSVs, Docker data files.

## Seeds

Default seeds from `configs/experiment.yaml`: `[42, 137, 256, 512, 1024]`.
Multiple seeds reduce variance. Metrics are computed across independent seeds.
For smoke/debug use one seed with small budget (`--epochs 2 --n-molecules 10`).
