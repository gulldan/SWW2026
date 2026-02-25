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
ChEMBL actives, GNINA binary, builds the FREED++ Docker image, and prepares
PDBQT receptors. All HTTP downloads use exponential-backoff retry
(`dualdocking.data.http_utils`). Supports flags:

| Flag | Effect |
|------|--------|
| `--baseline-url URL` | Also download ZINC baseline archive |
| `--skip-docker-build` | Skip Docker image build |
| `--skip-chembl` | Skip ChEMBL actives fetch (Step 7) |
| `--skip-gnina` | Skip GNINA binary download (Step 8) |
| `--verify-only` | Check completeness without downloading |
| `--dry-run` | Print commands without executing |

Stage 0 builds `dualdocking/freedpp:latest` only. To build all images
(including optional gnina and md), use `./scripts/build_docker_images.sh`
(see [setup.md](setup.md) for full options).

### What gets fetched

| Step | What | Destination |
|------|------|-------------|
| Python packages | PyPI, GitHub | `.venv/` |
| FREED++ data | GitHub | `docker/freedpp/data/` |
| FREED++ Docker image | NVIDIA, PyPI, conda | Docker cache |
| Calibration SMILES | HuggingFace | `output/data/calibration/` |
| Baseline ZINC | User URL (optional) | `output/data/baseline/` |
| PDB structures | RCSB | `output/data/targets/` |
| FASTA sequences | UniProt | `output/data/targets/` |
| MSA | ColabFold API | `output/cache/boltz_msa/` |
| PDBQT receptors | Local Docker | `output/data/targets/` |
| ChEMBL actives | ChEMBL API | `output/data/chembl/` |
| GNINA binary | GitHub releases | `output/bin/gnina` |
| Fetch manifest | — (generated) | `output/data/targets/fetch_manifest.json` |

## Stage 2: Calibration & Validation (offline, GPU)

```bash
uv run python cli/s2_calibrate/dock_calibration.py
uv run python cli/s2_calibrate/compute_normalization.py
uv run python cli/s2_calibrate/validate_redocking.py         # SCI-04: RMSD check
uv run python cli/s2_calibrate/enrichment_benchmark.py        # SCI-05: AUROC/BEDROC
```

Calibration docking automatically parallelizes across GPUs when multiple are
available — each target gets its own Docker container pinned to a specific GPU.
Targets without existing score files are skipped by `compute_normalization.py`
(incremental calibration).

## Stage 3 + 4: Full Experiment (offline)

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

### GPU Affinity Rules (important)

The scheduler assigns logical slots (`0..parallel-1`) and maps them to actual
CUDA device tokens before running generate/score commands.

- If `CUDA_VISIBLE_DEVICES` is unset: slot `0 -> GPU 0`, slot `1 -> GPU 1`, ...
- If `CUDA_VISIBLE_DEVICES=1,3`: slot `0 -> GPU 1`, slot `1 -> GPU 3`
- Each FREED++ container receives `FREEDPP_CUDA_DEVICE=<mapped_token>`
- Docker launch uses `--gpus device=<mapped_token>` by default

Examples:

```bash
# Use exactly one GPU
CUDA_VISIBLE_DEVICES=0 \
EXPERIMENT_RUN_ID=run01 uv run python cli/pipeline/run_full_experiment.py \
  --resume --parallel 1 --with-post --latex-mode draft

# Use a subset of physical GPUs on a shared server
CUDA_VISIBLE_DEVICES=2,5 \
EXPERIMENT_RUN_ID=run01 uv run python cli/pipeline/run_full_experiment.py \
  --resume --parallel 2 --with-post --latex-mode draft
```

Advanced override:

- `FREEDPP_DOCKER_GPUS` can force Docker `--gpus` (for example `all` or `device=1,3`)
- Keep it unset for normal pipeline scheduling; use only for manual/debug runs

Or split into stages:

```bash
# Stage 3+4: generate + score (all pairs x modes x seeds)
EXPERIMENT_RUN_ID=run01 uv run python cli/pipeline/run_full_experiment.py --resume --parallel 1

# Stage 5+6: aggregate + metrics + plots + report
EXPERIMENT_RUN_ID=run01 uv run python cli/pipeline/run_post_pipeline.py --latex-mode draft
```

### Filtering

The `--pair` and `--seeds` filters apply to **all** pipeline stages including
post-processing (aggregate, metrics, sensitivity, correlation, summarize).

```bash
# Single pair, subset of seeds
uv run python cli/pipeline/run_full_experiment.py --pair CDK2_ERK2 --seeds 42,137 --resume --with-post

# Multiple pairs, single seed, smoke test epochs
FREEDPP_EPOCHS=10 uv run python cli/pipeline/run_full_experiment.py \
  --pair EGFR_HER2 --pair MEK1_MEK2 --pair BRD4_BRD2 \
  --seeds 42 --parallel 2 --resume --with-post --latex-mode draft
```

### Training Resume

FREED++ automatically resumes from checkpoints when a previous run was interrupted:
- If output samples exist → skip (reuse previous results)
- If checkpoint files (`model_*.pth`) exist → resume training from latest checkpoint
- Otherwise → clean start

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `FREEDPP_EPOCHS` | from config (100) | Override epoch count (e.g. `10` for smoke tests) |
| `FREEDPP_NUM_SUB_PROC` | auto-detect by VRAM | Concurrent docking subprocesses |
| `REINVENT4_N_STEPS` | from config (500) | Override REINVENT4 RL step count |
| `EXPERIMENT_RUN_ID` | — | Run identifier for logs and manifests |

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
| `fetch` | Internet | PDB, FASTA, SMILES, FREED++ Docker image, ChEMBL actives, GNINA binary |
| `prepare` | fetch outputs | Prepared targets and baseline artifacts for downstream stages |
| `calibrate` | prepare outputs | `normalization.yaml`, calibration scores |
| `generate` | calibrate outputs | `generated.parquet` per run |
| `score` | generated molecules | `scored_joined.parquet` per run (configured scorer engine; default `boltz2`) |
| `analyze` | scored results | `all_results.parquet`, `metrics.json`, `stats.json` |
| `report` | analysis outputs | Figures, LaTeX report |
| `audit` | analysis outputs | `claim_audit.json` |
| `gnina` | scored results, GNINA binary | GNINA score Parquet per run |
| `md` | scored results, PDB structures | MD trajectories, RMSD/RMSF/Rg/SASA/PCA CSVs, convergence assessment |

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

# Analyze (filter to same pair)
uv run python cli/s5_analyze/aggregate.py --pair CDK2_ERK2
uv run python cli/s5_analyze/compute_metrics.py --pair CDK2_ERK2
uv run python cli/s6_report/plot_results.py
uv run python cli/s6_report/render_molecules.py
uv run python cli/s5_analyze/summarize.py --pair CDK2_ERK2
```

## Full Matrix (all pairs)

```bash
for pair in CDK2_ERK2 AChE_BChE BACE1_GSK3B AChE_BACE1 GSK3B_CDK5 EGFR_HER2 BACE1_CDK5 AChE_MAOB MEK1_MEK2 FGFR1_FGFR2 EZH2_EZH1 BRD4_BRD2; do
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
| `s7_audit/verify_pair.py` | — | Target pair verification (PDB, DOI/URL, sequence, domain identity, ChEMBL) |

## Target Pair Catalog Verification

Parses `configs/pairs.yaml` by default (with legacy `target_pairs_catalog.html`
support via `--catalog`) and verifies pair metadata against external APIs
(RCSB, doi.org, UniProt, ChEMBL). Includes both full-length
and **domain-specific** Needleman-Wunsch sequence identity.

### Why domain-specific identity?

Literature-cited similarity percentages (e.g. "~80% kinase domain ID") refer
to **functional domain** identity, not full-protein identity.  Drugs bind to a
specific pocket within a domain (kinase ATP site, NR LBD pocket), so the
relevant comparison is between those domain sequences.  Two proteins with only
20% full-length identity can share 80% identity in their kinase domain
ATP-pocket, allowing a single inhibitor to bind both.

The verification pipeline auto-discovers domain boundaries from UniProt
(e.g. "Protein kinase 712-979"), computes domain-specific NW identity, and
compares with catalog claims.  8 pairs with wrong PDB-to-UniProt
auto-resolution (fusion proteins, cryo-EM complexes, shared PDBs, polyproteins)
have manual UniProt overrides.

### Quick start

```bash
# Parse + verify DOI/URL for a single pair (by number)
uv run python cli/s7_audit/verify_pair.py --pair 1 --verify-doi

# By pair name
uv run python cli/s7_audit/verify_pair.py --pair "EGFR/HER2" --verify-pdb --verify-doi

# Full verification: PDB + DOI + sequence identity + domain identity + ChEMBL
uv run python cli/s7_audit/verify_pair.py --pair 1 --full

# Full verification with explicit domain ranges
uv run python cli/s7_audit/verify_pair.py --pair 1 \
  --verify-pdb --verify-doi --verify-sequence \
  --uniprot-a P00533 --uniprot-b P04626 \
  --domain-a 712-979 --domain-b 720-987

# Verify all configured pairs with all checks, save JSON
uv run python cli/s7_audit/verify_pair.py --all --full \
  --output-json report/data/verify_all.json

# Inject results into HTML catalog
python3 scripts/update_catalog_verification.py
```

### CLI options

| Option | Description |
|--------|-------------|
| `--pair N` or `--pair "NAME"` | Pair number or target names (e.g. `"EGFR/HER2"` or `"EGFR_HER2"`) |
| `--all` | Verify all pairs from the selected source |
| `--full` | Enable all checks (PDB + DOI + sequence + ChEMBL) |
| `--pairs-config PATH` | Path to `pairs.yaml` (default: `configs/pairs.yaml`) |
| `--catalog PATH` | Legacy HTML catalog path (if set, overrides `--pairs-config`) |
| `--verify-pdb / --no-verify-pdb` | Check PDB metadata against RCSB (resolution, organism, ligands) |
| `--verify-doi / --no-verify-doi` | Check DOI resolves via doi.org |
| `--verify-sequence / --no-verify-sequence` | Compute full-length + domain NW identity (auto-resolves UniProt) |
| `--verify-chembl / --no-verify-chembl` | Query ChEMBL bioactivity (auto-resolves target IDs) |
| `--uniprot-a ID`, `--uniprot-b ID` | UniProt accessions (override auto-resolution) |
| `--domain-a RANGE`, `--domain-b RANGE` | Domain range (e.g. `671-998`) to slice before alignment |
| `--chembl-a ID`, `--chembl-b ID` | ChEMBL target IDs (override auto-resolution) |
| `--output-json PATH` | Write JSON report to file |
| `--cache-dir PATH` | Disk cache directory (default: `output/cache/verify`) |
| `--no-cache` | Disable disk cache |

### What it checks

- **PDB metadata**: resolution, organism (Homo sapiens), gene names, bound ligands via RCSB API
- **DOI/URL resolution**: HTTP HEAD to doi.org or URL endpoint, reports status code and redirect
- **Full-length sequence identity**: Needleman-Wunsch alignment (BLOSUM62, affine gaps) on UniProt canonical sequences
- **Domain-specific identity**: Auto-discovers domain boundaries from UniProt features (kinase, LBD, SET, BD1, BH3, JmjC, sirtuin, catalytic, HDAC, PDE domains), computes NW identity on domain subsequences, compares with catalog literature claims
- **ChEMBL bioactivity**: IC50/Ki/Kd counts per target

### Module architecture

The verification library is split into focused modules:

| Module | Lines | Purpose |
|--------|-------|---------|
| `nw_align.py` | ~180 | BLOSUM62 + Needleman-Wunsch global alignment |
| `pair_catalog.py` | ~350 | HTML catalog parser, `PairRecord` dataclass |
| `domain_identity.py` | ~330 | Domain boundary discovery, selection, match quality |
| `pair_verify.py` | ~670 | Orchestrator: cache, PDB/DOI/ChEMBL/sequence verification |

All public symbols are re-exported from `pair_verify` for backward compatibility.

### Programmatic use

```python
from dualdocking.validation.pair_verify import parse_pairs_yaml, verify_pair, nw_align

pairs = parse_pairs_yaml("configs/pairs.yaml")  # -> list[PairRecord]
result = verify_pair(pairs[0], do_pdb=True, do_doi=True, do_sequence=True)

# Full-length identity
si = result["verification"]["sequence_identity"]["alignment"]
print(f"Full-seq: {si['identity_pct']}% ({si['matches']}/{si['aligned_length']})")

# Domain-specific identity
di = result["verification"]["domain_identity"]
if di["alignment"]:
    print(f"Domain: {di['alignment']['identity_pct']}% {di['domain_a']} vs catalog {di['catalog_claim_pct']}")

# Standalone NW alignment
aln = nw_align("ACDEFGHIK", "ACDXFGHIK")
print(f"{aln.identity_pct}% identity, {aln.matches}/{aln.aligned_length}")

# Direct submodule imports also work
from dualdocking.validation.nw_align import nw_align, AlignmentResult
from dualdocking.validation.pair_catalog import PairRecord, parse_catalog, parse_pairs_yaml
from dualdocking.validation.domain_identity import select_domains_for_pair
```

## MD Validation (Optional, Post-Hoc)

> **Status: VERIFIED** (2026-02-17). Smoke test passed end-to-end on
> CDK2_ERK2/single_a with 1 ns simulation time (~15 min on RTX 4090).
> Produces `md_result.json`, per-frame CSVs, and `trajectory.xtc`.

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
   These are produced by Stage 3+4 (generate + score).

3. **PDB structures** must exist in `output/data/targets/{pair}/{A,B}/{pdb_id}.pdb`.
   Produced by Stage 0 (fetch).

### Docker Mode — Step by Step

```bash
# 1. Check what molecules would be selected (no Docker needed)
uv run python cli/s9_md/run_md.py --pair CDK2_ERK2 --dry-run

# 2. Quick smoke test: 1 molecule, 1 target mode, 1 ns simulation
#    (~5-10 min on RTX 4090 with GPU acceleration)
uv run python cli/s9_md/run_md.py --pair CDK2_ERK2 --docker \
  --mode single_a --top-n 1 --simulation-time 1

# 3. Full run: top-3 molecules, all modes, default 50 ns
#    (~2-4 hours per molecule×target on RTX 4090)
uv run python cli/s9_md/run_md.py --pair CDK2_ERK2 --docker

# 4. Shorter simulation (10 ns) for faster turnaround
uv run python cli/s9_md/run_md.py --pair CDK2_ERK2 --docker --simulation-time 10

# 5. With MM-GBSA binding free energy (requires gmx_MMPBSA in image)
uv run python cli/s9_md/run_md.py --pair CDK2_ERK2 --docker --mmpbsa

# 6. Disable GPU (CPU-only GROMACS)
uv run python cli/s9_md/run_md.py --pair CDK2_ERK2 --docker --no-gpu

# 7. Local mode (requires all tools in PATH — see setup.md Option B)
uv run python cli/s9_md/run_md.py --pair CDK2_ERK2
```

### What `--docker` Does Under the Hood

The CLI (`cli/s9_md/run_md.py --docker`) calls `dualdocking.validation.md_docker`:

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
   paths, then runs `python /workspace/cli/s9_md/run_md.py <args>` (without `--docker`).

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
parmchk2 --help
obabel --version
python -c "from dualdocking.validation.md_config import MDConfig; print('OK')"

# Run MD manually (see full output/errors):
python /workspace/cli/s9_md/run_md.py --pair CDK2_ERK2 \
  --mode single_a --top-n 1 --simulation-time 1

# Or run with --dry-run to check molecule selection:
python /workspace/cli/s9_md/run_md.py --pair CDK2_ERK2 --dry-run
```

### Pipeline Per Molecule×Target

24 GROMACS commands per molecule×target:

| Phase | Commands | What |
|-------|----------|------|
| **Setup** (14) | pdb2gmx, editconf, solvate, grompp×4, genion, mdrun×3, make_ndx | PDB repair → topology → solvation → ions → EM → NVT → NPT → production MD |
| **Preprocessing** (3) | trjconv ×3 | PBC fix → center → least-squares fit |
| **Analysis** (7) | rms ×2, rmsf, gyrate, hbond, sasa, covar | Backbone RMSD, ligand RMSD, RMSF, Rg, H-bonds, SASA, PCA |

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
| `sasa.csv` | time_ns, sasa_A2 |
| `eigenvalues.csv` | mode, eigenvalue |
| `trajectory.xtc` | Fitted trajectory (PBC-corrected, centered) |
| `md_result.json` | All summary metrics (25 fields) |
| `mmpbsa.csv` | MM-GBSA output (only with `--mmpbsa`) |

Global summary: `output/md/md_summary.parquet` (appended after each run).

**Resumability**: If `md_result.json` exists for a molecule×target, that run is
skipped. To re-run, delete the specific `target_{A,B}/` directory.

### Timing Estimates (RTX 4090)

Measured on CDK2_ERK2/single_a (2026-02-17):

| Phase | 1 ns measured |
|-------|--------------|
| Ligand parameterization (antechamber AM1-BCC) | ~4 min |
| EM + NVT + NPT equilibration | ~3 min |
| Production MD | ~8 min |
| Trajectory preprocessing + analysis | ~10 sec |
| **Total per molecule×target** | **~15 min** |

Extrapolated estimates:

| Simulation | Time/molecule×target | Total for top-3 × 3 modes × 2 targets |
|------------|---------------------|---------------------------------------|
| 1 ns (smoke) | ~15 min | ~4.5 hours |
| 10 ns | ~45-60 min | ~13-18 hours |
| 50 ns (default) | ~3-4 hours | ~54-72 hours |

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

- Stage 3+4: `report/experiment_<run_id>_<utc>.log`
- Stage 5+6: `report/post_<run_id>_<utc>.log`
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

## SLURM Cluster Execution

For HPC clusters with SLURM workload manager. Two strategies are provided in
`slurm/`:

| Strategy | When to use | Parallelism |
|----------|-------------|-------------|
| **Monolithic** (`slurm/monolithic/`) | DGX / multi-GPU workstation | 1 node, N GPUs (task parallelism via `run_stages.py --parallel`) |
| **Array** (`slurm/array/`) | HPC cluster, many nodes | 1 GPU per task; array size from `slurm/task_matrix.py --count` (currently 150) |

### Quick Start

```bash
# Strategy A: single node, 4 GPUs
slurm/monolithic/submit.sh

# Strategy B: array jobs, max 8 concurrent
slurm/array/submit.sh

# Skip fetch stage (data already present)
slurm/array/submit.sh --skip-fetch

# Limit concurrent array tasks
slurm/array/submit.sh --max-concurrent=4
```

### Dependency Chain

Both strategies submit a core 8-stage chain plus optional validation stages:

```
fetch (s0, CPU) -> prepare (s1, CPU) -> calibrate (s2, 1 GPU) -> generate (s3, GPU) -> score (s4, GPU)
    -> analyze (s5, CPU) -> report (s6, CPU)
                         -> audit  (s7, CPU)
    -> gnina  (s8, optional, GPU)
    -> md     (s9, optional, GPU)
```

Report and audit both depend on analyze but run in parallel with each other.
GNINA and MD depend on score and run in parallel with analyze.

| sbatch | Stages | Partition | Resources | Wall time |
|--------|--------|-----------|-----------|-----------|
| `00_fetch` | s0 fetch | cpu | 4 CPU, 16G | 2h |
| `01_prepare` | s1 prepare | cpu | 4 CPU, 16G | 2h |
| `02_calibrate` | s2 calibrate | gpu | 1 GPU, 16 CPU, 64G | 4h |
| `03_generate` | s3 generate | gpu | 1 GPU, 16–64 CPU, 48–256G | 12h–7d |
| `04_score` | s4 score | gpu | 1 GPU, 16–64 CPU, 48–256G | 12h–7d |
| `05_analyze` | s5 analyze | cpu | 8 CPU, 32G | 2h |
| `06_report` | s6 report | cpu | 8 CPU, 32G | 2h |
| `07_audit` | s7 audit | cpu | 4 CPU, 16G | 1h |
| `08_gnina` | s8 GNINA cross-validation | gpu | 1 GPU, 8–16 CPU, 32–64G | 4–12h |
| `09_md` | s9 MD validation (GROMACS) | gpu | 1 GPU, 16 CPU, 64G | 3d |

Enable optional stages with `--with-gnina` and/or `--with-md`:

```bash
slurm/array/submit.sh --with-gnina --with-md
```

In the array strategy, GNINA runs as a full task array (same matrix as
generate/score), while MD runs as one task per active pair (`--pair-count`).

### Task Matrix (Array Strategy)

Each array element maps to a single (pair, mode, seed) combination:

```bash
# Total task count (active pairs × modes × seeds; currently 10 × 3 × 5 = 150)
uv run python slurm/task_matrix.py --count

# Active pair count (used by MD array stage)
uv run python slurm/task_matrix.py --pair-count

# Print full mapping
uv run python slurm/task_matrix.py --list

# Single task lookup (used inside sbatch)
uv run python slurm/task_matrix.py --index 0 --format env
# -> export TASK_PAIR=EGFR_HER2 TASK_MODE=single_a TASK_SEED=42

# Pair lookup by index (used by array MD stage)
uv run python slurm/task_matrix.py --pair-index 0 --format env
# -> export TASK_PAIR=EGFR_HER2
```

`task_matrix.py` uses the same `PipelineConfig.generate_tasks()` as the Python
orchestrator, so task ordering is identical.

### Retry Failed Tasks

SLURM `--requeue` handles preemption automatically. For manually retrying
specific failed array elements:

```bash
# Retry tasks 3, 17, and 42
sbatch --array=3,17,42 slurm/array/03_generate.sbatch
```

The pipeline's built-in resume logic (check for `scored_joined` / `generated`)
makes retry idempotent — completed tasks are skipped at no cost.

### NIM on Compute Nodes

`DD_USE_NIM` defaults to `auto` in `slurm/common.sh`:

- if `scorer.backend=nim` in `configs/experiment.yaml`, score stages start local
  NIM container(s) automatically;
- if backend is not `nim`, NIM lifecycle is skipped.

Force behavior explicitly when needed:

```bash
# Always manage local NIM containers in score stages
DD_USE_NIM=1 slurm/monolithic/submit.sh
DD_USE_NIM=1 slurm/array/submit.sh

# Never manage local NIM containers (expect external NIM if backend=nim)
DD_USE_NIM=0 slurm/array/submit.sh
```

NIM lifecycle is implemented in `slurm/nim/lifecycle.sh` and runs only in score
stages (`s4`). Requires Docker access on compute nodes and `NGC_API_KEY`.

### Shared Filesystem & Caching

- Boltz-2 cache uses `fcntl.flock` — safe on NFS (advisory locking)
- Array tasks targeting the same protein compete on cache files; flock serializes
- No additional configuration needed for shared filesystems

### Environment Variables

All SLURM scripts inherit from `slurm/common.sh`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `DD_ROOT` | Auto-detected from script location | Project root |
| `EXPERIMENT_RUN_ID` | `slurm_<timestamp>` | Shared run ID across all jobs |
| `DD_LOG_DIR` | `$DD_ROOT/slurm/logs/$EXPERIMENT_RUN_ID` | Log directory |
| `DD_CONTAINER_RUNTIME` | `docker` | Container runtime (`docker` or `apptainer`) |
| `DD_USE_NIM` | `auto` | NIM lifecycle mode: `auto`, `0`, or `1` |
| `DD_LATEX_MODE` | `off` | LaTeX report mode (`off`, `draft`, `full`) |

### Customization

Resource requests (`--mem`, `--gres`, `--time`, `--partition`) are set in each
`.sbatch` file and should be adjusted for your cluster. Common changes:

- **Partition names**: Replace `cpu`/`gpu` with your cluster's partition names
- **GPU type**: Change `--gres=gpu:4` to `--gres=gpu:a100:4` for specific GPU models
- **Module loads**: Uncomment and edit `module load` lines in `slurm/common.sh`

### Docker vs Apptainer

Most HPC clusters do not allow Docker. Conversion to Apptainer/Singularity
images is a one-time operation:

```bash
apptainer build freedpp.sif docker-daemon://dualdocking/freedpp:latest
apptainer build boltz2_nim.sif docker-daemon://nvcr.io/nim/mit/boltz2:1.5.0
```

Set `DD_CONTAINER_RUNTIME=apptainer` in `slurm/common.sh`. Full Apptainer
integration (container runtime shim) is planned as a separate task.
