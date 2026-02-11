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
uv run python scripts/s0_fetch/fetch_all.py
```

This downloads PDB structures, FASTA sequences, MSA data, calibration SMILES,
builds the Docker image, and prepares PDBQT receptors. Supports flags:

| Flag | Effect |
|------|--------|
| `--baseline-url URL` | Also download ZINC baseline archive |
| `--skip-docker-build` | Skip Docker image build |
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

## Stage 0.5: Calibration & Validation (offline, GPU)

```bash
uv run python scripts/s2_calibrate/dock_calibration.py
uv run python scripts/s2_calibrate/compute_normalization.py
uv run python scripts/s2_calibrate/validate_redocking.py         # SCI-04: RMSD check
uv run python scripts/s2_calibrate/enrichment_benchmark.py        # SCI-05: AUROC/BEDROC
```

## Stage 1 + 2: Full Experiment (offline)

Orchestrated run — generation, scoring, and analysis:

```bash
EXPERIMENT_RUN_ID=run01 uv run python scripts/pipeline/run_full_experiment.py \
  --resume --parallel 2 --with-post --latex-mode draft
```

Or split into stages:

```bash
# Stage 1: generate + score (all pairs x modes x seeds)
EXPERIMENT_RUN_ID=run01 uv run python scripts/pipeline/run_full_experiment.py --resume --parallel 2

# Stage 2: aggregate + metrics + plots + report
EXPERIMENT_RUN_ID=run01 uv run python scripts/pipeline/run_post_pipeline.py --latex-mode draft
```

### Filtering

```bash
# Single pair, subset of seeds
uv run python scripts/pipeline/run_full_experiment.py --pair CDK2_ERK2 --seeds 42,137 --resume --with-post
```

## Stage Runner

Run individual stages or subsets with preflight input validation:

```bash
# List available stages
uv run python scripts/pipeline/run_stages.py --list-stages

# Run only analysis + report
uv run python scripts/pipeline/run_stages.py --stages analyze,report

# Run scoring for a specific pair
uv run python scripts/pipeline/run_stages.py --stages score --pair CDK2_ERK2 --seeds 42

# Run everything
uv run python scripts/pipeline/run_stages.py --stages all

# Dry run (show what would execute)
uv run python scripts/pipeline/run_stages.py --stages all --dry-run
```

Stages and dependencies:

| Stage | Requires | Produces |
|-------|----------|----------|
| `fetch` | Internet | PDB, FASTA, SMILES, Docker image |
| `calibrate` | fetch outputs | `normalization.yaml`, calibration scores |
| `generate` | calibrate outputs | `generated.parquet` per run |
| `score` | generated molecules | `scored_joined.parquet` per run |
| `analyze` | scored results | `all_results.parquet`, `metrics.json`, `stats.json` |
| `report` | analysis outputs | Figures, LaTeX report |
| `audit` | analysis outputs | `claim_audit.json` |

## Pilot Run (SCI-09)

Quick end-to-end check (1 pair x 3 modes x 2 seeds) before committing to full matrix:

```bash
uv run python scripts/pipeline/run_pilot.py                       # CDK2_ERK2, seeds 42+137
uv run python scripts/pipeline/run_pilot.py --pair AChE_BChE       # Custom pair
uv run python scripts/pipeline/run_pilot.py --dry-run              # Dry run
uv run python scripts/pipeline/run_pilot.py --verify-only          # Check existing artifacts
```

## Manual Step-by-Step (single pair)

For debugging or understanding the pipeline:

```bash
# Generate (3 modes)
uv run python scripts/s3_generate/generate.py --pair CDK2_ERK2 --mode single_a
uv run python scripts/s3_generate/generate.py --pair CDK2_ERK2 --mode single_b
uv run python scripts/s3_generate/generate.py --pair CDK2_ERK2 --mode double

# Score
uv run python scripts/s4_score/score.py --pair CDK2_ERK2 --mode single_a
uv run python scripts/s4_score/score.py --pair CDK2_ERK2 --mode single_b
uv run python scripts/s4_score/score.py --pair CDK2_ERK2 --mode double

# Score custom molecules (skip generation)
uv run python scripts/s4_score/score.py --pair CDK2_ERK2 --mode double \
  --molecules my_compounds.csv

# Analyze
uv run python scripts/s5_analyze/aggregate.py
uv run python scripts/s5_analyze/compute_metrics.py
uv run python scripts/s6_report/plot_results.py
uv run python scripts/s6_report/render_molecules.py
uv run python scripts/s5_analyze/summarize.py
```

## Full Matrix (all pairs)

```bash
for pair in CDK2_ERK2 AChE_BChE BACE1_GSK3B AChE_BACE1 GSK3B_CDK5 EGFR_HER2 BACE1_CDK5 AChE_MAOB; do
  for mode in single_a single_b double; do
    uv run python scripts/s3_generate/generate.py --pair $pair --mode $mode
    uv run python scripts/s4_score/score.py --pair $pair --mode $mode
  done
done

uv run python scripts/s5_analyze/aggregate.py
uv run python scripts/s5_analyze/compute_metrics.py
uv run python scripts/s6_report/plot_results.py
uv run python scripts/s6_report/render_molecules.py
uv run python scripts/s5_analyze/summarize.py
```

## Validation Scripts

| Script | SAP ref | What it checks |
|--------|---------|----------------|
| `s2_calibrate/validate_redocking.py` | SCI-04 | Co-crystal re-docking RMSD <= 2.0 A |
| `s2_calibrate/enrichment_benchmark.py` | SCI-05 | AUROC > 0.6, BEDROC > 0.3 per target |
| `s5_analyze/threshold_sensitivity.py` | SCI-06 | Robustness at T = -1, 0, +1 (Parquet output) |
| `s5_analyze/proxy_correlation.py` | SCI-07 | Spearman rho (docking vs Boltz-2) (Parquet output) |
| `s7_audit/claim_audit.py` | SCI-10 | Claim traceability (8 checks) |

## LaTeX Report

```bash
# Draft (.tex only)
uv run python scripts/s6_report/generate_latex_report.py --mode draft \
  --chapters-dir report/chapters --chapters-manifest report/chapters/manifest.yaml

# Full PDF compilation
uv run python scripts/s6_report/generate_latex_report.py --mode full
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
uv run python scripts/migrate_csv_to_parquet.py --dry-run

# Convert with verification, keep CSV originals
uv run python scripts/migrate_csv_to_parquet.py --verify --keep-csv

# Convert and rename CSVs to .csv.bak
uv run python scripts/migrate_csv_to_parquet.py --verify --remove-csv
```

Migrated files: `generated.parquet`, `scored_joined.parquet`, `scores_A/B.parquet`,
calibration scores, enrichment active scores, and report data files.
Not migrated: Boltz MSA cache, FREED++ training CSVs, Docker data files.

## Seeds

Default seeds from `configs/experiment.yaml`: `[42, 137, 256, 512, 1024]`.
Multiple seeds reduce variance. Metrics are computed across independent seeds.
For smoke/debug use one seed with small budget (`--epochs 2 --n-molecules 10`).
