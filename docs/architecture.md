# Architecture

Technical details of scoring, generation, and data flow.

## Score Conventions

**Lower is better** everywhere:

| Score | Source | Units | Role |
|-------|--------|-------|------|
| `d_A`, `d_B` | UniDock GPU | kcal/mol | RL reward signal |
| `a_A`, `a_B` | Boltz-2 | log10(IC50-like, uM) | Evaluation surrogate |

- IC50 = 1 nM = 0.001 uM -> log10 = -3 (strong binder)
- IC50 = 1 uM = 1.0 uM -> log10 = 0 (threshold)
- `combined_min` = worst-case = `max(score_A, score_B)` for lower-is-better

Boltz-2 output is **log10(IC50 in uM)**, NOT pIC50.
Convert: `pIC50 = 6 - log10(IC50_uM)`.

Full notation: see `docs/goal.md` (Section "Nотация").

## Boltz-2 Scorer

Boltz-2 has no Python API. Interaction is via CLI (`boltz predict`) as subprocess.

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
  --recycling_steps 3 --sampling_steps 200 --diffusion_samples 4 \
  --sampling_steps_affinity 200 --diffusion_samples_affinity 5 --override

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

Disk cache segmented by target: `output/cache/{target}.{ctx[:8]}.json`.

- Key: `SHA256(target || smi || ctx_hash || seq_hash || msa_hash)`
- Context includes: Boltz version, CLI args, accelerator, Docker digest
- I/O: `fcntl.flock` + atomic write (temp file + `os.replace`)
- Safe for concurrent single-machine access, NOT for NFS

## FREED++ Generator

FREED++ (Telepov et al., TMLR 2023) is an RL molecule generator (SAC + fragment assembly).
Runs in Docker due to dependency conflicts (Python 3.10, PyTorch 1.12, DGL 0.9).

### Data Flow

```
scripts/s3_generate/generate.py
  -> generation.freedpp.generate(config)
      1. _ensure_docker_image()       # build if absent
      2. _prepare_receptor()          # PDB -> PDBQT + binding box
      3. _run_freedpp()               # docker run --gpus all ... python main.py
      4. _parse_output_smiles()       # CSV from mounted volume
      -> list[str] SMILES
```

### Modes

| Mode | Reward | Description |
|------|--------|-------------|
| `single_a` | `-z_A` | Optimize for target A only |
| `single_b` | `-z_B` | Optimize for target B only |
| `double` | `-max(z_A, z_B)` | Worst-case over both targets |

Where `z_X = (d_X - mu_X) / sigma_X` (per-target z-score from calibration).

### Model Cache

Trained models saved in `output/cache/freedpp_models/{hash}/`.
Hash includes: reward mode, targets, epochs, steps_per_epoch, seed, etc.
Rerun with same params uses `--commands "sample"` (minutes vs hours).

## Organism Gate

`load_pairs()` blocks non-human targets (`organism != "Homo sapiens"`).
COX1_COX2 is blocked (COX1 = Ovis aries, COX2 = Mus musculus).
Bypass: `DUALDOCKING_ALLOW_NON_HUMAN=true`.

## Run Manifest

Each `run_full_experiment.py` invocation creates
`report/run_manifest_{run_id}.json` tracking parameters,
task status, timestamps, and config hashes.

## Offline Principle

All external data is fetched once during `scripts/s0_fetch/fetch_all.py`.
After that, the entire pipeline (generation, scoring, analysis) runs offline.

| Data | Location | Source |
|------|----------|--------|
| Protein sequences | `output/data/targets/{id}.fasta` | UniProt |
| PDB structures | `output/data/targets/{pair}/{A,B}/` | RCSB |
| MSA | `output/cache/boltz_msa/{id}.csv` | ColabFold API |
| Boltz checkpoints | `~/.boltz/` | Auto on first run |
| Score cache | `output/cache/{target}.{ctx}.json` | Computed |

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
│   ├── single_a/        # generated.csv, scores_A.csv, scores_B.csv, scored_joined.csv
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
