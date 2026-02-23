# Boltz-2 Backend Benchmark: CLI vs NIM

## How to Run

```bash
# Full benchmark (requires GPU + NIM container):
uv run pytest tests/scoring/test_boltz2_benchmark.py -m benchmark -v -n0 -s

# Configure runs (default 3):
BENCH_RUNS=5 uv run pytest tests/scoring/test_boltz2_benchmark.py -m benchmark -v -n0 -s

# Configure noise repetitions (default 10):
BENCH_NOISE_REPS=15 uv run pytest tests/scoring/test_boltz2_benchmark.py -m benchmark -v -n0 -s
```

## What Boltz-2 Does Per Molecule

Each molecule undergoes two sequential phases:
1. **Structure prediction** — diffusion-based protein-ligand complex generation
2. **Affinity prediction** — trained head estimates log10(IC50 in μM) from the predicted structure

Both phases are stochastic (diffusion sampling) → repeated runs produce slightly different scores.

## Parameters (matched between backends)

- Model: Boltz 2.2.1
- `recycling_steps=3`, `sampling_steps=50`, `diffusion_samples=1`
- `sampling_steps_affinity=50`, `diffusion_samples_affinity=1`
- `step_scale=1.638` (boltz2 default)
- Diffusion is stochastic — scores vary between runs

> These are the **production parameters** used in `boltz2.py:_BOLTZ_CLI_ARGS_BASE`.
> Reduced from initial 200/4 for computational feasibility; Boltz-2 is used as a
> relative ranking tool (not absolute affinity), so the quality trade-off is acceptable.

## Benchmark Design

The benchmark (`tests/scoring/test_boltz2_benchmark.py`) uses **20 diverse drug-like molecules** covering a range of sizes (6–28 heavy atoms) to ensure representative timing and score agreement results.

### Multi-run methodology

- **N_RUNS** (default 3, configurable via `BENCH_RUNS`): each backend scores the full batch N times
- **Warmup**: NIM gets 1 untimed warmup run to prime JIT/caches; CLI always spawns fresh subprocesses so warmup is unnecessary
- **Timing**: `time.perf_counter()` per run; reported as mean ± std seconds/molecule
- **Score averaging**: per-molecule scores are averaged across runs to reduce stochastic noise before computing agreement metrics (Pearson r, Spearman ρ, MAE)

### Tests

| Test | What it measures |
|------|-----------------|
| `test_cli_batch_multirun` | CLI batch: N runs, mean ± std s/mol |
| `test_nim_batch_multirun` | NIM batch: N runs (+ warmup), mean ± std s/mol |
| `test_comparison_multirun` | Both backends N runs each → timing, score agreement, BF16 analysis |
| `test_stochastic_noise` | Same molecule M times per backend → per-backend σ |
| `test_scaling` | Vary batch size (5, 10, 20) → throughput curve |

### Performance

CDK2 (298 residues), 20 molecules, 3 runs (+ 1 warmup for NIM).

| Backend | s/mol (mean ± std) | Total (20 mol) | Speedup |
|---------|-------------------|----------------|---------|
| CLI batch | 10.21 ± 0.08 | ~204s | 1.0x |
| NIM batch | 7.14 ± 0.04 | ~143s | 1.4x |

CLI batch amortizes model loading (~35s) across all molecules in the subprocess.
NIM keeps the model loaded in a persistent GPU server — no per-batch loading overhead,
but HTTP round-trip and sequential processing limit the speedup to 1.4x at batch=20.

> The old 2.8x speedup (6 molecules) was inflated: with only 6 molecules the
> 35s model-load overhead dominated CLI per-mol cost (14 s/mol). At 20 molecules
> CLI amortizes loading better → 10.2 s/mol, closing the gap.

### Score Agreement (averaged over 3 runs per backend)

- **Pearson r = 0.938**, Spearman ρ = 0.904
- **MAE = 0.160** log10(IC50 μM)
- Valid pairs: 20/20 (no NaN)
- Score range: [0.59, 3.16] log10(IC50 μM)

Within-backend stochastic noise (per-molecule σ across 3 runs):
- CLI: 0.008–0.073 (typical ~0.03–0.05)
- NIM: 0.012–0.107 (typical ~0.02–0.05)

Most inter-backend deltas are within stochastic noise. Notable outlier:
aspirin (+0.69) — BF16 quantization + stochastic variance combine on this small molecule.

## BF16 Quantization

NIM runs Boltz-2 in BF16 precision. BF16 has a 7-bit mantissa, which means:
- In the score range [2, 4), the quantization step is ≈ 0.016
- In the range [1, 2), the step is ≈ 0.008
- Scores returned by NIM snap to this grid

The benchmark detects BF16-grid alignment and reports the percentage of NIM scores that fall exactly on BF16 grid points, along with the maximum quantization step in the observed score range. This quantization is a systematic (not stochastic) source of CLI-vs-NIM disagreement.

## TensorRT Backend

- TRT engines are GPU-specific, not portable between GPU models
- TRT available on H100/A100/L40S — expected 1.2–2.3x over torch

## NIM Container

- Image: `nvcr.io/nim/mit/boltz2:1.5.0`
- Works on RTX 4090 (24GB) with torch fallback despite docs saying 48GB minimum
- VRAM: 7.7 GB idle
- Launch:
  ```
  docker run -d --gpus all --shm-size=16G \
    -e NGC_API_KEY=... \
    -v cache:/opt/nim/.cache \
    -p 8000:8000 \
    nvcr.io/nim/mit/boltz2:1.5.0
  ```
- NIM processes requests sequentially (FastAPI, not Triton) — concurrent requests give no speedup

## Backend Selection

- Set `BOLTZ_BACKEND=nim` and `BOLTZ_NIM_URL=http://localhost:8000`
- Default: `BOLTZ_BACKEND=cli` (subprocess `boltz predict`)
- NIM does not require MSA files (server handles internally)
- Cache files are separated by context hash (NIM/CLI never collide)
