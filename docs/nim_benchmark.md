# Boltz-2 Backend Benchmark: CLI vs NIM

## Parameters (matched between backends)

- Model: Boltz 2.2.1
- `recycling_steps=3`, `sampling_steps=50`, `diffusion_samples=1`
- `sampling_steps_affinity=50`, `diffusion_samples_affinity=1`
- `step_scale=1.5` (boltz2 default; NIM default was 1.638, fixed)
- Diffusion is stochastic -- scores vary between runs

> These are the **production parameters** used in `boltz2.py:_BOLTZ_CLI_ARGS_BASE`.
> Reduced from initial 200/4 for computational feasibility; Boltz-2 is used as a
> relative ranking tool (not absolute affinity), so the quality trade-off is acceptable.

## Performance

RTX 4090, CDK2 (298 residues), 6 molecules.

| Mode | CLI | NIM | Speedup |
|------|-----|-----|---------|
| Batch (6 mol) | 14.0 s/mol | 5.1 s/mol | 2.8x |
| Single | 43.7 s/mol | 5.1 s/mol | 8.5x |

CLI single is slow because each `score_single` call spawns a new subprocess
(model loading ~35s overhead). CLI batch amortizes model loading across all
molecules. NIM keeps the model loaded in a persistent GPU server.

## Score Agreement

- Pearson r = 0.84, Spearman rho = 0.64, MAE = 0.25
- Within-backend noise: NIM std = 0.032 (5 runs, aspirin on CDK2)
- Differences are within stochastic noise for `diffusion_samples=1`

## TensorRT Backend

- TRT failed on RTX 4090: L40S engines (142 SM) deadlock on RTX 4090 (128 SM)
- TRT engines are GPU-specific, not portable between GPU models
- TRT available on H100/A100/L40S -- expected 1.2-2.3x over torch

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
- NIM processes requests sequentially (FastAPI, not Triton) -- concurrent requests give no speedup

## Backend Selection

- Set `BOLTZ_BACKEND=nim` and `BOLTZ_NIM_URL=http://localhost:8000`
- Default: `BOLTZ_BACKEND=cli` (subprocess `boltz predict`)
- NIM does not require MSA files (server handles internally)
- Cache files are separated by context hash (NIM/CLI never collide)
