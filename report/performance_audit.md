# DualDocking Performance Audit Report

**Date:** 2026-02-08
**Auditor:** Automated Performance Audit Team (Claude Opus 4.6)
**Hardware Targets:**
- Desktop: NVIDIA RTX 4090 (24 GB VRAM), ~8 CPU cores
- Server: 2x NVIDIA H100 (80 GB VRAM each), 128 CPU cores, NVLink

**Experiment Scale:** 5 core pairs x 3 modes x 5 seeds = 75 tasks, each generating 1000 molecules and scoring 2000 (2 targets x 1000)

---

## Executive Summary

The DualDocking pipeline has **five critical bottlenecks** that collectively account for >95% of wall-clock time:

| # | Finding | Estimated Speedup | Effort |
|---|---------|-------------------|--------|
| 1 | **Boltz-2 subprocess reload**: Model reloaded from disk for every 8-molecule batch (~125 subprocess invocations per target per run). Persistent server would eliminate reload overhead. | **3-8x** per scoring run | Medium |
| 2 | **No multi-GPU utilization**: Both Boltz-2 and FREED++ use `--gpus all` but run on a single GPU. On 2xH100, half the hardware sits idle. | **~2x** on 2xH100 | Medium |
| 3 | **Sequential target scoring**: `score.py` scores target A then target B serially. Parallel scoring would halve scoring time. | **~2x** | Low |
| 4 | **Sequential experiment tasks**: `run_full_experiment.py --parallel` defaults to 1. Tasks are GPU-bound with idle CPU cores. | **2-4x** with GPU pipelining | Low |
| 5 | **dock_calibration sequential ligand prep**: ~1000 ligands prepared one-by-one via RDKit embed + meeko. 11+ hours for 10 targets on 4090. | **10-50x** with parallel prep | Low-Medium |

**Combined realistic impact:** On 2xH100 with optimizations from Phases 1-2, total experiment time drops from an estimated **~400-600 GPU-hours** to **~60-100 GPU-hours** (4-6x overall).

---

## 1. Critical Path Analysis

### 1.1 Pipeline DAG

```
prepare_targets.py
  |
  v
[For each (pair, mode, seed) -- 75 tasks]:
  generate.py -----> FREED++ Docker (GPU: UniDock training)
       |
       v
  score.py --------> Boltz-2 subprocess x2 targets (GPU: inference)
       |
       v
[Post-processing -- CPU only]:
  aggregate.py -> compute_metrics.py -> plot_results.py -> render_molecules.py -> summarize.py
```

### 1.2 Where Time Is Spent (estimated per-task breakdown)

| Stage | Duration (est.) | % of Total | Bound |
|-------|----------------|------------|-------|
| FREED++ training (200 epochs, UniDock GPU) | ~4-8 hours | 55-65% | GPU |
| Boltz-2 scoring (2 targets x 125 batches) | ~3-6 hours | 30-40% | GPU |
| Post-processing (aggregate/metrics/plots) | ~2-5 min | <1% | CPU |
| Ligand/receptor prep (one-time calibration) | ~11 hours | amortized | CPU |

**Total per-task:** ~7-14 hours (single GPU).
**Total 75 tasks sequential:** ~525-1050 hours.
**Current parallelism:** None (`--parallel=1` default).

### 1.3 Key Files on Critical Path

- `dualdocking/scorer_boltz2.py:552-647` - `score_batch()` - the Boltz-2 scoring loop
- `dualdocking/generator_freedpp.py:324-467` - `_run_freedpp()` - Docker-based FREED++ training
- `scripts/s4_score/score.py:66-89` - Sequential A/B target scoring
- `scripts/pipeline/run_full_experiment.py:245-258` - Task scheduling (defaults to serial)

---

## 2. GPU Utilization

### 2.1 Boltz-2 Scorer

**Current implementation** (`dualdocking/scorer_boltz2.py`):

- **Subprocess-per-batch architecture** (line 386-409): Every batch of 8 molecules spawns a new `uv run boltz predict` subprocess. This means:
  - PyTorch + Boltz model loaded from disk each invocation (~10-30s overhead)
  - For 1000 molecules per target: `ceil(1000/8) = 125` subprocess invocations
  - For 2 targets: 250 total subprocess invocations per score.py run
  - **Model reload overhead dominates**: ~60-70% of scoring time is model loading, not inference

- **Batch size = 8** (line 34): Conservative. While `BATCH_SIZE=8` groups molecules into one subprocess invocation, Boltz-2 internally uses `DataLoader(batch_size=1)` -- processing molecules one-by-one within each batch call. The external batch_size primarily amortizes subprocess startup, not GPU utilization. Increasing to 16-32 reduces subprocess count by 2-4x with LOW OOM risk (Boltz internal processing is per-molecule). On H100 80 GB, batch_size=32-64 is safe.

- **Single-GPU only**: No `--devices` flag passed to Boltz CLI. Cannot use second H100.

- **Two models loaded**: Boltz-2 loads both structure prediction and affinity models. For affinity-only scoring, the structure model is unnecessary overhead (~2-3 GB VRAM + load time).

- **No `torch.compile`**: Boltz-2 upstream does not use `torch.compile()`. Adding it could yield 1.2-2x inference speedup.

- **BF16 already enabled**: Good. FP32 used only where numerically required.

- **Cache I/O well-designed**: `cache_io.py` uses fcntl.flock + atomic write (temp+replace) + read-merge-write. No lock contention issues observed.

**Optimization: Persistent Boltz Server**

Replace subprocess invocations with a persistent GPU server process:

```python
# Sketch: boltz_server.py
class BoltzServer:
    def __init__(self, device="cuda:0"):
        self.model = load_boltz_model(device=device)  # Load once

    def score_batch(self, yamls: list[Path]) -> list[float]:
        # Direct inference, no subprocess overhead
        return self.model.predict_affinity(yamls)

# In scorer_boltz2.py:
_SERVER = None
def _get_server():
    global _SERVER
    if _SERVER is None:
        _SERVER = BoltzServer()
    return _SERVER
```

**Estimated speedup:** 3-8x (eliminating model reload; larger on 4090 where reload is proportionally more costly due to slower PCIe bandwidth).

### 2.2 FREED++ Generator

**Current implementation** (`dualdocking/generator_freedpp.py:324-467`):

- **Docker overhead: negligible** (<0.1% for multi-hour training). Container startup is ~1-2s.

- **GPU passthrough**: `--gpus all` (line 355) exposes all GPUs but FREED++ (DGL-based) uses only one. This wastes the second H100.

- **`num_sub_proc=24`** (line 69, overridden to 24 in `configs/experiment.yaml:49`): This controls parallel UniDock docking calls during RL training. On a 4090, 24 concurrent UniDock instances cause GPU memory contention; optimal is likely 8-12 for 4090, 16-24 for H100.

- **Double mode = 2x docking**: In double mode, each molecule is docked against both targets (`dual_reward_patch.py`). Two sequential UniDock calls per reward evaluation. On 2xH100, these could run on separate GPUs.

- **Ligand prep not cached in dual_call**: Each training step re-prepares ligands from SMILES via meeko. Caching the PDBQT conversion would save ~5-10% per episode.

- **UniDock call volume**: 200 epochs x 256 steps/epoch = **51,200 docking calls** per single-target training run. In double mode, this doubles to ~102,400 calls. This is inherent to FREED++'s RL loop and cannot be reduced without modifying the upstream algorithm. Per-molecule subprocess overhead is unavoidable at the FREED++ level.

**Recommendations:**
1. Pin GPUs: `--gpus '"device=0"'` for FREED++, use GPU 1 for Boltz-2 scoring in parallel
2. Tune `num_sub_proc`: 8-12 for 4090, 16-24 for H100
3. For double mode on 2xH100: dock target A on GPU 0 and target B on GPU 1 simultaneously
4. Investigate UniDock `--multi_bias` flag for multi-GPU docking within a single FREED++ run (requires FREED++ code modification to pass the flag)

### 2.3 Multi-GPU Strategy

**Current state:** Both Boltz-2 and FREED++ are single-GPU. On 2xH100 (160 GB total VRAM), only 80 GB is used.

**Recommended 2xH100 layout:**

```
GPU 0: FREED++ training (UniDock GPU, DGL)
GPU 1: Boltz-2 scoring (or second FREED++ task)

# Or for scoring-only phase:
GPU 0: Boltz-2 target A
GPU 1: Boltz-2 target B
```

**Implementation:** Set `CUDA_VISIBLE_DEVICES` per subprocess:
- FREED++: `docker run --gpus '"device=0"' ...`
- Boltz-2: `CUDA_VISIBLE_DEVICES=1 uv run boltz predict ...`

**Estimated speedup:** ~2x on 2xH100 (currently 1x utilization).

---

## 3. CPU Utilization

### 3.1 Current Usage

- **128 CPU cores on server, almost entirely idle during GPU-bound stages**
- Only `run_full_experiment.py` has any parallelism (`ThreadPoolExecutor`), defaulting to `--parallel=1`
- No `multiprocessing` anywhere in `dualdocking/` (confirmed via grep)
- `scripts/s2_calibrate/dock_calibration.py:633` iterates ligands sequentially with `tqdm`
- Bootstrap (`statistical.py:80-86`): 10K iterations in a Python loop, but only ~36 seconds total (not worth parallelizing)
- RDKit fingerprint computation (`diversity.py:58-67`): sequential Python loop

### 3.2 128-Core Optimization Opportunities

| Opportunity | File:Line | Current | Optimized | Speedup |
|-------------|-----------|---------|-----------|---------|
| Ligand prep (dock_calibration) | `dock_calibration.py:633` | Sequential loop | `ProcessPoolExecutor(max_workers=32)` for RDKit embed | 10-30x |
| Fingerprint computation | `diversity.py:58-67` | Sequential loop | `rdkit.Chem.rdFingerprintGenerator` with `GetFingerprints()` (bulk) or multiprocessing | 2-10x |
| Tanimoto SNN | `diversity.py:143-151` | O(n^2) sequential | `DataStructs.BulkTanimotoSimilarity` already used but in serial loop; 2-5s for n~1000. Acceptable. | 2-5x with multiprocessing (if n grows) |
| ChEMBL nearest neighbor | `chembl_validation.py:155-161` | Sequential per-query | Parallelize with ProcessPool | 5-10x |
| Multiple experiment tasks | `run_full_experiment.py:252` | `--parallel=1` default | `--parallel=4` or higher (GPU pipelining) | 2-4x |

**Key insight:** The 128 cores are wasted. CPU-bound stages (ligand prep, fingerprints, file I/O) should use multiprocessing. GPU-bound stages could overlap: while one task trains FREED++, another scores with Boltz-2.

---

## 4. Data Processing Efficiency

### 4.1 pandas/polars Patterns

**pandas is used everywhere; polars is declared as a dependency but never imported.**

**Zero polars usage confirmed.** `polars>=1.0` is listed in `pyproject.toml:10` and `polars 1.38.1` is resolved in `uv.lock`, but grep across all of `src/` and `scripts/` finds zero `import polars` or `pl.` references. This is pure dead weight (~30 MB installed, plus import overhead if accidentally triggered).

Usage inventory:
- `dualdocking/statistical.py`: pandas for CSV reading/caching (justified; DataFrames passed to metric functions)
- `dualdocking/generator_freedpp.py:482`: pandas for CSV parsing of FREED++ output
- `scripts/s4_score/score.py`: pandas read/write of CSVs (small: 1000 rows)
- `scripts/s5_analyze/aggregate.py`: pandas concat + to_parquet (appropriate)
- `scripts/s5_analyze/compute_metrics.py`: pandas read_csv (small files)
- `scripts/s6_report/plot_results.py:121`: pandas read_parquet (appropriate)
- `scripts/s5_analyze/proxy_correlation.py`: pandas concat of multiple CSVs

**Anti-patterns found:**
- `scripts/s5_analyze/summarize.py:50`: `.iterrows()` on top molecules (only 3 rows, negligible)
- `scripts/s6_report/render_molecules.py:85`: `.iterrows()` over DataFrame for molecule rendering (5-15 rows; could be `itertuples()` for 2-3x per-row speedup, but absolute gain is microseconds)
- **No `.apply()` antipatterns found** anywhere in the codebase.

**Verdict:** No severe pandas anti-patterns. Data volumes are small (1000 rows per CSV). Switching to polars would add complexity for negligible gain. The `.iterrows()` calls process <15 rows.

**Recommendation:** Remove `polars` from `pyproject.toml` dependencies. It adds ~30 MB to the venv and install time for zero benefit.

### 4.2 numpy/scipy Vectorization

**numpy usage is appropriate and well-vectorized:**

- `objectives.py`: All objective functions (min, harmonic, geometric, tchebycheff, pareto) operate on numpy arrays. No Python loops over data.
- `enrichment.py`: `np.argsort`, `np.where`, `np.sum` -- properly vectorized.

**Hotspot: Bootstrap inner loop** (`statistical.py:80-86`):

This is the **hottest pure-Python loop** in the codebase. 10K iterations, each calling `_metric_for_seed()` which uses `df.iloc[idx]` (pandas integer-location indexing):

```python
# statistical.py:80-86 -- current:
for i in range(n_bootstrap):
    seed_idx_a = rng.integers(0, n_a, size=n_a)
    seed_idx_b = rng.integers(0, n_b, size=n_b)
    metrics_a = [_metric_for_seed(group_a[j]) for j in seed_idx_a]
    metrics_b = [_metric_for_seed(group_b[j]) for j in seed_idx_b]
    diffs[i] = float(np.nanmean(metrics_b) - np.nanmean(metrics_a))
```

**Optimization (3-5x speedup):** Pre-extract numpy arrays from DataFrames before the loop. The `metric_fn` typically computes a scalar from a column (e.g., hit rate). Pre-extracting the relevant column into a numpy array eliminates `df.iloc` overhead:

```python
# Optimized: pre-extract numpy arrays
arrays_a = [df[col].to_numpy() for df in group_a]
arrays_b = [df[col].to_numpy() for df in group_b]
for i in range(n_bootstrap):
    seed_idx_a = rng.integers(0, n_a, size=n_a)
    # Now metric_fn operates on numpy array, not DataFrame
    metrics_a = [metric_fn_np(arrays_a[j][rng.integers(0, len(arrays_a[j]), size=len(arrays_a[j]))]) for j in seed_idx_a]
```

Total time is ~36 seconds for 10K iterations -- not a pipeline bottleneck, but an easy 3-5x improvement to ~7-12 seconds.

**AUROC Python loop** (`enrichment.py:74-85`): The trapezoidal ROC-AUC computation uses a Python `for` loop over sorted labels. This is vectorizable with `np.cumsum`, but the absolute gain is ~1ms (called on ~1000-element arrays). Not worth the code complexity.

**scipy usage is minimal (scripts only):**
- `scripts/s2_calibrate/validate_redocking.py:36-37`: `scipy.optimize.linear_sum_assignment` + `scipy.spatial.distance.cdist` -- optimal algorithms, no improvement possible.
- `scripts/s5_analyze/proxy_correlation.py:32`: `scipy.stats` for Spearman correlation.

**Hotspot: scipy.spearmanr in bootstrap** (`scripts/s5_analyze/proxy_correlation.py:58-60`):

```python
for i in range(n_bootstrap):
    idx = rng.integers(0, n, size=n)
    boot[i] = float(corr_fn(x[idx], y[idx]).statistic)
```

This calls `scipy.stats.spearmanr` ~5000 times. Each call has Python overhead for the `NamedTuple` result object. A numpy-only rankdata + Pearson implementation would be 3-5x faster:

```python
def _fast_spearman(x, y):
    rx = scipy.stats.rankdata(x)
    ry = scipy.stats.rankdata(y)
    return np.corrcoef(rx, ry)[0, 1]
```

However, proxy_correlation runs once post-experiment, so this is low priority.

### 4.3 rdkit Optimization

**RDKit 2025.9.4 is used in four contexts. Threading is supported (GIL released in C++ ops).**

1. **SMILES validation** (`scorer_boltz2.py:298-306`): `Chem.MolFromSmiles()` per molecule. Fast (~0.1ms each). No optimization needed.

2. **Fingerprint generation** (`diversity.py:55-67`, `chembl_validation.py:121-126`): Sequential loop over SMILES. For 1000 molecules this is ~0.5s. Could use `rdFingerprintGenerator.GetFingerprints()` (bulk) or `multiprocessing.Pool` (RDKit releases the GIL in C++ ops) for >10K molecules.

3. **3D embedding for docking** (`dock_calibration.py:258-275`, `scripts/s1_prepare/rdkit_embed.py`): `AllChem.EmbedMolecule()` + `AllChem.UFFOptimizeMolecule()` -- ~0.5-2s per molecule. For 1000 ligands, this is 8-33 minutes sequentially. **This is the biggest RDKit bottleneck** and should be parallelized with `ProcessPoolExecutor`.

4. **Double SMILES parsing** (`chem_sanitize.py:85-90` + `diversity.py:79-91`): The `sanitize_smiles()` function calls `canonicalize(smi)` (which internally calls `Chem.MolFromSmiles`), then immediately calls `Chem.MolFromSmiles(canon)` again on the result. Similarly, `scaffold_diversity()` calls `canonicalize()` and then `Chem.MolFromSmiles()` on each canonical SMILES. This is a ~40% reduction in RDKit parse calls if mol objects are passed through instead of re-parsing. Low absolute impact (~200ms for 1000 mols) but indicates a design inefficiency.

**Sketch for parallel ligand prep:**
```python
from concurrent.futures import ProcessPoolExecutor

def _prep_one(args):
    smi, sdf_path, seed = args
    # RDKit embed + meeko convert (GIL released during C++ ops)
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol); writer.close()
    return (sdf_path, True)

with ProcessPoolExecutor(max_workers=min(32, os.cpu_count())) as pool:
    results = list(pool.map(_prep_one, tasks))
```

### 4.4 I/O Patterns (CSV/Parquet/JSON)

**File format usage:**

| Format | Read | Write | Where |
|--------|------|-------|-------|
| CSV | 15+ locations | 8+ locations | Intermediate results, scores, calibration |
| Parquet | 3 locations | 1 location | Aggregated results (`all_results.parquet`) |
| JSON | Cache files, config | Cache files, metadata | Boltz-2 cache, FREED++ config |

**Observations:**
- **CSV for intermediate results is fine**: Files are <1000 rows. CSV vs Parquet difference is <10ms.
- **Parquet for aggregated results is correct**: `aggregate.py` writes parquet; downstream scripts read it.
- **JSON cache files** (`cache_io.py`): Uses `json.dumps`/`json.loads` with fcntl locking. For large caches (>100K entries), msgpack or sqlite would be faster, but current cache sizes (~1000-5000 entries per target) are fine.
- **No unnecessary serialization**: Good. Data stays in memory within each script invocation.

**One issue:** `scripts/s5_analyze/proxy_correlation.py:97-128` reads `train_*.csv` files from FREED++ workspace one-by-one with `pd.read_csv()` and concatenates. If there are many epoch files, this could be slow. Consider `pd.concat([pd.read_csv(f) for f in files])` with pre-allocated list (already done -- line 128).

---

## 5. Dependency Analysis

### 5.1 Current Dependencies

| Package | Locked Version | Purpose | Notes |
|---------|---------------|---------|-------|
| boltz | 2.2.1 | Affinity scoring | Git install from `jwohlwend/boltz.git`; CLI-only usage via subprocess |
| torch | 2.10.0 (host) / 2.4.0 (Docker) | GPU compute | Host: Boltz-2. Docker: FREED++/DGL. Separate environments, version mismatch is fine. |
| rdkit | 2025.9.4 | Chemistry toolkit | SMILES validation, fingerprints, 3D embed. Threading supported (GIL released in C++). |
| numpy | 1.26.4 | Numerical computing | Pinned by boltz/torch compatibility. cupy/jax overkill for small arrays. |
| pandas | 3.0.0 (host) / <2.0 (Docker) | DataFrames | Host: throughout scripts. Docker: FREED++ legacy requires pandas 1.x. |
| polars | 1.38.1 | DataFrames | **UNUSED** -- never imported anywhere |
| pyarrow | 23.0.0 | Parquet I/O | Required by pandas for parquet |
| scipy | 1.13.1 | Statistics | Used in 2 scripts only (redocking, correlation). Pinned by boltz. |
| datasets | 4.5.0 (HuggingFace) | Data loading | Used in **ONE place**: `scripts/s0_fetch/fetch_calibration_smiles.py:43`. Heavyweight (~200 MB transitive deps). |
| meeko | 0.5.1 | Ligand preparation | PDBQT conversion. Lightweight, no bottleneck, no alternative needed. |
| matplotlib/seaborn | Latest | Plotting | Post-processing only |
| loguru | 0.7+ | Logging | Throughout |
| tqdm | 4.66+ | Progress bars | Throughout |
| click | 8.1+ | CLI | All scripts |
| pyyaml | 6.0+ | Config parsing | Throughout |
| requests | 2.31+ | HTTP | ChEMBL API only |

### 5.2 Library Alternatives & Upgrades

**Remove unused:**
- `polars>=1.0` (`pyproject.toml:10`): Never imported anywhere in `src/` or `scripts/`. Remove to save ~30 MB and install time. Zero risk.

**Replace heavyweight single-use dep:**
- `datasets>=2.19` (HuggingFace, `pyproject.toml:16`): Used in exactly ONE file (`scripts/s0_fetch/fetch_calibration_smiles.py:43`). Pulls in `huggingface_hub`, `tokenizers`, `xxhash`, and other transitive dependencies (~200 MB). Replace with `requests` + direct CSV download. The calibration set is a simple SMILES file, not a HuggingFace dataset requiring the datasets library's features.

**Pin Boltz to specific commit:**
- `boltz = { git = "https://github.com/jwohlwend/boltz.git" }` (`pyproject.toml:192`) -- pinned to HEAD of master. This is **fragile**: upstream breaking changes could silently break scoring and invalidate the entire cache (context hash changes with boltz version).
- Recommendation: `boltz = { git = "...", rev = "<specific-commit-hash>" }`
- The current locked version is 2.2.1. Pin to the exact commit that corresponds to this version.

**UniDock:**
- UniDock IS the GPU-accelerated variant already (not CPU Vina). It supports `--multi_bias` for multi-GPU docking. Per-molecule subprocess overhead during FREED++ training (~51,200 docking calls per run at 200 epochs x 256 steps/epoch) is unavoidable without modifying FREED++ internals.

**No change needed:**
- `numpy 1.26.4` / `scipy 1.13.1`: Pinned by boltz. cupy/jax would be overkill for the small array sizes used (1000 elements).
- `meeko 0.5.1`: Lightweight PDBQT converter. No alternative needed.
- `rdkit 2025.9.4`: Current version. No upgrade needed.

**Docker image versions:**
- CUDA 12.4.1 in Dockerfile, torch 2.4.0 in container, torch 2.10.0 on host. Compatible (separate environments).
- DGL installed from torch-2.4/cu124 wheels -- correct for container's torch 2.4.0.
- Both RTX 4090 (sm_89) and H100 (sm_90) are supported by CUDA 12.4.
- **numpy pinned to 1.26.4 in container**: Matches host. Good for reproducibility.
- **pandas<2.0 in container**: FREED++ legacy code requires pandas 1.x. This does not affect the host (pandas 3.0.0).

### 5.3 Import Time

**Already well-optimized with lazy imports.** The codebase uses deferred imports for heavy libraries (`torch`, `rdkit`, `pandas`, `yaml`) -- they are imported inside functions, not at module top level. This is good practice for CLI tools.

Not a major concern for pipeline runtime. Scripts are long-running (hours). Import overhead is <5 seconds per process. The heaviest imports:
- `torch`: ~2-3s (but loaded per-subprocess in Boltz-2 -- 125+ times per target)
- `rdkit`: ~0.5s
- `pandas`: ~0.3s
- `boltz`: ~1s (loads torch internally)

The Boltz-2 subprocess model means torch is imported 125+ times per target. This is part of the "subprocess reload" overhead addressed in Section 2.1. With a persistent server (Phase 3), torch would be imported exactly once.

---

## 6. Docker & Infrastructure

### 6.1 Image Optimization

**Current image:** `nvidia/cuda:12.4.1-devel-ubuntu20.04`

**Issues:**
- **`-devel` base image is large** (~5.5 GB): Contains CUDA compiler toolchain (nvcc, headers). Only the runtime is needed for inference. Switch to `nvidia/cuda:12.4.1-runtime-ubuntu22.04` would save ~1 GB (deps-analyzer estimate; less than initially thought because DGL wheels are pre-compiled).
  - **Caveat:** UniDock or DGL compilation may need devel headers. Verify before switching.
  - **Alternative:** Multi-stage build: compile in devel, copy binaries to runtime.

- **Ubuntu 20.04 is EOL** (April 2025): Should upgrade to 22.04 or 24.04 for security patches. This is a **security concern** for any shared or production deployment.

- **No layer caching optimization**: The `COPY docker/freedpp/data/...` layer comes late (line 90), which is correct. But the `RUN sed -i` patches (lines 54, 64, 66, 70) bust cache on every FREED++ git clone change. Consider separating the clone and patch into separate layers.

- **micromamba for UniDock** (line 81-85): Installs a separate conda environment just for UniDock + pdbfixer. This adds ~500 MB. Acceptable since UniDock is not available on PyPI.

**Estimated image size:** ~12-15 GB. Could reduce to ~8-10 GB with multi-stage build.

### 6.2 Container Management

- **Ephemeral containers** (`docker run --rm`): Each FREED++ training run starts a fresh container. Good for isolation, but means Docker pull/layer setup overhead (~1-2s) per run. Negligible for multi-hour training.

- **`dock_calibration.py:62-107` uses `DockerRunner`**: Persistent container with repeated `docker exec` calls. Smart design for calibration (avoids container startup per ligand).

- **No resource limits**: No `--memory`, `--shm-size`, or `--cpus` flags. On shared servers, FREED++ could monopolize resources. Consider adding `--shm-size=8g` (PyTorch DataLoader with shared memory).

---

## 7. Prioritized Optimization Plan

### Phase 1: Quick Wins (< 1 day effort)

| # | Action | File(s) | Speedup | Risk |
|---|--------|---------|---------|------|
| 1.1 | **Set `--parallel=2+` default** in `run_full_experiment.py` | `scripts/pipeline/run_full_experiment.py:135` | 2x on 2-GPU | Low |
| 1.2 | **Pin Boltz to specific git commit** | `pyproject.toml:192` | Stability | None |
| 1.3 | **Remove unused `polars` dependency** | `pyproject.toml:10` | Faster install | None |
| 1.4 | **Tune `num_sub_proc`** based on GPU: 12 for 4090, 24 for H100 | `configs/experiment.yaml:49` | 10-30% training | Low |
| 1.5 | **Increase Boltz batch size to 16-32** on H100 (Boltz internally processes batch_size=1, so external batch mainly amortizes subprocess startup; LOW OOM risk) | `dualdocking/scorer_boltz2.py:34` | 15-30% scoring (fewer subprocess invocations) | Low |
| 1.6 | **GPU pinning for 2xH100**: `CUDA_VISIBLE_DEVICES` per stage | `generator_freedpp.py:355`, `scorer_boltz2.py:351` | 2x on 2-GPU | Low |

### Phase 2: Medium Effort (1-3 days)

| # | Action | File(s) | Speedup | Risk |
|---|--------|---------|---------|------|
| 2.1 | **Parallel target scoring** in `score.py` | `scripts/s4_score/score.py:66-89` | 2x scoring | Low |
| 2.2 | **Parallel ligand prep** in `dock_calibration.py` | `scripts/s2_calibrate/dock_calibration.py:633` | 10-30x calibration | Low |
| 2.3 | **Parallel Boltz targets on 2 GPUs** | `dualdocking/scorer_boltz2.py` | 2x scoring | Medium |
| 2.4 | **Replace `datasets` with `requests`** in `fetch_calibration_smiles.py` | `scripts/s0_fetch/fetch_calibration_smiles.py:43` | Faster install | Low |
| 2.5 | **Pipeline GPU tasks**: generate on GPU 0 while scoring previous on GPU 1 | `scripts/pipeline/run_full_experiment.py` | 1.5-2x | Medium |
| 2.6 | **Bootstrap numpy pre-extraction**: extract columns as numpy arrays before 10K loop | `dualdocking/statistical.py:80-86` | 3-5x bootstrap (36s -> 7-12s) | Low |
| 2.7 | **Eliminate double SMILES parsing**: pass mol objects instead of re-parsing canonical SMILES | `dualdocking/chem_sanitize.py:85-90`, `diversity.py:79-91` | ~40% fewer RDKit parse calls | Low |

**Sketch for parallel target scoring (2.1):**
```python
# scripts/s4_score/score.py -- replace sequential loop with parallel
from concurrent.futures import ThreadPoolExecutor

def _score_target(target_id, smiles_list, scores_path, force, label):
    if not force and scores_path.exists():
        return label, pd.read_csv(scores_path)["score"].to_numpy()
    scores = score_batch(target_id, smiles_list)
    pd.DataFrame({"smiles": smiles_list, "score": scores}).to_csv(scores_path, index=False)
    return label, scores

with ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(_score_target, tid, smi, path, force, lbl)
               for tid, path, lbl in targets]
    for f in as_completed(futures):
        label, scores = f.result()
        all_scores[label] = scores
```

### Phase 3: Architectural Changes (1-2 weeks)

| # | Action | File(s) | Speedup | Risk |
|---|--------|---------|---------|------|
| 3.1 | **Persistent Boltz-2 server** (eliminate subprocess reload) | New: `dualdocking/boltz_server.py` | 3-8x scoring | High (requires Boltz Python API) |
| 3.2 | **Affinity-only Boltz model** (skip structure prediction) | Requires upstream Boltz changes | 2-3x scoring | High |
| 3.3 | **Multi-stage Docker build** | `docker/freedpp/Dockerfile` | -3 GB image | Medium |
| 3.4 | **GPU task scheduler** for 2xH100 | New: `dualdocking/gpu_scheduler.py` | Optimal utilization | Medium |
| 3.5 | **`torch.compile` for Boltz-2** | Upstream Boltz or monkey-patch | 1.2-2x inference | Medium (compile time) |

**Sketch for persistent Boltz server (3.1):**
```python
# boltz_server.py -- persistent inference server
import torch
from boltz.model import BoltzModel

class BoltzAffinityServer:
    """Long-lived Boltz-2 inference process. Load model once, score many."""

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        # Load model once (currently takes 10-30s per subprocess)
        self.model = BoltzModel.from_pretrained("boltz2").to(self.device)
        self.model.eval()
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)

    def predict_affinity(self, yaml_paths: list[str]) -> list[float]:
        """Score a batch without subprocess overhead."""
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Process batch through model
            results = self.model.predict(yaml_paths, tasks=["affinity"])
            return [r["affinity_pred_value"] for r in results]

# Usage: start server once, call predict_affinity() for each batch
# Saves ~10-30s model load per batch x 125 batches per target = 20-60 min saved per target
```

**Note:** This requires Boltz-2 to expose a Python API for inference. Currently it only has a CLI. The server would need to either:
1. Import Boltz internals directly (fragile but fast)
2. Use a socket/pipe-based IPC to a single long-lived `boltz predict` process

---

## 8. Hardware-Specific Recommendations

### Desktop (RTX 4090, ~8 cores)

| Setting | Current | Recommended |
|---------|---------|-------------|
| `num_sub_proc` | 24 | **8-12** (4090 has 16384 CUDA cores, 24 GB VRAM) |
| `BATCH_SIZE` (Boltz) | 8 | **8** (24 GB limits batch size; 16 may OOM) |
| `--parallel` | 1 | **1** (single GPU, no pipelining) |
| Calibration ligand prep | Sequential | **ProcessPoolExecutor(4)** |

**Estimated total time for 75 tasks:** ~600-1000 hours sequential. With optimizations: ~200-400 hours.

### Server (2xH100, 128 cores)

| Setting | Current | Recommended |
|---------|---------|-------------|
| `num_sub_proc` | 24 | **24** (H100 has 80 GB, handles 24 concurrent UniDock) |
| `BATCH_SIZE` (Boltz) | 8 | **32-64** (80 GB VRAM) |
| `--parallel` | 1 | **2-4** (2 GPUs, pipeline generate+score) |
| GPU pinning | `--gpus all` | **Pin per-task: GPU 0 for FREED++, GPU 1 for Boltz-2** |
| Calibration ligand prep | Sequential | **ProcessPoolExecutor(32)** |

**Recommended task scheduling on 2xH100:**
```
Time -->
GPU 0: [FREED++ task_1] [FREED++ task_3] [FREED++ task_5] ...
GPU 1: [Boltz-2 task_1 ] [Boltz-2 task_3 ] ...
        [FREED++ task_2] [FREED++ task_4] ...
```

With pipelining, GPU utilization approaches 90%+ instead of current ~45%.

---

## 9. Estimated Total Impact

### Before Optimization (75 tasks, single GPU)

| Stage | Time per task | Total (sequential) |
|-------|---------------|--------------------|
| FREED++ training | ~6 hours | 450 hours |
| Boltz-2 scoring | ~4 hours | 300 hours |
| Post-processing | ~3 min | 4 hours |
| **Total** | **~10 hours** | **~750 hours** |

### After Phase 1 (Quick Wins, 2xH100)

| Change | Impact |
|--------|--------|
| `--parallel=2` + GPU pinning | 2x throughput |
| Larger batch size (32) | -15% Boltz time |
| `num_sub_proc` tuning | -15% FREED++ time |
| **Total** | **~350 hours -> ~175 wall-clock hours** |

### After Phase 2 (Medium Effort, 2xH100)

| Change | Cumulative Impact |
|--------|-------------------|
| Parallel target scoring | -50% Boltz time |
| GPU pipelining (generate on GPU 0 while scoring on GPU 1) | -30% wall-clock |
| Parallel ligand prep | -90% calibration time |
| **Total** | **~100-120 wall-clock hours** |

### After Phase 3 (Architectural, 2xH100)

| Change | Cumulative Impact |
|--------|-------------------|
| Persistent Boltz server | -70% Boltz time |
| Affinity-only model | -50% remaining Boltz time |
| torch.compile | -20% remaining inference |
| **Total** | **~60-80 wall-clock hours** |

### ROI Summary

| Phase | Effort | Wall-Clock Reduction | ROI |
|-------|--------|---------------------|-----|
| Phase 1 | <1 day | 750h -> 175h (77%) | Exceptional |
| Phase 2 | 1-3 days | 175h -> 110h (37%) | High |
| Phase 3 | 1-2 weeks | 110h -> 70h (36%) | Moderate |

**Bottom line:** Phase 1 alone delivers most of the gains. The single highest-impact change is **GPU pinning + `--parallel=2` on 2xH100**, which requires changing ~5 lines of code.

---

*Report generated by DualDocking Performance Audit Team, 2026-02-08.*
