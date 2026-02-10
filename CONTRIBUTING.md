# Contributing to DualDocking

Practical guide for setting up the development environment, running tests, adding features, and debugging common issues.

## Development Setup

### Prerequisites

- Python >= 3.11 (tested on 3.12)
- [uv](https://docs.astral.sh/uv/) >= 0.9
- Docker >= 20.x with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- NVIDIA GPU with >= 16 GB VRAM (RTX 4090 recommended)

### Install

```bash
git clone <repo-url> && cd SWW2026

# Python environment (includes Boltz-2 from source)
uv sync --group dev

# Pre-commit hooks (ruff format + ruff check)
pre-commit install

# FREED++ Docker image
uv run python scripts/s0_fetch/fetch_freed_fragments.py
docker build -f docker/freedpp/Dockerfile -t dualdocking/freedpp:latest .
```

### Verify

```bash
uv run pytest tests/ -v
uv run ruff check dualdocking/ scripts/ docker/
uv run ty check dualdocking/
```

## Testing

### Commands

| What | Command |
|------|---------|
| Full suite | `uv run pytest tests/ -v` |
| Single file | `uv run pytest tests/scoring/test_boltz2.py -v` |
| Integration only | `uv run pytest -m integration` |
| Benchmarks (GPU) | `uv run pytest -m benchmark` |
| Coverage | `uv run pytest --cov=dualdocking tests/` |
| Lint | `uv run ruff check dualdocking/ scripts/ docker/` |
| Type check | `uv run ty check dualdocking/` |
| Auto-format | `uv run ruff format dualdocking/ scripts/` |

Tests run in parallel by default (`-n auto` in `pyproject.toml`). Override with `-n0` for serial execution during debugging.

### Testing Patterns

- **monkeypatch** is used extensively to override module-level paths (`CACHE_DIR`, `DATA_DIR`) and reset cached state like `_CONTEXT_HASH`.
- **FakeDockingVina**: a real class with `_scores: list[float]` for integration tests through `_dual_call`. Use it to test reward logic without GPU docking.
- **dual_reward_patch**: auto-applies on import. Tests must mock `ffreed` BEFORE import and clean `sys.modules` AFTER.
- **`load_pairs()` schema validation**: test fixtures must provide the full structure (`target_a.id`, `target_b.id`, etc.), not shorthand.

## Code Style

### Ruff

- Rule selection: `ALL` with curated ignores (see `pyproject.toml`)
- Line length: 130
- Docstrings: Google convention
- Quotes: double
- Format: `ruff format` (Black-compatible)

### Type Checking

- `ty` in strict mode on `dualdocking/` only
- `tests/` and `scripts/` are excluded from type checking
- External library stubs are `warn` level (see `[tool.ty.rules]` in `pyproject.toml`)

### Conventions

**lower_is_better = True everywhere.** Docking scores are negative (more negative = better binding). Utility = -score.

Naming:

| Symbol | Meaning |
|--------|---------|
| `d_A`, `d_B` | UniDock docking scores (kcal/mol) -- RL reward signal |
| `a_A`, `a_B` | Boltz-2 affinity predictions (log10 IC50-like uM) -- evaluation surrogate |
| `T_A`, `T_B` | Hit thresholds (log10 IC50 uM); 0.0 = 1 uM |
| `z_A`, `z_B` | Per-target z-scores from calibration |

Target pairs are named `{TARGET_A}_{TARGET_B}`, e.g., `CDK2_ERK2`. Pairs belong to a `set`: `core` (5 pairs, primary analysis), `auxiliary` (3 pairs, robustness), or `blocked` (excluded).

### Files You Should Never Commit

`output/`, `.env`, credential files -- all gitignored. Double-check `git status` before pushing.

## How to Add a New Target Pair

1. **Add entry to `configs/pairs.yaml`.**
   Follow the existing schema. Required fields: `description`, `boltz2_status`, `set`, and for each target: `id`, `name`, `uniprot`, `pdb_id`, `organism`, `binding_site`. Assign `set: auxiliary` until validated.

2. **Prepare receptor structures.**
   Download PDB files into `output/data/targets/{pair}/{A,B}/`. Convert to PDBQT:
   ```bash
   uv run python scripts/s1_prepare/prepare_targets.py --pair YOUR_PAIR
   ```
   This downloads the PDB, removes heterogens, adds hydrogens, and produces the PDBQT. It also downloads the full UniProt FASTA -- if your target uses a domain construct (not full-length), you must manually truncate the FASTA to the relevant residues.

3. **Generate MSA for Boltz-2.**
   ```bash
   uv run python scripts/s1_prepare/prepare_targets.py --pair YOUR_PAIR --prepare-all
   ```
   MSA output goes to `output/cache/boltz_msa/{target_id}.csv`. If you change the FASTA sequence, regenerate the MSA.

4. **Run calibration docking.**
   ```bash
   uv run python scripts/s2_calibrate/dock_calibration.py --pair YOUR_PAIR
   uv run python scripts/s2_calibrate/compute_normalization.py
   ```
   This docks a reference SMILES set (MOSES) against both targets and computes z-score normalization parameters (mean/std) for the reward function.

5. **Validate redocking.**
   ```bash
   uv run python scripts/s2_calibrate/validate_redocking.py --pair YOUR_PAIR
   ```
   Confirms that the binding box and receptor preparation are reasonable.

6. **Add thresholds to `configs/metrics.yaml`** under `thresholds:`. Default: `target_a: 0.0` / `target_b: 0.0` (1 uM).

7. **Update documentation.** If the pair is `core`, update `docs/selection_criteria.md` and `report/chapters/04_experimental_protocol.tex`.

## How to Add a New Objective Function

Objective functions live in `dualdocking/scoring/objectives.py`. They combine two per-target scores into a single scalar.

1. **Register the kind.** Add an entry to the `ObjectiveKind` enum:
   ```python
   class ObjectiveKind(StrEnum):
       MIN = "min"
       HARMONIC = "harmonic"
       # ...
       YOUR_NEW = "your_new"
   ```

2. **Implement the function.** Follow the existing pattern:
   ```python
   def combined_your_new(
       scores_a: NDArray[np.float64],
       scores_b: NDArray[np.float64],
       *,
       lower_is_better: bool = True,
       # any extra params
   ) -> NDArray[np.float64]:
       """One-line description.

       Longer explanation if needed.
       """
       # Convert to utility if your function needs "higher is better"
       util_a = _to_utility(scores_a, lower_is_better)
       util_b = _to_utility(scores_b, lower_is_better)
       # ... compute combined utility ...
       return _from_utility(combined_utility, lower_is_better)
   ```
   - If your function works directly in score-space (like Tchebycheff), skip the utility conversion.
   - All functions must accept `NDArray[np.float64]` and return the same shape.

3. **Wire it into the dispatch.** Add a case in `compute_combined()` that maps the new `ObjectiveKind` to your function.

4. **Add to `configs/experiment.yaml`** under `objectives.alternatives` if it should be available for sensitivity analysis.

5. **Write tests.** Add a test class in `tests/scoring/test_objectives.py` covering: basic correctness, edge cases (NaN, identical scores, extreme values), and the `lower_is_better` convention.

## Troubleshooting

### GPU OOM with Boltz-2

**Symptoms**: NaN affinity scores, `CUDA out of memory` in logs.

**Causes**: Ring-heavy molecules (ring_atom_fraction > 0.85) exceed VRAM during structure prediction. VRAM competition with FREED++ when running `--parallel 2`.

**Fixes**:
- Use `--parallel 1` on single-GPU setups (eliminates VRAM competition).
- Switch to the NIM backend (`BOLTZ_BACKEND=nim`) which manages memory internally.
- For domain constructs (EGFR, HER2), ensure FASTA is truncated to the kinase domain, not full-length UniProt.

### "Charged species not supported"

**Cause**: Boltz-2 v2.2.1 does not support charged molecules. One charged SMILES crashes the entire batch subprocess, losing up to 63 neighboring molecules.

**Fix**: Already handled -- `_neutralize_smiles()` using `rdMolStandardize.Uncharger()` is applied before Boltz-2 scoring. If you see this error, check that your input SMILES are being routed through the scoring pipeline (not called directly).

### Docker Image Build Fails

**Cause**: Missing data files that get copied into the image.

**Fix**: Run `uv run python scripts/s0_fetch/fetch_freed_fragments.py` before building. Then:
```bash
docker build -f docker/freedpp/Dockerfile -t dualdocking/freedpp:latest .
```

### Score Corruption in UniDock

**Symptoms**: Unrealistic docking scores (e.g., > 0 kcal/mol or < -20 kcal/mol).

**Cause**: GPU batch docking has known issues (uninitialized BFGS, NaN quaternions). Affects 7-11% of FREED++-generated molecules.

**Protection**: Score validation rejects values outside [-20, 10] kcal/mol. Rejected molecules fall back to CPU docking. Z-scores are clamped to +/-10 to prevent gradient domination. No action needed unless the fallback rate exceeds `quality_gate.max_invalid_fraction` (10%).

### Cache Invalidation

**When does the cache reset?** The Boltz-2 cache key includes a context hash computed from: Boltz version, CLI arguments, GPU accelerator type, and Docker digest. Changing any of these produces a new cache file, so old cached scores are never mixed with new settings.

**Force re-score**: Delete the relevant cache file in `output/cache/{target}.{ctx[:8]}.json`, or change a Boltz-2 parameter.

**NaN scores are never cached.** Failed molecules retry on every run automatically.

### NIM vs CLI Backend

| | CLI (`BOLTZ_BACKEND=cli`) | NIM (`BOLTZ_BACKEND=nim`) |
|--|--------------------------|--------------------------|
| How it works | `boltz predict` subprocess per batch | HTTP POST to persistent GPU server |
| Speed | Varies (model load per batch) | ~5-6 sec/molecule on RTX 4090 |
| Setup | Just `uv sync` | Docker container + NGC API key |
| MSA | Required (pre-computed) | Not needed (server handles it) |
| VRAM | Loads/unloads per batch | ~7.7 GB persistent |
| Cache | Separate context hash | Separate context hash |

Start the NIM container:
```bash
docker run -d --gpus all --shm-size=16G \
  -e NGC_API_KEY=<your-key> \
  -e NIM_BOLTZ_STRUCTURE_OPTIMIZED_BACKEND=torch \
  -e NIM_BOLTZ_AFFINITY_OPTIMIZED_BACKEND=torch \
  -v /path/to/cache:/opt/nim/.cache \
  -p 8000:8000 \
  nvcr.io/nim/mit/boltz2:1.5.0
```

Set environment variables: `BOLTZ_BACKEND=nim` and `BOLTZ_NIM_URL=http://localhost:8000`.

## Workflow Decision Tree

| I want to... | Command |
|--------------|---------|
| Set up from scratch | `uv sync --group dev && pre-commit install` |
| Run all tests | `uv run pytest tests/ -v` |
| Run tests for one module | `uv run pytest tests/scoring/test_boltz2.py -v` |
| Check code style | `uv run ruff check dualdocking/ scripts/ docker/` |
| Auto-fix lint issues | `uv run ruff check --fix dualdocking/ scripts/ docker/` |
| Format code | `uv run ruff format dualdocking/ scripts/` |
| Type check | `uv run ty check dualdocking/` |
| Build FREED++ Docker image | `docker build -f docker/freedpp/Dockerfile -t dualdocking/freedpp:latest .` |
| Prepare a new target pair | `uv run python scripts/s1_prepare/prepare_targets.py --pair PAIR_NAME --prepare-all` |
| Run calibration | `uv run python scripts/s2_calibrate/dock_calibration.py && uv run python scripts/s2_calibrate/compute_normalization.py` |
| Validate redocking | `uv run python scripts/s2_calibrate/validate_redocking.py --pair PAIR_NAME` |
| Run a smoke test | `uv run python scripts/smoke/boltz_smoke.py` |
| Run the full experiment | `uv run python scripts/pipeline/run_full_experiment.py` |
| Score molecules with Boltz-2 | `uv run dd-score --pair CDK2_ERK2 --input generated.csv` |
| Compute metrics | `uv run dd-metrics` |
| Generate plots | `uv run dd-plot` |
| Generate summary report | `uv run dd-summarize` |

## Performance Expectations

These numbers are for an RTX 4090 (24 GB VRAM). Expect roughly 2x slower on 16 GB GPUs.

| Operation | Time |
|-----------|------|
| FREED++ epoch (single target) | ~7-9 min |
| FREED++ epoch (double target) | ~12-16 min |
| Boltz-2 per molecule (NIM, torch backend) | ~5-6 sec |
| Boltz-2 per molecule (CLI) | Varies; model load overhead per batch |
| Full experiment (5 core pairs x 3 modes x 5 seeds x 100 epochs) | ~50 GPU-days on 1 RTX 4090 |

Tips for faster iteration during development:
- Use `epochs: 10` in `configs/experiment.yaml` for quick pipeline tests.
- Use `--parallel 1` on single-GPU setups to avoid VRAM competition.
- The NIM backend avoids repeated model loading -- preferred for scoring-heavy workflows.
- Boltz-2 `diffusion_samples=1` + `sampling_steps=50` gives ~2-3x speedup at the cost of prediction quality (acceptable for smoke tests, not for production).

## Project Source Priority

When documents or code disagree, higher-numbered sources override lower ones:

1. `docs/goal.md` -- supreme authority (read-only without user approval)
2. `configs/*.yaml` -- runtime parameters (must agree with goal.md)
3. `dualdocking/` -- computation logic
4. `docs/selection_criteria.md` -- pair selection rationale
5. `report/chapters/*.tex` -- narrative
6. `README.md` -- operational guide

If you change a parameter in one place, check the synchronization table in `AGENTS.md` to update all related files.
