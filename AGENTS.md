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
| `dualdocking/**/*.py`   | Corresponding `tests/test_*.py`, `docs/goal.md` if behavior changes            |
| `docs/goal.md`          | `report/chapters/` (methods, protocol, limitations)                              |
| `report/references.bib` | All `\cite{}` in `report/chapters/*.tex`                                         |
| Install/Docker/pipeline | `docs/setup.md`, `docs/pipeline.md`                                             |
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

# Run full test suite (396 tests, ~7 min)
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_objectives.py -v

# Linting (must pass with zero warnings)
uv run ruff check dualdocking/ cli/ docker/

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
  config.py                   #   YAML config loading & validation
  scoring/                    #   Scoring & docking subpackage
    objectives.py             #     Multi-objective functions (min, harmonic, Pareto, etc.)
    boltz2.py                 #     Boltz-2 affinity scorer with disk cache
    cache_io.py               #     Atomic disk I/O (fcntl + temp-and-replace)
    binding_box.py            #     PDB binding site extraction
  generation/                 #   Molecule generation subpackage
    freedpp.py                #     FREED++ Docker subprocess wrapper
  analysis/                   #   Statistical analysis subpackage
    statistical.py            #     Bootstrap, FDR, Cliff's delta
    fairness.py               #     Fairness metrics
    enrichment.py             #     Enrichment analysis
    diversity.py              #     Chemical diversity metrics
  data/                       #   Data handling subpackage
    chem_sanitize.py          #     SMILES validation & dedup
    baseline.py               #     Baseline comparison
    chembl_validation.py      #     ChEMBL target validation
  reporting/                  #   Reporting subpackage
    run_manifest.py           #     Lifecycle tracking for experiment runs
    latex_report.py           #     LaTeX report generation
cli/                      # CLI entry points (organized by pipeline stage)
  s0_fetch/                   #   Data download (PDB, FASTA, FREED fragments, ZINC, ChEMBL)
  s1_prepare/                 #   Target preparation (PDB -> PDBQT, binding box, MSA)
  s2_calibrate/               #   Calibration and validation
  s3_generate/                #   Molecule generation (FREED++)
  s4_score/                   #   Affinity scoring (Boltz-2)
  s5_analyze/                 #   Statistical analysis (aggregate, metrics, correlation)
  s6_report/                  #   Visualization and manuscript (plots, molecules, LaTeX)
  s7_audit/                   #   Post-hoc verification (claim traceability)
  pipeline/                   #   End-to-end orchestrators (full experiment, pilot, post)
  smoke/                      #   Smoke tests (Boltz-2 health check, reward capping)
tests/                        # pytest suite, monkeypatch-heavy
configs/                      # YAML: pairs.yaml, metrics.yaml, experiment.yaml
docker/freedpp/               # Dockerfile + dual_reward_patch.py monkey-patch
docs/                         # goal.md (spec), selection_criteria.md, tasks.md
report/                       # LaTeX manuscript (chapters/, figures/, templates/)
output/                       # Runtime data (gitignored)
  data/                       #   Target structures, calibration, baseline
  cache/                      #   Boltz-2 score cache, keyed by context hash
  results/                    #   Experiment outputs
```

### Data Flow

Stage 0 (fetch) → Generation (FREED++, 3 modes × 5 pairs × 5 seeds) → Scoring (Boltz-2) → Aggregation → Metrics → Report

## Code Style & Conventions

- **Python 3.12**, package manager `uv`, build system `hatchling`
- **Ruff**: `select = ["ALL"]` with curated ignores, line-length 130, Google docstrings, double quotes
- **Type checker**: `ty` in strict mode (`dualdocking/` only; tests and cli excluded)
- **lower_is_better = True everywhere** — docking scores are negative (more negative = better binding). Utility = −score.
- All objective functions convert scores to utilities before aggregation
- Tchebycheff is the exception: works directly in score-space (abs is symmetric)

### Naming

- `d_A`, `d_B` — UniDock docking scores (kcal/mol, RL reward signal)
- `a_A`, `a_B` — Boltz-2 affinity predictions (log10 IC50-like μM, evaluation surrogate)
- Target pairs: `CDK2_ERK2`, `AChE_BChE`, etc. — see `configs/pairs.yaml`
- Sets: `core` (5 pairs, primary analysis), `auxiliary` (3), `blocked` (1, COX1_COX2)

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
- `load_pairs()` validates schema (requires `target_a.id`, `target_b.id`) — test fixtures must provide full structure
- Harmonic/geometric means use shift for positive-domain requirement

## Gotchas

- `_build_freedpp_args` signature: `(config, pdbqt_a, center_a, size_a, commands)` — no workspace param
- `INVALID_Z_PENALTY` is validated `> 0 && finite` at module load; negative values would invert reward
- `REWARD_METADATA_PATH` needs `mkdir(parents=True)` — custom paths may have non-existent parents
- `pareto_mask` uses O(n log n) lexsort + sweep; handles ties with `−sb` tiebreak
- Boltz-2 output is `log10(IC50 in μM)`, NOT pIC50

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
