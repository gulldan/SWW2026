# Documentation Guide

Navigation hub for the DualDocking project documentation.

Updated: 2026-02-11

---

## Reading Order by Persona

### "I'm new to this project"

1. [`goal.md`](goal.md) -- What we're testing and why (hypothesis, notation, endpoints)
2. [`setup.md`](setup.md) -- Install Python, Docker, download data
3. [`pipeline.md`](pipeline.md) -- Run the experiment end-to-end
4. [`architecture.md`](architecture.md) -- How scoring, generation, and caching work

### "I'm reviewing the science"

1. [`goal.md`](goal.md) -- Hypothesis, notation, success criteria
2. [`preregistration.md`](preregistration.md) -- Frozen hypotheses, endpoints, decision rules
3. [`statistical_analysis_plan.md`](statistical_analysis_plan.md) -- Frozen inference procedures, claim rules, code mapping
4. [`selection_criteria.md`](selection_criteria.md) -- Why these target pairs, factor design, cofactor caveats

### "I'm running the experiment"

1. [`setup.md`](setup.md) -- Environment, Docker image, calibration
2. [`pipeline.md`](pipeline.md) -- Stage 0 fetch through analysis
3. [`architecture.md`](architecture.md) -- Troubleshooting, backend architecture, score conventions
4. [`receptor_qc.md`](receptor_qc.md) -- Per-target structural quality checks
5. [`nim_benchmark.md`](nim_benchmark.md) -- CLI vs NIM performance comparison

### "I'm developing code"

1. [`architecture.md`](architecture.md) -- Technical internals, data flow, caching
2. [`AGENTS.md`](../AGENTS.md) -- Development conventions, code style, testing
3. [`CONTRIBUTING.md`](../CONTRIBUTING.md) -- Developer onboarding (if available)

---

## Document Map

| # | Document | Status | Summary |
|---|----------|--------|---------|
| 1 | [`goal.md`](goal.md) | Operational spec | Supreme authority. Hypothesis, notation, endpoints, success criteria. Read-only without user approval. |
| 2 | [`preregistration.md`](preregistration.md) | Frozen (v1.1) | Pre-registered hypotheses, study design, decision rules. Amendments tracked in Section 11. |
| 3 | [`statistical_analysis_plan.md`](statistical_analysis_plan.md) | Frozen (v1.1) | Inference procedures: bootstrap, effect sizes, claim thresholds, code mapping. |
| 4 | [`selection_criteria.md`](selection_criteria.md) | Reference | Pair selection rationale: factor design (Disease/Structure/Cofactor), core vs auxiliary split, Boltz-2 compatibility. |
| 5 | [`setup.md`](setup.md) | Canonical | Installation: Python/uv, Docker/FREED++, scorer setup, calibration, baseline. |
| 6 | [`pipeline.md`](pipeline.md) | Canonical | Full pipeline: fetch (Stage 0) through run, scoring, and analysis. |
| 7 | [`architecture.md`](architecture.md) | Reference | Technical internals: pluggable generator/scorer backends, Boltz-2 CLI/NIM, FREED++ Docker, caching, troubleshooting. |
| 8 | [`receptor_qc.md`](receptor_qc.md) | Reference | Per-target structural QC: resolution, chain, ligand, mutations, coverage. |
| 9 | [`nim_benchmark.md`](nim_benchmark.md) | Reference | Boltz-2 CLI vs NIM backend: latency, score agreement, TensorRT notes. |
| 10 | [`tasks.md`](tasks.md) | Living log | Task journal, execution plan, daily progress. |

**Status definitions:**
- **Operational spec** -- single source of truth; changes require explicit approval
- **Frozen** -- locked at a specific commit; amendments tracked in-document
- **Canonical** -- authoritative for its domain; updated as needed
- **Reference** -- supporting detail; consult when needed
- **Living log** -- updated continuously

---

## Key Concepts Quick Reference

### Notation

| Symbol | Source | Units | Role |
|--------|--------|-------|------|
| `d_A`, `d_B` | UniDock GPU (docking) | kcal/mol, lower-is-better | **RL reward** -- drives molecule generation |
| `a_A`, `a_B` | Boltz-2 (learned affinity) | log10(IC50-like, uM), lower-is-better | **Evaluation surrogate** -- orthogonal post-hoc assessment |

The two scorers are never mixed: FREED++ optimizes docking scores `d_A, d_B`; Boltz-2 evaluates generated molecules independently.

### Convention

`lower_is_better=True` everywhere. Utility = -score. More negative = better binding.

### Target Pairs

- **5 core pairs** (primary analysis): CDK2\_ERK2, AChE\_BChE, BACE1\_GSK3B, AChE\_BACE1, GSK3B\_CDK5
- **3 auxiliary pairs** (exploratory): EGFR\_HER2, BACE1\_CDK5, AChE\_MAOB
- **1 blocked pair**: COX1\_COX2 (non-human PDB structures)

Pair definitions live in `configs/pairs.yaml`. Each has a `set: core/auxiliary/blocked` field.

### Generation Modes

| Mode | Reward | Purpose |
|------|--------|---------|
| `single_a` | `f(d_A)` | Baseline: optimize for target A only |
| `single_b` | `f(d_B)` | Baseline: optimize for target B only |
| `double` | `f(combined(d_A, d_B))` | Experimental: joint dual-target optimization |

### Success Criterion

`double` mode must show higher Boltz-2 dual hit-rate than both single-target baselines on at least 3 of 5 core pairs, with Cliff's delta > -0.47 (not worse than medium effect).

---

## Source Priority

Higher rank wins when documents conflict. Each level serves a distinct role.

| Rank | Source | Why it has priority |
|------|--------|---------------------|
| 1 | `docs/goal.md` | Defines the hypothesis, endpoints, and success criteria that everything else implements. Changes here cascade to all downstream documents. |
| 2 | `configs/*.yaml` + `dualdocking/` | Runtime truth -- what actually executes. Code and config are the ground truth for parameters and logic. |
| 3 | `docs/preregistration.md`, `docs/statistical_analysis_plan.md` | Frozen analysis contracts that lock down inference before data collection. Amendments are versioned in-document. |
| 4 | `docs/selection_criteria.md` | Pair rationale -- justifies experimental choices but does not define endpoints or analysis. |
| 5 | `report/chapters/*.tex` | Narrative reporting -- must faithfully reflect the sources above, not introduce new claims. |
| 6 | `docs/README.md` (this file) | Navigation guide -- may simplify for readability but must not misrepresent. |

## Conflict Resolution

If a command, parameter, or threshold in a document conflicts with code or configuration:

1. Check `configs/*.yaml` and `dualdocking/` for the runtime value.
2. Check `docs/goal.md` for the intended value.
3. If they disagree, `goal.md` intent wins and the code should be fixed.
4. If `goal.md` is silent, code is authoritative.

---

## Cross-Reference: Where to Find Specific Information

| Topic | Location |
|-------|----------|
| Generator/scorer backend selection | `configs/experiment.yaml` (`generator.backend`, `scorer.engine`) |
| Binding box coordinates, grid sizes | `architecture.md` (Binding Box section) + `configs/pairs.yaml` |
| Affinity thresholds (T_A, T_B) | `configs/metrics.yaml` (fixed at 0.0 = 1 uM) |
| Boltz-2 cache format, context hashing | `architecture.md` (Caching section) |
| Boltz-2 NIM setup and Docker launch | `architecture.md` + `nim_benchmark.md` |
| Bootstrap procedure, claim rules | `statistical_analysis_plan.md` (Sections 3-5) |
| ChEMBL target IDs | `configs/pairs.yaml` (per-target `chembl_id` field) |
| Cofactor handling and caveats | `selection_criteria.md` (Cofactor section) |
| Docker image build | `setup.md` (Docker section) + `pipeline.md` (Stage 0) |
| Factor design (Disease/Structure/Cofactor) | `selection_criteria.md` (Factor Design section) |
| FREED++ dual-target patch | `architecture.md` (Generation section) |
| Per-target PDB, resolution, mutations | `receptor_qc.md` (Summary Table) |
| Score validation and corruption handling | `architecture.md` (UniDock GPU section) |
| Sign convention, utility conversion | `architecture.md` (Score Conventions) + `goal.md` (Notation) |

---

See `AGENTS.md` for development conventions (architecture, code style, testing).
