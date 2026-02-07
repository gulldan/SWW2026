# Pre-registration: Double-Target RL Generation vs Single-Target Baselines

| Field | Value |
|-------|-------|
| **Date** | 2026-02-07 |
| **Status** | FROZEN |
| **Freeze commit** | `f2eceb2` |

---

## 1. Research Question & Hypothesis

**Question:** Does RL molecule generation optimized for a joint dual-target objective (combined docking score over targets A and B) produce molecules with higher dual hit-rate than single-target optimization followed by cross-evaluation?

**Hypothesis (H1):** The `double` mode (`reward = -max(d_A, d_B)`) yields a higher fraction of dual hits (Boltz-2 affinity `a_A <= T_A AND a_B <= T_B`) than either `single_a` or `single_b` mode, when evaluated on the same molecule budget.

**Null (H0):** `hr_both(double) <= hr_both(single_x)` for at least 3 of 5 core pairs.

## 2. Study Design

### 2.1 Target Pairs

| # | Pair | Set | Structure | Cofactor | PDB A / PDB B |
|---|------|-----|-----------|----------|---------------|
| 1 | CDK2_ERK2 | **Core** | Homo | Clean | 1HCL / 4QTA |
| 2 | AChE_BChE | **Core** | Homo | Clean | 4EY7 / 4BDS |
| 3 | BACE1_GSK3B | **Core** | Hetero | Clean | 6EQM / 5KPK |
| 4 | AChE_BACE1 | **Core** | Hetero | Clean | 4EY7 / 6EQM |
| 5 | GSK3B_CDK5 | **Core** | Hetero | Clean | 5KPK / 1UNL |
| 6 | EGFR_HER2 | Auxiliary | Homo | Clean | 1M17 / 3PP0 |
| 7 | BACE1_CDK5 | Auxiliary | Hetero | Clean | 6EQM / 1UNL |
| 8 | AChE_MAOB | Auxiliary | Hetero | Present (FAD) | 4EY7 / 2V5Z |
| 9 | COX1_COX2 | Blocked | Homo | Present (heme) | 1EQG / 3LN1 |

Source: `configs/pairs.yaml`, `docs/selection_criteria.md`

### 2.2 Modes

| Mode | Reward function | Description |
|------|-----------------|-------------|
| `single_a` | `-d_A` | Optimize for target A only |
| `single_b` | `-d_B` | Optimize for target B only |
| `double` | `-max(d_A, d_B)` | Worst-case scalarization over both targets |

### 2.3 Budget & Seeds

| Parameter | Value | Source |
|-----------|-------|--------|
| Molecules per run | 1000 | `configs/experiment.yaml: budget.n_molecules` |
| Seeds | [42, 137, 256, 512, 1024] | `configs/experiment.yaml: seeds` |
| Runs per (pair, mode) | 5 | `configs/experiment.yaml: n_runs` |
| RL epochs | 200 | `configs/experiment.yaml: generator.freedpp.epochs` |
| Generator | FREED++ (UniDock GPU) | `configs/experiment.yaml: generator.name` |
| Scorer | Boltz-2 affinity | `configs/experiment.yaml: scorer.name` |

Total experimental matrix: 8 active pairs x 3 modes x 5 seeds = **120 runs**.

## 3. Primary Endpoint

**Metric:** Dual hit-rate difference `hr_both(double) - hr_both(single_x)` on 5 core pairs.

**Hit definition:** A molecule is a dual hit if `a_A <= T_A AND a_B <= T_B`, where:
- `a_A`, `a_B` = Boltz-2 `affinity_pred_value` (log10 IC50-like, uM; lower is better)
- `T_A = T_B = 0.0` for all pairs (= 1 uM)

Source: `configs/metrics.yaml: thresholds`, `configs/metrics.yaml: hit_definition`

**Comparison:** For each core pair, two contrasts:
1. `hr_both(double) - hr_both(single_a)`
2. `hr_both(double) - hr_both(single_b)`

The primary comparison uses the contrast yielding the smaller difference (conservative: double must beat the better single baseline).

## 4. Decision Rules

### 4.1 Success Criteria

All three conditions must hold simultaneously:

1. **Prevalence:** `hr_both(double) > hr_both(single_x)` for **>= 3 of 5** core pairs (>= 60%).
2. **Statistical significance:** For pairs satisfying (1) — BH FDR-adjusted `p < 0.05`, bootstrap 95% CI excludes 0.
3. **Practical significance:** For pairs satisfying (1) — `|Cliff's delta| >= 0.33` (medium effect).

Source: `docs/goal.md` lines 192-198

### 4.2 No-Degradation Guard

For every core pair: per-target hit-rates `hr_a(double)` and `hr_b(double)` must not show large negative effect vs the corresponding single mode.

**Threshold:** Cliff's delta > -0.47 (below the large-effect boundary).

If violated: the pair is flagged with "per-target penalty" and excluded from success count in (4.1.1).

### 4.3 No-Claim Rules

- **< 3/5 core pairs pass:** "No evidence that double-target RL improves dual hit-rate."
- **Degradation on any core pair:** "Double-target RL incurs per-target penalty on pair X."
- **Cliff's delta < 0.33 but > 0:** "Statistically detectable but practically negligible improvement."

## 5. Secondary Endpoints

| Metric | Definition | Source |
|--------|------------|--------|
| Enrichment@K | `(hits_in_top_K / K) / (total_hits / N)`, top-K by `max(a_A, a_B)` ascending; K = 10, 50, 100, 500 | `configs/metrics.yaml: enrichment.k_values` |
| Pareto hypervolume (HV) | Area dominated by non-dominated set in `(a_A, a_B)` space; reference point `r = (2.0, 2.0)` | `docs/goal.md` lines 125-128 |
| Per-target `hr_a`, `hr_b` | Single-target hit-rates for degradation monitoring | `configs/metrics.yaml: computed_metrics` |

Effect size: Cohen's d (secondary).

## 6. Exploratory Endpoints

| Metric | Purpose |
|--------|---------|
| Tanimoto overlap with ChEMBL dual-actives | Structural similarity to known dual-active chemotypes |
| Boltz-2 `affinity_probability_binary` | Binary hit/non-hit for cofactor-sensitive targets (AChE_MAOB) |
| Internal diversity (inter-mode) | Whether double mode collapses to narrow chemical space |
| Spearman rho (`d_X` vs `a_X`) per target | Sanity check: docking-Boltz-2 concordance |

## 7. Statistical Analysis Plan (Summary)

### 7.1 Hierarchical Bootstrap

- **Level 1 (seeds):** Resample with replacement from 5 RL runs.
- **Level 2 (molecules):** Within each resampled run, resample with replacement from `n_molecules` molecules.
- **Statistic:** `hr_both(double) - hr_both(single_x)` per bootstrap replicate.
- **Replicates:** 10,000.
- **RNG seed:** 42.
- **95% CI:** Percentile method (2.5th, 97.5th percentiles).
- **p-value:** Fraction of replicates where `|diff| >= |observed_diff|` (two-sided).

Implementation: `src/dualdocking/statistical.py: hierarchical_bootstrap_test()`

### 7.2 Multiple Testing Correction

- **Method:** Benjamini-Hochberg FDR.
- **Alpha:** 0.05.
- **Scope:** Across the 5 core pairs (up to 10 contrasts: 5 pairs x 2 single baselines).

Implementation: `src/dualdocking/statistical.py: fdr_correction()`

### 7.3 Effect Sizes

| Measure | Role | Thresholds | Implementation |
|---------|------|------------|----------------|
| Cliff's delta | Primary | < 0.147 negligible, < 0.33 small, < 0.474 medium, >= 0.474 large | `statistical.py: cliffs_delta()` |
| Cohen's d | Secondary | 0.2 / 0.5 / 0.8 (small / medium / large) | `statistical.py: cohens_d()` |

### 7.4 Missing Data Policy

- **NaN affinity:** Molecule treated as non-hit (conservative).
- **Completeness gate:** A run is invalid if > 10% invalid molecules or > 5% timeout molecules.
- Invalid runs are excluded from analysis entirely.

Source: `configs/experiment.yaml: quality_gate`

## 8. Sensitivity Analyses

| Analysis | Description |
|----------|-------------|
| **Threshold sensitivity** | Repeat primary analysis at T = -1.0 (100 nM), 0.0 (1 uM), +1.0 (10 uM) |
| **Equal docking calls** | Single modes with 400 RL epochs (2x training budget); same n_molecules = 1000 |
| **Alternative scalarizations** | Offline recomputation with geometric, harmonic, weighted_sum (w=0.5), weighted_product (w=0.5), Tchebycheff (w=0.5, utopia=-3.0) |

Source: `configs/experiment.yaml: objectives.alternatives`, `docs/goal.md` lines 43-63

## 9. Known Limitations

1. **m=5 seeds:** Bootstrap CI and p-values are coarse estimates with only 5 clusters. Primary evidence weight is on effect size and cross-pair consistency, not p-values alone (Cameron/Gelbach/Miller 2008).
2. **Target overlap in core set:** AChE appears in 2 core pairs, BACE1 in 2, GSK3B in 2. Core pairs are not fully independent observations.
3. **Boltz-2 not calibrated:** Predicted values are IC50-like surrogates (trained on mixed Ki/Kd/IC50/EC50), not calibrated IC50. All comparisons are relative (double vs single), not absolute activity predictions.
4. **In-domain bias:** PDB structures likely appear in Boltz-2 training data. Metrics are interpreted as relative ranking between strategies.
5. **Docking compute asymmetry:** Double mode uses ~2x docking calls per molecule. Primary comparison is fixed molecule count; equal-docking-calls is sensitivity only.

## 10. Config References

| File | Contents |
|------|----------|
| `configs/pairs.yaml` | Target pairs, PDB IDs, set assignment (core/auxiliary/blocked) |
| `configs/metrics.yaml` | Hit thresholds (T=0.0), enrichment K values, computed metrics |
| `configs/experiment.yaml` | Budget (n=1000), seeds, epochs, objectives, quality gate |
| `docs/goal.md` | Full hypothesis, notation, endpoint hierarchy, success criteria |
| `docs/selection_criteria.md` | Pair rationale, factorial design, core vs auxiliary split |
| `src/dualdocking/statistical.py` | Bootstrap test, FDR correction, Cliff's delta, Cohen's d |
