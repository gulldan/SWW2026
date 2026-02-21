# Statistical Analysis Plan (SAP)

## Double-Target RL Generation vs Single-Target Baselines

| Field | Value |
|-------|-------|
| **Version** | 1.1 |
| **Date** | 2026-02-08 |
| **Status** | FROZEN (amended v1.1 — see Section 11) |
| **Pre-registration ref** | `docs/preregistration.md` (commit `f2eceb2`) |
| **Pipeline entry point** | `cli/s5_analyze/compute_metrics.py` |

---

## 1. Study Overview

This SAP formalizes the statistical inference procedures for comparing
dual-target RL molecule generation (`double` mode) against single-target
baselines (`single_a`, `single_b`) on Boltz-2 evaluated dual hit-rate.

**Design:** Paired comparison across 5 core target pairs, each run with
5 independent seeds (RL random seeds), 3 modes, yielding 75 core runs
(5 pairs x 3 modes x 5 seeds).

**Unit of analysis:** Per-seed aggregate (one scalar metric per RL run).
Within-seed molecule-level resampling is handled by the hierarchical bootstrap.

## 2. Populations and Data Flow

### 2.1 Analysis Sets

| Set | Definition | N runs |
|-----|-----------|--------|
| **Primary** | Core 5 pairs, all 3 modes, 5 seeds | 75 |
| **Extended** | Core + Auxiliary (8 pairs) | 120 |
| **Per-protocol** | Runs passing fairness + quality gates | <= 120 |

### 2.2 Data Pipeline

```
results/{pair}/{mode}/seed_{seed}/
  ├── generated.parquet         # FREED++ output SMILES (Parquet with CSV fallback)
  ├── scored_joined.parquet     # Boltz-2 affinities (affinity_a, affinity_b)
  └── freedpp_workspace/
      └── reward_metadata.json  # Invalid/timeout fractions
```

**Gating order** (implemented in `dualdocking/analysis/metrics/computation.py:547-599`):

1. **Existence gate:** `scored_joined.parquet` must exist (CSV fallback supported).
2. **Fairness gate:** `validate_fairness(run_dir)` checks n_molecules in
   [990, 1010] and duplication_rate < 0.20
   (`dualdocking/analysis/fairness.py:44-86`).
3. **Quality gate:** `validate_reward_quality(run_dir)` checks
   invalid_fraction <= 0.10 and timeout_fraction <= 0.05
   (`dualdocking/analysis/fairness.py:89-167`).

Runs failing gate 2 (fairness) trigger hard abort (exit 1).
Runs failing gate 3 (quality) are excluded from analysis with a warning.

### 2.3 Missing Data

- **NaN affinity values:** Molecule treated as non-hit (conservative).
  Implementation: `np.isfinite()` mask in `dualdocking/analysis/metrics/computation.py:372-376`.
- **Invalid runs:** Excluded entirely after quality gate.
- **Missing runs:** Tolerated with `--allow-incomplete` (default);
  strict mode (`--no-allow-incomplete`) aborts if any expected run is missing.
- **Completeness reporting:** Missing rate recorded in
  `report/stats.json: _completeness` block.

## 3. Endpoints

### 3.1 Primary Endpoint

**Metric:** Dual hit-rate (`hr_both`)

```
hr_both = mean( (affinity_a <= T_A) AND (affinity_b <= T_B) )
```

where `T_A = T_B = 0.0` (1 uM) for all pairs.

Source: `configs/metrics.yaml: thresholds`, `dualdocking/analysis/metrics/computation.py:114-116`.

**Contrasts (per core pair):**

| Test ID | Treatment | Control | Statistic |
|---------|-----------|---------|-----------|
| H1a | `double` | `single_a` | `hr_both(double) - hr_both(single_a)` |
| H1b | `double` | `single_b` | `hr_both(double) - hr_both(single_b)` |

Implementation: `dualdocking/analysis/metrics/computation.py:684-689`.

### 3.2 Secondary Endpoints

| Test ID | Metric | Treatment | Control |
|---------|--------|-----------|---------|
| H2a | `balance` (median |a_A - a_B|) | `double` | `single_a` |
| H2b | `balance` (median |a_A - a_B|) | `double` | `single_b` |

Implementation: `dualdocking/analysis/metrics/computation.py:119-124`.

Additional secondary metrics (reported but not formally tested):

- Enrichment@K (K = 10, 50, 100, 500)
- Pareto hypervolume (reference point `r = (2.0, 2.0)`)
- Per-target `hr_a`, `hr_b`

### 3.3 Exploratory Endpoints

- Spearman rho (docking vs Boltz-2) per target — `cli/s5_analyze/proxy_correlation.py`
- Tanimoto overlap with ChEMBL dual-actives
- Internal diversity (inter-mode)
- Boltz-2 `affinity_probability_binary`

## 4. Statistical Methods

### 4.1 Hierarchical Bootstrap Test

**Rationale:** Standard bootstrap treats all molecules as independent,
ignoring within-seed correlation (pseudo-replication). The two-level
bootstrap correctly accounts for the nested structure: molecules within
seeds within modes.

**Procedure:**

For each contrast (e.g., H1a: `double` vs `single_a` on a given pair):

1. Let `G_ctrl = {df_1, ..., df_m}` be the DataFrames for control mode
   (one per seed, m = 5).
2. Let `G_treat = {df_1, ..., df_m}` be the DataFrames for treatment mode.
3. Compute observed difference: `obs_diff = mean(metric(G_treat)) - mean(metric(G_ctrl))`.
4. For b = 1, ..., B (B = 10,000):
   a. **Level 1 (seed resampling):** Draw m indices with replacement from {1, ..., m}
      for both control and treatment groups.
   b. **Level 2 (molecule resampling):** Within each resampled seed DataFrame,
      draw n rows with replacement (n = number of molecules in that seed).
   c. Compute metric on each resampled DataFrame.
   d. Record `diff_b = mean(metric_treat) - mean(metric_ctrl)`.
5. **95% CI:** Percentile method — `[percentile(diffs, 2.5), percentile(diffs, 97.5)]`.
6. **p-value:** `mean(diff_b <= 0)` (one-sided; fraction of replicates
   where treatment does not improve over control).

**Parameters:**

| Parameter | Value | Justification |
|-----------|-------|---------------|
| B (replicates) | 10,000 | Standard for percentile CI (Efron & Tibshirani) |
| RNG seed | 42 | Fixed for reproducibility |
| CI level | 95% | Convention |
| p-value type | One-sided | Directional: H1 is double > single (goal.md line 99) |

Implementation: `dualdocking/analysis/statistical.py:44-171`
(`hierarchical_bootstrap_test()`; includes fast numpy path for column-mean metrics).

### 4.2 Multiple Testing Correction

**Method:** Benjamini-Hochberg (BH) FDR correction.

**Scope:** All primary contrasts across core pairs. With 5 core pairs and
2 contrasts each (H1a, H1b), up to 10 raw p-values are corrected jointly.

**Alpha:** 0.05.

**Procedure:**
1. Rank all p-values in ascending order.
2. For rank i out of n tests: `p_adj[i] = min(p_adj[i+1], p_raw[i] * n / i)`.
3. Clip to [0, 1].

Implementation: `dualdocking/analysis/statistical.py:174-197` (`fdr_correction()`).

Applied in: `dualdocking/analysis/metrics/computation.py:731-750` (primary family FDR + secondary/auxiliary marking).

### 4.3 Effect Sizes

#### 4.3.1 Cliff's Delta (Primary)

Non-parametric effect size comparing all pairwise ordinal relationships
between treatment and control seed-level metric values.

```
delta = (n_greater - n_less) / (n_treat * n_ctrl)
```

where `n_greater` = number of pairs where treat > ctrl,
`n_less` = number of pairs where treat < ctrl.

**Interpretation thresholds** (Romano et al. 2006):

| |delta| | Interpretation |
|---------|----------------|
| < 0.147 | Negligible |
| < 0.33 | Small |
| < 0.474 | Medium |
| >= 0.474 | Large |

Implementation: `dualdocking/analysis/statistical.py:291-314` (`cliffs_delta()`).

#### 4.3.2 Cohen's d (Secondary)

Standardized mean difference using pooled standard deviation.

```
d = (mean_treat - mean_ctrl) / sqrt(pooled_var)
```

**Interpretation:** 0.2 small, 0.5 medium, 0.8 large (Cohen 1988).

Implementation: `dualdocking/analysis/statistical.py:317-340` (`cohens_d()`).

### 4.4 Significance Decision Rule

A contrast is declared **significant** if ALL three conditions hold
(conjunctive criterion):

1. BH-adjusted p-value < 0.05
2. |Cliff's delta| >= 0.33 (at least medium effect)
3. Bootstrap 95% CI excludes zero (ci_lower > 0 OR ci_upper < 0)

Implementation: `dualdocking/analysis/metrics/computation.py:762-765`.

```python
entry["significant"] = (
    p_value_adj < 0.05
    and abs(float(entry["cliffs_delta"])) >= 0.33
    and (float(entry["ci_lower"]) > 0 or float(entry["ci_upper"]) < 0)
)
```

## 5. Success Criteria

### 5.1 Primary Success (H1)

All three conditions must hold simultaneously:

1. **Prevalence:** `hr_both(double) > hr_both(single_x)` for >= 3 of 5
   core pairs (>= 60%).
2. **Statistical significance:** For pairs satisfying (1) — the contrast
   is significant per Section 4.4.
3. **Practical significance:** For pairs satisfying (1) — |Cliff's delta| >= 0.33.

### 5.2 No-Degradation Guard

For every core pair: per-target hit-rates `hr_a(double)` and `hr_b(double)`
must not show a large negative effect vs the corresponding single mode.

**Threshold:** Cliff's delta > -0.47 (below the large-effect boundary).

If violated: the pair is flagged and excluded from the success count in 5.1.

### 5.3 Sensitivity Robustness

The primary conclusion is considered robust if the success criterion (5.1)
holds at all three pre-registered thresholds: T = {-1.0, 0.0, +1.0}
(100 nM, 1 uM, 10 uM).

Implementation: `cli/s5_analyze/threshold_sensitivity.py`.

## 6. Claim Language Rules

Based on the outcome of the analysis, one of the following claim
statements is used. These are the **only** permitted claim formulations.

### 6.1 Full Success

**Conditions:** >= 3/5 core pairs significant (Section 5.1), no degradation
(Section 5.2), robust across thresholds (Section 5.3).

**Claim:**
> "Double-target RL generation with worst-case scalarization produces a
> significantly higher dual hit-rate than single-target baselines on
> {N}/5 core target pairs (BH-adjusted p < 0.05, |Cliff's delta| >= 0.33),
> with no per-target degradation. The effect is robust across hit-rate
> thresholds (100 nM, 1 uM, 10 uM)."

### 6.2 Partial Success (Primary Only)

**Conditions:** >= 3/5 core pairs significant, no degradation, but NOT
robust across all thresholds.

**Claim:**
> "Double-target RL generation improves dual hit-rate on {N}/5 core pairs
> at the primary threshold (T = 0.0, 1 uM), but the effect is sensitive
> to threshold choice."

### 6.3 Weak Evidence

**Conditions:** >= 3/5 core pairs show positive direction (double > single)
but |Cliff's delta| < 0.33 or CI includes zero.

**Claim:**
> "A consistent directional advantage for double-target RL is observed on
> {N}/5 core pairs, but the effect size is negligible-to-small
> (median Cliff's delta = {d}) and does not meet the pre-registered
> practical significance threshold."

### 6.4 Degradation Detected

**Conditions:** Any core pair shows Cliff's delta <= -0.47 on per-target
hit-rate.

**Claim (appended to any of the above):**
> "Double-target RL incurs a per-target hit-rate penalty on pair(s) {X}
> (Cliff's delta = {d}, large negative effect)."

### 6.5 No Evidence

**Conditions:** < 3/5 core pairs pass prevalence criterion (5.1.1).

**Claim:**
> "No evidence that double-target RL improves dual hit-rate: only {N}/5
> core pairs show positive direction."

### 6.6 Insufficient Data

**Conditions:** Missing rate > 20% of expected runs.

**Claim:**
> "Analysis inconclusive due to missing data ({missing_rate}% of expected
> runs unavailable). Results are reported descriptively only."

## 7. Pipeline-to-SAP Mapping

Every statistical procedure in this SAP is implemented in code with no
manual computation steps required.

| SAP Section | Code Location | Function/Block |
|-------------|---------------|----------------|
| 2.2 Fairness gate | `dualdocking/analysis/fairness.py:44-86` | `validate_fairness()` |
| 2.2 Quality gate | `dualdocking/analysis/fairness.py:89-167` | `validate_reward_quality()` |
| 2.3 Missing data | `dualdocking/analysis/metrics/computation.py:576-630` | `load_all_data()` completeness check |
| 3.1 hr_both | `dualdocking/analysis/metrics/computation.py:120-122` | `(aff_a <= thr_a) & (aff_b <= thr_b)` |
| 3.1 Thresholds | `configs/metrics.yaml:13-43` | `T_A = T_B = 0.0` |
| 4.1 Bootstrap | `dualdocking/analysis/statistical.py:97-254` | `hierarchical_bootstrap_test()` |
| 4.2 FDR | `dualdocking/analysis/statistical.py:257-288` | `fdr_correction()` |
| 4.3.1 Cliff's delta | `dualdocking/analysis/statistical.py:291-314` | `cliffs_delta()` |
| 4.3.2 Cohen's d | `dualdocking/analysis/statistical.py:317-340` | `cohens_d()` |
| 4.4 Significance rule | `dualdocking/analysis/metrics/computation.py:762-765` | Conjunctive criterion |
| 5.2 No-degradation guard | `dualdocking/analysis/metrics/computation.py:152-200` | Degradation Cliff's delta |
| 5.3 Threshold sensitivity | `cli/s5_analyze/threshold_sensitivity.py` | Full script |
| 6.x Claim audit | `cli/s7_audit/claim_audit.py` | Full script |
| Proxy-final correlation | `cli/s5_analyze/proxy_correlation.py` | Full script |

### Output Artifacts

| Artifact | Path | Contents |
|----------|------|----------|
| Summary metrics | `report/metrics.json` | hr_a, hr_b, hr_both, balance per pair/mode |
| Statistical tests | `report/stats.json` | Bootstrap results, effect sizes, FDR p-values |
| Threshold sensitivity | `report/data/threshold_sensitivity.json` | Per-threshold robustness |
| Proxy correlation | `report/data/proxy_final_corr.json` | Docking-Boltz-2 Spearman/Pearson |
| Degradation guard | `report/stats.json: _degradation` | Per-target Cliff's delta, pass/fail |
| Claim audit | `report/data/claim_audit.json` | Traceability, scenario selection, evidence map |

## 8. Reproducibility

| Parameter | Value | Source |
|-----------|-------|--------|
| Bootstrap RNG seed | 42 | `statistical.py:103` |
| Bootstrap replicates | 10,000 | `dualdocking/analysis/metrics/computation.py:27` |
| FDR alpha | 0.05 | `dualdocking/analysis/metrics/computation.py:758` |
| RL seeds | [42, 137, 256, 512, 1024] | `configs/experiment.yaml:11` |
| Molecules per run | 1,000 | `configs/experiment.yaml:7` |
| Hit threshold T | 0.0 (1 uM) | `configs/metrics.yaml:16-39` |
| Sensitivity thresholds | [-1.0, 0.0, 1.0] | `configs/metrics.yaml:68` |

**Execution:**

```bash
# Primary analysis
uv run python cli/s5_analyze/compute_metrics.py

# Threshold sensitivity
uv run python cli/s5_analyze/threshold_sensitivity.py

# Proxy-final correlation
uv run python cli/s5_analyze/proxy_correlation.py
```

All scripts read from `configs/` and `output/results/`, write to `report/`.
No manual editing of intermediate files is required.

## 9. Known Limitations

1. **m=5 seeds:** Bootstrap with 5 clusters yields coarse CI and p-values.
   Primary evidence weight is on effect size (Cliff's delta) and cross-pair
   consistency, not p-values alone (Cameron/Gelbach/Miller 2008).
2. **Target overlap:** AChE in 2 core pairs, BACE1 in 2, GSK3B in 2.
   Core pairs are not fully independent.
3. **Boltz-2 calibration:** Predicted values are IC50-like surrogates,
   not calibrated IC50. All comparisons are relative.
4. **In-domain bias:** PDB structures may appear in Boltz-2 training data.
5. **Docking compute asymmetry:** Double mode uses ~2x docking calls.
   Primary comparison is fixed molecule count; equal-docking-calls is
   sensitivity only.

## 10. References

- Cameron, A.C., Gelbach, J.B., Miller, D.L. (2008). Bootstrap-based
  improvements for inference with clustered errors. *Review of Economics
  and Statistics*, 90(3), 414-427.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
  2nd ed. Routledge.
- Benjamini, Y., Hochberg, Y. (1995). Controlling the false discovery rate.
  *J. Royal Statistical Society B*, 57(1), 289-300.
- Romano, J., Kromrey, J.D., Coraggio, J., Skowronek, J. (2006). Appropriate
  statistics for ordinal level data. *J. of Modern Applied Statistical Methods*.
- Efron, B., Tibshirani, R. (1993). *An Introduction to the Bootstrap*.
  Chapman & Hall/CRC.
- Passaro, S., et al. (2025). Boltz-2: Atomic Coordinate Generation with
  Confidence. *bioRxiv*. PMID 40667369.

## 11. Amendment Log

### v1.1 (2026-02-08)

**Rationale:** Align SAP with `docs/goal.md` (authoritative spec) after
code audit identified discrepancy between SAP v1.0 and goal.md line 99.

**Changes:**

1. **Section 4.1 p-value type:** Changed from two-sided
   (`mean(|diff_b| >= |obs_diff|)`) to one-sided (`mean(diff_b <= 0)`).
   Goal.md line 99 specifies "доля реплик с разностью ≤ 0" (fraction of
   replicates with difference <= 0), which is a directional (one-sided)
   test consistent with H1: double > single. The two-sided formulation in
   v1.0 was an error introduced during initial SAP drafting.
2. **Section 7 line references:** Updated to reflect `fast_column`/`fast_fn`
   optimization added to `hierarchical_bootstrap_test()` (performance
   improvement; no change to statistical semantics).
3. **Section 8 line references:** Updated accordingly.
4. **Blocked pair filtering:** `compute_metrics.py` and `aggregate.py` now
   exclude pairs with `set: blocked` in `pairs.yaml`. This was already
   implied by the design (blocked pairs have no results) but is now
   enforced explicitly in code.
5. **Degradation re-derivation:** `claim_audit.py` now independently
   re-derives degradation status from raw Cliff's delta instead of
   trusting pre-computed boolean flags, per SAP Section 5.2.
