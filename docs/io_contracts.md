# Pipeline I/O Contracts

Complete specification of inputs, outputs, file formats, and data schemas
for every stage of the DualDocking pipeline.

All paths are relative to `PROJECT_ROOT` unless noted otherwise.
Tabular data uses **Parquet** as the primary format with CSV fallback
(see `dualdocking/data/io_utils.py`).

---

## Stage 0 — Fetch

**Script**: `cli/s0_fetch/fetch_all.py`

Downloads all external data needed for offline operation.
After completion, no internet access is required.

### Inputs (Configuration)

| File | Format | Key Fields |
|------|--------|------------|
| `configs/pairs.yaml` | YAML | `pairs.{pair_id}.target_{a,b}.{id, pdb_id, uniprot, chain}` |
| `configs/calibration.yaml` | YAML | `selected`, `output_path`, `sample.{n, seed}`, `sources.*` |

### Outputs

| File | Format | Contents |
|------|--------|----------|
| `output/data/targets/{pair_id}/{A,B}/{pdb_id}.pdb` | PDB | 3D protein structure from RCSB (with retry) |
| `output/data/targets/{pair_id}/{A,B}/meta.yaml` | YAML | Copy of target config (id, pdb_id, uniprot, chain) |
| `output/data/targets/{target_id}.fasta` | FASTA | Protein amino-acid sequence from UniProt (with retry) |
| `output/data/targets/{target_id}.pdbqt` | PDBQT | Receptor prepared for docking (Meeko via Docker) |
| `output/data/targets/fetch_manifest.json` | JSON | SHA-256 checksums, URLs, sizes, timestamps for every downloaded PDB/FASTA |
| `output/cache/boltz_msa/{target_id}.csv` | CSV | Pre-computed MSA from ColabFold |
| `output/data/calibration/{source}_{n}_seed{seed}.smi` | SMI | Reservoir-sampled calibration SMILES |
| `output/data/calibration/manifest.json` | JSON | SHA256 checksums, timestamps, source metadata |
| `docker/freedpp/data/zinc_crem.json` | JSON | CReM fragment library for FREED++ |
| `docker/freedpp/data/alert_collections.csv` | CSV | Structural alert patterns (PAINS, SureChEMBL, Glaxo) |
| `output/cache/prepare_checkpoint.json` | JSON | Step completion tracking (`"{pair}/{role}/{step}": true`) |
| `output/data/chembl/{target_id}_actives.json` | JSON | Known actives from ChEMBL API (array of SMILES) |
| `output/data/chembl/{pair_id}_dual_actives.json` | JSON | Molecules active on both targets |
| `output/data/chembl/summary.json` | JSON | Per-pair active counts and dual-active overlaps |
| `output/bin/gnina` | Binary | GNINA static binary (Linux x86_64) for cross-validation scoring |
| Docker image `dualdocking/freedpp:latest` | — | CUDA + PyTorch + UniDock GPU + FREED++ |

### Optional Outputs

| File | Format | Contents |
|------|--------|----------|
| `output/data/baseline/zinc-druglike-cano.tar.xz` | tar.xz | ZINC20 druglike molecules archive |

---

## Stage 1 — Calibration

**Scripts**: `cli/s2_calibrate/dock_calibration.py` →
`cli/s2_calibrate/compute_normalization.py`

Docks the calibration SMILES set against every target, then computes
z-score normalization parameters for FREED++ reward.

### Inputs

| File | Format | Source |
|------|--------|--------|
| `output/data/calibration/*.smi` | SMI | Stage 0 (fetch) |
| `output/data/targets/{pair_id}/{A,B}/{pdb_id}.pdb` | PDB | Stage 0 |
| `configs/calibration.yaml` | YAML | Config (docking params) |
| `configs/pairs.yaml` | YAML | Config (target list) |

### Outputs

| File | Format | Schema / Key Fields |
|------|--------|---------------------|
| `output/data/calibration/scores/{target_id}.parquet` (+`.csv`) | Parquet | `score` (float64) — docking score kcal/mol |
| `output/data/calibration/scores/manifest.json` | JSON | `docking_program`, `docking_version`, `docker_image_digest`, `params.*`, `targets[]`, `date` |
| `configs/normalization.yaml` | YAML | Per-target: `mean`, `std`, `ci_mean`, `ci_std`, `n_samples`, `stable_mean`, `stable_std`; global: `version`, `calibration_set`, `checksum`, `boltz_version` |

### Intermediate Artifacts (Calibration)

| File | Format | Contents |
|------|--------|----------|
| `output/data/calibration/receptors/{target_id}.pdbqt` | PDBQT | Prepared docking receptor |
| `output/data/calibration/receptors/{target_id}.fixed.pdb` | PDB | PDBFixer-processed PDB (hydrogens added, HETATM removed) |

> **Important**: `.fixed.pdb` files are intermediate artifacts created by `ensure_receptor_pdbqt()` in `dualdocking/scoring/docking.py`. They should only exist in calibration/validation work directories — never in the raw targets directory (`output/data/targets/{pair}/{A,B}/`). The generation stage (`_resolve_pdb_path()`) explicitly excludes `.fixed.pdb` files to avoid name collisions.

### Validation Outputs (Optional)

| File | Format | Schema |
|------|--------|--------|
| `report/data/redocking_metrics.parquet` (+`.csv`) | Parquet | `target_id`, `pdb_id`, `ligand_resname`, `docking_score`, `rmsd`, `centroid_dist`, `status` |
| `report/data/enrichment_metrics.parquet` (+`.csv`) | Parquet | `target_id`, `pdb_id`, `n_actives`, `n_decoys`, `auroc`, `bedroc_alpha20`, `ef_1pct`, `ef_5pct`, `status`, `passed` |
| `output/data/enrichment/scores/{target_id}_actives.parquet` (+`.csv`) | Parquet | `smiles` (str), `score` (float64) — docking scores for ChEMBL actives |
| `output/data/enrichment/enrichment_results.json` | JSON | Full enrichment results manifest |
| `report/figures/enrichment/roc_{target_id}.png` | PNG | ROC curve per target |

---

## Stage 2 — Generation

**Scripts**: `cli/s3_generate/generate.py` →
`dualdocking/generation/backends.py` (resolver) →
built-in `dualdocking/generation/freedpp/` or external generator backend.

Runs the configured generator backend to produce molecules optimized for one
or both targets. Default backend is FREED++ (Docker).

### Inputs

| File | Format | Source |
|------|--------|--------|
| `configs/pairs.yaml` | YAML | Config |
| `configs/experiment.yaml` | YAML | Config (`budget.n_molecules`, `generator.backend`/`generator.name`, `generator.freedpp.*` incl. reward-mode passthrough toggles, `quality_gate.*`) |
| `configs/normalization.yaml` | YAML | Stage 1 |
| `output/data/targets/{pair_id}/{A,B}/{pdb_id}.pdb` | PDB | Stage 0 |
| `docker/freedpp/data/zinc_crem.json` | JSON | Stage 0 |
| `docker/freedpp/data/alert_collections.csv` | CSV | Stage 0 |
| Docker image `dualdocking/freedpp:latest` | — | Stage 0 |

### Outputs

| File | Format | Schema / Contents |
|------|--------|-------------------|
| `output/results/{pair}/{mode}/seed_{seed}/generated.parquet` (+`.csv`) | Parquet | **`smiles`** (str) — raw SMILES from FREED++ |
| `output/results/{pair}/{mode}/seed_{seed}/metadata.json` | JSON | `pair_id`, `mode`, `seed`, `generator_backend`, `timestamp`, `git_commit`, `docker_image_digest`, `n_molecules_raw`, `n_molecules_unique`, `duplication_rate`, `diversity.*`, `training_time_seconds`, `hardware.*` |

**`metadata.json` — `diversity` sub-object**:
`uniqueness`, `scaffold_diversity`, `internal_diversity`,
`internal_diversity_stderr`, `tanimoto_snn` — all float.

### Workspace Artifacts (intermediate, retained for provenance)

| File | Format | Contents |
|------|--------|----------|
| `…/freedpp_workspace/normalization.yaml` | YAML | Copy of configs/normalization.yaml |
| `…/freedpp_workspace/{receptor}.pdbqt` | PDBQT | Prepared docking receptor |
| `…/freedpp_workspace/dual_config.json` | JSON | Second receptor config (double mode only) |
| `…/freedpp_workspace/reward_metadata.json` | JSON | `total_calls`, `invalid_count_{a,b}`, `invalid_reward_fraction_{a,b}`, `timeout_count_{a,b}`, `timeout_fraction_{a,b}`, `reward_mode`, `reward_params`, `primary_target`, `run_valid` |
| `…/freedpp_workspace/output/run/mols/sample_*.json` | JSON | Arrays of generated SMILES (primary format) |
| `…/freedpp_workspace/output/run/mols/train_*.csv` | CSV | Per-epoch training molecules with docking scores |
| `…/freedpp_workspace/*.log` | Text | Docker stdout/stderr log |

**`train_*.csv` columns** (read-only, consumed by proxy_correlation):
- Single mode: `Smiles`/`smiles`, `DockingScoreProperty`
- Double mode: `Smiles`/`smiles`, `docking_a`, `docking_b`

### Model Cache

| File | Format | Contents |
|------|--------|----------|
| `output/cache/freedpp_models/{cache_key}/` | Directory | Trained FREED++ model checkpoint (reused when config matches) |

---

## Stage 3 — Scoring

**Script**: `cli/s4_score/score.py`

Scores generated molecules against both targets using the configured scorer
engine (default built-in Boltz-2). Supports checkpoint recovery, parallel GPU
scoring, and user-supplied molecule files.

### Inputs

| File | Format | Source |
|------|--------|--------|
| `output/results/{pair}/{mode}/seed_{seed}/generated.parquet` (or `.csv`) | Parquet | Stage 2 |
| `configs/pairs.yaml` | YAML | Config (`target_a.id`, `target_b.id`) |
| `configs/experiment.yaml` | YAML | Config (`objectives.*`, `scorer.engine`/`scorer.name`, scorer-specific params) |
| `configs/metrics.yaml` | YAML | Config (`thresholds.{pair}.target_{a,b}`) |
| `output/data/targets/{target_id}.fasta` | FASTA | Stage 0 |
| `output/cache/boltz_msa/{target_id}.csv` | CSV | Stage 0 (CLI backend only) |
| `output/cache/boltz_scores/{target_id}.{ctx[:8]}.json` | JSON | Score disk cache (auto-populated) |

**`--molecules PATH`** option: overrides `generated.parquet` with a user-supplied
file (`.parquet`/`.csv` with `smiles` column, or `.smi` one-per-line).

### Outputs

| File | Format | Schema |
|------|--------|--------|
| `…/scores_A.parquet` (+`.csv`) | Parquet | `smiles` (str), `score` (float64) |
| `…/scores_B.parquet` (+`.csv`) | Parquet | `smiles` (str), `score` (float64) |
| `…/scored_joined.parquet` (+`.csv`) | Parquet | See schema below |
| `…/scoring_context.json` | JSON | `scorer_engine`, `cache_context_hash`, `backend`, + backend-specific fields, `timestamp` |

### `scored_joined.parquet` schema

| Column | Type | Description |
|--------|------|-------------|
| `smiles` | str | Original SMILES from input |
| `smiles_canonical` | str | RDKit canonical SMILES (empty string if fails) |
| `affinity_a` | float64 | Scorer prediction for target A (built-in Boltz-2: log₁₀ IC₅₀ µM, lower=better) |
| `affinity_b` | float64 | Scorer prediction for target B |
| `is_hit_both` | bool | `affinity_a ≤ thr_a AND affinity_b ≤ thr_b` (inclusive) |
| `combined_min` | float64 | `max(affinity_a, affinity_b)` — worst-case objective |
| `combined_*` | float64 | Additional objectives from `experiment.yaml` (geometric, harmonic, weighted_sum, weighted_product, tchebycheff) |

Rows sorted ascending by the primary objective column (default: `combined_min`).

### Built-in Boltz-2 Cache

| File | Format | Contents |
|------|--------|----------|
| `output/cache/boltz_scores/{target_id}.{ctx[:8]}.json` | JSON | `{cache_key_hash: affinity_score, …}` — persisted after each sub-batch |
| `output/cache/boltz_scores/{target_id}.{ctx[:8]}.lock` | Lock | Advisory lock for concurrent access |

**Cache key** = SHA-256(`target_id || smiles || ctx_hash || seq_hash || msa_hash`).

### NIM Model Cache (built-in Boltz-2 NIM backend)

| File | Format | Contents |
|------|--------|----------|
| `output/cache/boltz_nim/` | Directory | NIM model weights, TensorRT engines (Docker-mounted as `/opt/nim/.cache`) |

External scorer engines may use additional cache locations outside this
contract; such paths must be documented by the backend implementation.

---

## Stage 4 — Analysis

**Scripts** (run in order):
1. `cli/s5_analyze/aggregate.py`
2. `cli/s5_analyze/compute_metrics.py`
3. `cli/s5_analyze/threshold_sensitivity.py`
4. `cli/s5_analyze/proxy_correlation.py`
5. `cli/s5_analyze/summarize.py`

### 4.1 aggregate.py

| Direction | File | Format | Schema |
|-----------|------|--------|--------|
| **In** | `output/results/{pair}/{mode}/seed_{seed}/scored_joined.parquet` | Parquet | All scored_joined columns |
| **In** | `configs/pairs.yaml`, `configs/experiment.yaml` | YAML | Pair defs, seeds, modes, quality_gate |
| **Out** | `report/data/all_results.parquet` | Parquet | All scored_joined columns + `pair` (str), `mode` (str), `seed` (int) |

Runs failing fairness/quality gates are excluded.

### 4.2 compute_metrics.py

| Direction | File | Format | Schema |
|-----------|------|--------|--------|
| **In** | `output/results/{pair}/{mode}/seed_{seed}/scored_joined.parquet` | Parquet | Core columns |
| **In** | `configs/pairs.yaml`, `configs/experiment.yaml`, `configs/metrics.yaml` | YAML | Thresholds, objectives, seeds |
| **Out** | `report/metrics.json` | JSON | See below |
| **Out** | `report/stats.json` | JSON | See below |

**`metrics.json`** structure:
```
{pair_id}.{mode}.{
  hr_a_mean, hr_a_sem,
  hr_b_mean, hr_b_sem,
  hr_both_mean, hr_both_sem,
  balance_mean, balance_sem,
  n_valid_runs, n_expected_runs, missing_rate,
  {objective}_mean, {objective}_sem
}
```

**`stats.json`** structure:
```
{pair_id}.{H1a|H1b|H2a|H2b}.{
  comparison, metric, observed_diff,
  p_value_raw, p_value_adjusted,
  cliffs_delta, cohens_d,
  ci_lower, ci_upper, significant
}
_degradation.{pair_id}.{check_name}.{cliffs_delta, degraded}
_completeness.{n_expected, n_valid, missing_rate, missing_runs[]}
```

Tests: H1a/H1b = dual hit rate (double vs single), H2a/H2b = balance.
FDR correction applied to core pairs H1a/H1b only.
Significance: `p_adj < 0.05 AND |Cliff's δ| ≥ 0.33 AND CI excludes zero`.

### 4.3 threshold_sensitivity.py

| Direction | File | Format | Schema |
|-----------|------|--------|--------|
| **In** | `output/results/…/scored_joined.parquet` | Parquet | `affinity_a`, `affinity_b`, `smiles_canonical` |
| **In** | `configs/metrics.yaml` | YAML | `sensitivity_thresholds` (list, default: [-1.0, 0.0, 1.0]) |
| **Out** | `report/data/threshold_sensitivity.parquet` (+`.csv`) | Parquet | `threshold`, `pair`, `mode`, `hr_both_mean`, `hr_both_sem`, `n_seeds`, `H1a_diff`, `H1a_p_adj`, `H1a_cliffs_d`, `H1a_sig`, `H1b_*` |
| **Out** | `report/data/threshold_sensitivity.json` | JSON | `thresholds_tested[]`, `results.{thr}.{pair}.{metrics, stats}`, `robustness_summary.{thr}.{n_core_significant, passes_success_criterion}` |

### 4.4 proxy_correlation.py

| Direction | File | Format | Schema |
|-----------|------|--------|--------|
| **In** | `output/results/…/scored_joined.parquet` | Parquet | `smiles_canonical`, `affinity_a`, `affinity_b` |
| **In** | `output/results/…/freedpp_workspace/output/run/mols/train_*.csv` | CSV | `Smiles`, `DockingScoreProperty`/`docking_a`/`docking_b` |
| **Out** | `report/data/proxy_final_corr.parquet` (+`.csv`) | Parquet | `pair`, `mode`, `target`, `n`, `spearman_rho`, `spearman_ci_lo`, `spearman_ci_hi`, `pearson_r`, `pearson_ci_lo`, `pearson_ci_hi`, `n_boltz`, `n_docking`, `n_joined`, `join_rate` |
| **Out** | `report/data/proxy_final_corr.json` | JSON | `pairs.{pair}.{mode}.{n_seeds_with_data, targets.{target_a,b}.{spearman_rho, pearson_r, …}}`, `summary.{median_spearman_rho, proxy_suitable}` |

### 4.5 summarize.py

| Direction | File | Format | Contents |
|-----------|------|--------|----------|
| **In** | `report/data/all_results.parquet` | Parquet | Aggregated data |
| **In** | `report/metrics.json`, `report/stats.json` | JSON | Metrics + tests |
| **In** | `configs/pairs.yaml`, `configs/metrics.yaml`, `configs/experiment.yaml` | YAML | Config |
| **Out** | `report/summary.txt` | Text | Per-pair score distributions, top-3 molecules, test results |

---

## Stage 5 — Report

**Scripts**:
1. `cli/s6_report/plot_results.py`
2. `cli/s6_report/render_molecules.py`
3. `cli/s6_report/generate_latex_report.py`

### 5.1 plot_results.py

| Direction | File | Format |
|-----------|------|--------|
| **In** | `report/data/all_results.parquet` | Parquet |
| **In** | `configs/experiment.yaml` | YAML |
| **Out** | `report/figures/{pair_id}_{scorecol}_modes.png` | PNG (150 DPI, 12×4 in) — histogram + ECDF per score column |
| **Out** | `report/figures/{pair_id}_boxplot.png` | PNG (150 DPI, 9×5 in) — grouped boxplot all metrics |

Score columns: `affinity_a`, `affinity_b`, `combined_min`, + configured objectives.

### 5.2 render_molecules.py

| Direction | File | Format |
|-----------|------|--------|
| **In** | `report/data/all_results.parquet` | Parquet |
| **In** | `configs/pairs.yaml`, `configs/metrics.yaml`, `configs/experiment.yaml` | YAML |
| **Out** | `report/figures/{pair_id}_molecules_{mode}.png` | PNG — 2D molecule grid per mode |
| **Out** | `report/figures/{pair_id}_molecules_top_hits.png` | PNG — combined top-N hits across modes |

### 5.3 generate_latex_report.py

| Direction | File | Format |
|-----------|------|--------|
| **In** | `report/metrics.json`, `report/stats.json`, `report/summary.txt` | JSON/Text |
| **In** | `report/templates/manuscript.tex.j2` | Jinja2 |
| **In** | `report/chapters/manifest.yaml` + `report/chapters/*.tex` | YAML/TeX |
| **In** | `report/figures/*.png` | PNG |
| **In** | `configs/pairs.yaml`, `configs/master_config.yaml` | YAML |
| **Out** | `report/manuscript.tex` | LaTeX |
| **Out** | `report/manuscript.pdf` | PDF (mode=full only, requires latexmk) |

---

## Stage 5b — GNINA Cross-Validation

**Script**: `cli/s6_report/score_gnina.py`

Scores generated molecules with GNINA CNN-based scoring as a third
independent scorer alongside Boltz-2 and UniDock.

### Inputs

| File | Format | Source |
|------|--------|--------|
| `output/results/{pair}/{mode}/seed_{seed}/scored_joined.parquet` | Parquet | Stage 3 |
| `output/data/targets/{pair_id}/{A,B}/{pdb_id}.pdb` | PDB | Stage 0 |
| `output/bin/gnina` | Binary | Stage 0 |

### Outputs

| File | Format | Schema |
|------|--------|--------|
| `output/results/{pair}/{mode}/seed_{seed}/gnina_scores_A.parquet` | Parquet | `smiles` (str), `gnina_score` (float64), `gnina_affinity` (float64), `cnn_score` (float64) |
| `output/results/{pair}/{mode}/seed_{seed}/gnina_scores_B.parquet` | Parquet | Same schema |

---

## Stage 5c — MD Validation

**Script**: `cli/s6_md/run_md.py`

Runs publication-quality molecular dynamics simulations for top candidate
molecules from each generation mode. Validates binding stability via
backbone/ligand RMSD, RMSF, radius of gyration, H-bonds, and convergence
assessment (KS test).

### Prerequisites

- GROMACS >= 2024 with GPU support (CUDA recommended, CPU fallback available)
- AmberTools + acpype (ligand parameterization)
- OpenBabel (`obabel`, SDF → MOL2 conversion)
- gmx_MMPBSA (optional, for MM-GBSA rescoring)

Installation: `conda install -c conda-forge -c bioconda gromacs ambertools acpype openbabel`
or via Docker image.

### CLI Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--pair` | str | required | Pair ID from `configs/pairs.yaml` |
| `--mode` | str | all | Generation mode (default: all configured modes) |
| `--top-n` | int | 3 | Top-N molecules per mode |
| `--mmpbsa` | flag | off | Run MM-GBSA via gmx_MMPBSA |
| `--simulation-time` | float | 50.0 | Override simulation time (ns) |
| `--no-gpu` | flag | off | Disable automatic GPU acceleration |
| `--dry-run` | flag | off | Print plan without executing |

### Inputs

| File | Format | Source |
|------|--------|--------|
| Top-N molecules per mode (from `scored_joined.parquet`) | Parquet | Stage 3 |
| `output/data/targets/{pair_id}/{A,B}/{pdb_id}.pdb` | PDB | Stage 0 |

### GROMACS Pipeline (22 commands)

**Setup (14 commands):**
pdb2gmx (`-ff amber14sb`) → editconf → solvate → grompp ions → genion (`stdin: SOL`) →
grompp EM → mdrun EM → make_ndx (`stdin: Protein | r LIG / q`) →
grompp NVT → mdrun NVT → grompp NPT → mdrun NPT → grompp prod → mdrun prod

**Trajectory preprocessing (3 commands):**
trjconv `-pbc whole` (`stdin: System`) →
trjconv `-center -pbc mol` (`stdin: Protein / System`) →
trjconv `-fit rot+trans` (`stdin: Backbone / System`)

**Analysis (5 commands):**
`gmx rms` backbone (`stdin: Backbone/Backbone`) →
`gmx rms` ligand (`stdin: LIG/LIG`) →
`gmx rmsf` per-residue (`stdin: Backbone`) →
`gmx gyrate` (`stdin: Protein`) →
`gmx hbond-legacy` (`stdin: Protein/LIG`, falls back to `gmx hbond`)

### MDP Protocol (Modern: amber14sb + C-rescale)

Force field: **amber14sb** (Maier et al. 2015, built into GROMACS 2026).
All MDPs include: `cutoff-scheme=Verlet`, `rcoulomb=1.0`, `rvdw=1.0`
(AMBER standard), `DispCorr=EnerPres`, `fourierspacing=0.12`, `pme_order=4`,
`vdw-modifier=Potential-shift`, `constraint-algorithm=LINCS`, `lincs_order=4`.
Thermostat coupling groups: `Protein_LIG` / `Water_and_ions`.

| Phase | Duration | Barostat | Thermostat τ | Notes |
|-------|----------|----------|--------------|-------|
| EM | 50k steps | — | — | Steepest descent, emtol=1000 |
| NVT | 200 ps (configurable) | none | V-rescale 0.1 ps | Position restraints (`-DPOSRES -DPOSRES_LIG`), velocity generation |
| NPT | 200 ps (configurable) | **C-rescale** τ=1.0 ps | V-rescale 0.1 ps | Position restraints, correct NPT ensemble (Bernetti & Bussi 2020) |
| Production | 50 ns (configurable) | **Parrinello-Rahman** τ=2.0 ps | V-rescale 1.0 ps | No position restraints, no energygrps |

### GPU Acceleration

When `gpu_auto=True` (default), `nvidia-smi` is probed at startup.
If an NVIDIA GPU is found, all `mdrun` calls receive:
`-nb gpu -pme gpu -bonded gpu -update gpu`. Disable with `--no-gpu`.

### Outputs

| File | Format | Contents |
|------|--------|----------|
| `output/md/{pair}/{mode}/mol_{id}/target_{A,B}/rmsd_backbone.csv` | CSV | Per-frame backbone RMSD (ns, Å) |
| `output/md/{pair}/{mode}/mol_{id}/target_{A,B}/rmsd_ligand.csv` | CSV | Per-frame ligand RMSD (ns, Å) |
| `output/md/{pair}/{mode}/mol_{id}/target_{A,B}/rmsf.csv` | CSV | Per-residue backbone RMSF (residue, Å) |
| `output/md/{pair}/{mode}/mol_{id}/target_{A,B}/radius_gyration.csv` | CSV | Per-frame protein radius of gyration (ps, Å) |
| `output/md/{pair}/{mode}/mol_{id}/target_{A,B}/hbonds.csv` | CSV | Per-frame protein-ligand hydrogen bond count |
| `output/md/{pair}/{mode}/mol_{id}/target_{A,B}/trajectory.xtc` | XTC | PBC-corrected, centered, fitted trajectory |
| `output/md/{pair}/{mode}/mol_{id}/target_{A,B}/mmpbsa.csv` | CSV | MM-GBSA binding ΔG (optional, kcal/mol) |
| `output/md/{pair}/{mode}/mol_{id}/target_{A,B}/md_result.json` | JSON | All metrics for this molecule×target run |
| `output/md/md_summary.parquet` | Parquet | See schema below |

### `md_summary.parquet` schema

| Column | Type | Description |
|--------|------|-------------|
| `pair` | str | Pair ID |
| `mode` | str | Generation mode |
| `mol_id` | int | Molecule index |
| `target` | str | Target ID |
| `target_label` | str | `A` or `B` |
| `smiles` | str | Molecule SMILES |
| `mean_rmsd` | float64 | Mean ligand RMSD over stability window (Å) |
| `max_rmsd` | float64 | Max ligand RMSD over stability window (Å) |
| `stable` | bool | `mean_rmsd < 3.0` Å |
| `mean_hbonds` | float64 | Mean protein-ligand H-bond count |
| `mean_energy` | float64 | Mean total interaction energy (kJ/mol) |
| `backbone_rmsd_mean` | float64 | Mean backbone RMSD (Å) |
| `backbone_rmsd_final` | float64 | Final-frame backbone RMSD (Å) |
| `rmsf_binding_pocket` | float64 | Mean per-residue backbone RMSF (Å) |
| `rg_mean` | float64 | Mean protein radius of gyration (Å) |
| `rg_std` | float64 | Std of radius of gyration (Å) |
| `converged` | bool | KS test: p > 0.05 AND |mean_diff| < 0.5 Å |
| `convergence_ks_pvalue` | float64 | 2-sample KS p-value (first vs second half of backbone RMSD) |
| `coulomb_sr` | float64 | Mean short-range Coulomb interaction energy (kJ/mol) |
| `lj_sr` | float64 | Mean short-range LJ interaction energy (kJ/mol) |
| `mmpbsa_dg` | float64 | MM-GBSA binding ΔG (kcal/mol, NaN if not run) |

### Convergence Assessment

Backbone RMSD is split into two halves. A 2-sample Kolmogorov-Smirnov test
determines whether the distributions differ significantly. The simulation
is considered converged when `p > alpha` (default 0.05) AND
`|mean_second_half - mean_first_half| < 0.5 Å`.

### Stability Criterion

Ligand RMSD < 3.0 Å (configurable) over the last 25 ns (configurable)
of the production trajectory.

---

## Stage 6 — Audit

**Script**: `cli/s7_audit/claim_audit.py`

Verifies claim traceability per SAP Section 6.

### Inputs

| File | Format | Source |
|------|--------|--------|
| `report/stats.json` | JSON | Stage 4.2 |
| `report/metrics.json` | JSON | Stage 4.2 |
| `report/data/threshold_sensitivity.json` | JSON | Stage 4.3 |
| `configs/pairs.yaml` | YAML | Config |

### Output

| File | Format | Contents |
|------|--------|----------|
| `report/data/claim_audit.json` | JSON | `checks[]`, `pass`, `issues[]`, `claim_scenario`, `evidence_map`, `summary` |

**Exit code**: 0 = PASS, 1 = FAIL (with `--strict`).

**Claim scenarios** (SAP Section 6):

| Scenario | Condition |
|----------|-----------|
| `6.1_full_success` | ≥3 core pairs significant + threshold-robust + no degradation |
| `6.2_partial_success` | ≥3 core pairs significant + not threshold-robust |
| `6.3_weak_evidence` | ≥3 positive direction but <3 significant |
| `6.4_degradation_detected` | Appended when Cliff's δ ≤ −0.47 for any per-target check |
| `6.5_no_evidence` | <3 positive direction |
| `6.6_insufficient_data` | Missing rate > 20% |

---

## Cache Directory Layout

All caches reside under `output/cache/`:

```
output/cache/
├── boltz_scores/          # Score cache (JSON per target×context)
│   ├── {target_id}.{ctx[:8]}.json
│   └── {target_id}.{ctx[:8]}.lock
├── boltz_msa/             # Pre-computed MSA (from Stage 0)
│   └── {target_id}.csv
├── boltz_nim/             # NIM model weights + TRT engines
│   └── (Docker-mounted as /opt/nim/.cache)
├── freedpp_models/        # Trained FREED++ model checkpoints
│   └── {cache_key}/
└── prepare_checkpoint.json
```

Boltz CLI model weights: controlled by `BOLTZ_CACHE` env var
(default `~/.boltz/`, recommend `output/cache/boltz_models/`).

---

## Configuration Files Summary

| File | Format | Used By |
|------|--------|---------|
| `configs/pairs.yaml` | YAML | All stages |
| `configs/experiment.yaml` | YAML | Generation, Scoring, Analysis |
| `configs/calibration.yaml` | YAML | Fetch, Calibration |
| `configs/normalization.yaml` | YAML | Generation, Scoring (produced by Stage 1) |
| `configs/metrics.yaml` | YAML | Scoring, Analysis, Report |
| `configs/reward_modes.yaml` | YAML | Shared reward-mode registry for generation backends |
| `configs/filters.yaml` | YAML | Post-scoring drug-likeness filtering (Lipinski, QED, PAINS, SA, Veber) |
| `configs/master_config.yaml` | YAML | LaTeX report |

---

## Path Placeholders

| Placeholder | Values |
|-------------|--------|
| `{pair_id}` | e.g. `CDK2_ERK2`, `BACE1_GSK3B`, `AChE_BChE` |
| `{target_id}` | e.g. `CDK2`, `ERK2`, `BACE1`, `GSK3B` |
| `{mode}` | `single_a`, `single_b`, `double`, `linear_equal`, `biased_a`, `selective_a`, `selective_weighted` (or plugin-defined mode names) |
| `{seed}` | `42`, `137`, `256`, `512`, `1024` (from experiment.yaml) |
| `{ctx[:8]}` | First 8 chars of scoring context SHA-256 hash |
| `{cache_key}` | Full SHA-256 hash of FREED++ config parameters |
