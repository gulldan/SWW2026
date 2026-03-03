# Protein Pair Selection Criteria

## Pair Requirements (Primary Analysis)

### For FREED++ (UniDock GPU Generation)

1. **PDB structure**: resolution ≤ 2.5 Å with a co-crystallized ligand (defines the grid box center). Exception: EGFR 1M17 (2.6 Å) — auxiliary pair, accepted as the only human EGFR co-crystal with erlotinib; does not affect primary analysis.
2. **Well-defined binding site**: unambiguous binding pocket for a correct reward signal.
3. **Size**: protein ≤ 600 residues (in `double` mode UniDock GPU is invoked twice per RL episode step).
4. **Biological rationale**: the pair must have justification in multi-target drug design.

### For Boltz-2 (Validation)

5. **Compatibility**: soluble proteins without cofactors in the binding pocket (Boltz-2 affinity head does not model cofactors). Pairs with cofactors (AChE_MAOB) are allowed only in exploratory analysis with `affinity_probability_binary` and are marked `warning` in the pair table.
6. **Sequence**: UniProt FASTA is sufficient (PDB is not required for Boltz-2).
7. **Size**: ≤ 1000 tokens (Boltz-2 limit).

### Cofactors — A Critical Boltz-2 Limitation

> Boltz-2 paper (Limitations): *"the affinity module does not explicitly handle such cofactors, including ions, water, or multimeric binding partners"*
> Rowan FAQ: *"affinity predictions will be unreliable"* for targets with cofactors in the binding pocket.

Pairs with cofactors (COX1_COX2 — heme, AChE_MAOB — FAD): the "Cofactor: Present" axis refers to the **target biology and Boltz-2 reliability**, not to the docking receptor state. The receptor preparation protocol (`removeHeterogens(keepWater=False)`) removes **all** HETATM records, including cofactors; thus UniDock docks to a **cofactor-free** receptor for all targets without exception. This means that docking scores for COX/MAO-B reflect binding to a "stripped" pocket and may be less biologically representative. **Boltz-2 affinity** for cofactor-bearing targets is also unreliable (the affinity head does not model cofactors). For MAO-B, `affinity_probability_binary` is recommended instead of `affinity_pred_value`.

## Summary Table of All 13 Pairs

Classification along three axes: **Disease** (AD / Non-AD / Oncology), **Structure** (Homo / Hetero), **Cofactor** (Clean / Present). See also: [Factorial Design](#factorial-experiment-design).

| # | Pair | Disease | Structure | Cofactor | Set | PDB A / PDB B | Size* | Boltz-2 |
|---|------|---------|-----------|----------|-----|---------------|-------|---------|
| 1 | CDK2_ERK2 | Non-AD | Homo | Clean | **Core** | 1HCL¹ / 4QTA | ~629 | ok |
| 2 | EGFR_HER2 | Non-AD | Homo | Clean | Auxiliary | 1M17 / 3PP0 | ~671 | ok |
| 3 | COX1_COX2 | Non-AD | Homo | Present | Blocked | 1EQG / 3LN1 | ~1120 | warning (heme) |
| 4 | AChE_BChE | AD | Homo | Clean | **Core** | 4EY7 / 4BDS | ~1073 | ok |
| 5 | **BACE1_GSK3B** | AD | Hetero | Clean | **Core** | 6EQM / 5KPK | ~724 | ok |
| 6 | **AChE_BACE1** | AD | Hetero | Clean | **Core** | 4EY7 / 6EQM | ~943 | ok |
| 7 | **GSK3B_CDK5** | AD | Hetero | Clean | **Core** | 5KPK / 1UNL | ~642 | ok |
| 8 | BACE1_CDK5 | AD | Hetero | Clean | Auxiliary | 6EQM / 1UNL | ~692 | ok |
| 9 | AChE_MAOB | AD | Hetero | Present | Auxiliary | 4EY7 / 2V5Z | ~1063 | warning (FAD) |
| 10 | MEK1_MEK2 | Oncology | Homo | Clean | Auxiliary | 7JUR / 3PP1 | ~740 | ok |
| 11 | FGFR1_FGFR2 | Oncology | Homo | Clean | Auxiliary | 5EW8 / 2PSQ | ~620 | ok |
| 12 | EZH2_EZH1 | Oncology | Homo | Clean | Auxiliary | 5LS6 / 5LS6² | ~760 | ok |
| 13 | BRD4_BRD2 | Oncology | Homo | Clean | Auxiliary | 3MXF / 3ONI | ~530 | ok |

\* **Size** — approximate total residue count for both proteins after PDBFixer processing (trimming, HETATM removal). Deviations of ±20–50 aa from the RCSB deposited sequence are possible due to missing residues, non-standard residue replacement, and removal of terminal tags.

² **EZH2_EZH1** — both targets use PDB 5LS6 (PRC2 complex structure); EZH2 and EZH1 share 94% SET domain identity and the same overall PRC2 architecture.

## Human-Only Enforcement

Starting with ENG-10, `load_pairs()` validates that all targets have `organism: "Homo sapiens"`.
Non-human targets (e.g. COX1 — Ovis aries) are blocked by default.

- **Bypass**: Set `DUALDOCKING_ALLOW_NON_HUMAN=true` for exploratory runs
- **COX1_COX2**: Blocked in production. Available for exploratory analysis only.

## Factorial Experiment Design

The selection of 13 pairs is determined by a pre-fixed factorial design along three axes:

- **Structural relation**: Homo (same family, high homology) / Hetero (different families)
- **Cofactor**: Clean (no cofactor in the pocket biologically) / Present (cofactor present biologically; affects interpretation of docking and Boltz-2 — see [Cofactors](#cofactors--a-critical-boltz-2-limitation))
- **Disease context**: AD (Alzheimer's disease) / Non-AD (controls)

### Design Matrix

| # | Pair | Structure | Cofactor | Disease | Set | ML Hypothesis |
|---|------|-----------|----------|---------|-----|---------------|
| 1 | CDK2_ERK2 | Homo | Clean | Non-AD | **Core** | High homology — double-dock should easily find dual inhibitors; "easy case" test |
| 2 | EGFR_HER2 | Homo | Clean | Non-AD | Auxiliary | Second homo control: generalizability to another kinase subfamily (ErbB) |
| 3 | COX1_COX2 | Homo | Present | Non-AD | Blocked | Non-human PDB: COX1 (Ovis aries), COX2 (Mus musculus); cofactor (heme). Exploratory only |
| 4 | AChE_BChE | Homo | Clean | AD | **Core** | AD-relevant homo pair: does double-dock retain advantage for large proteins (~1073 residues) |
| 5 | BACE1_GSK3B | Hetero | Clean | AD | **Core** | **Central test**: can double-dock find molecules active on structurally dissimilar targets |
| 6 | AChE_BACE1 | Hetero | Clean | AD | **Core** | Maximum ChEMBL data (3524 dual compounds) — hypothesis + validation via chemical similarity |
| 7 | GSK3B_CDK5 | Hetero | Clean | AD | **Core** | Dual tau kinases at minimal compute (~642 residues). Functional proximity vs structural test |
| 8 | BACE1_CDK5 | Hetero | Clean | AD | Auxiliary | Robustness: does double-dock advantage persist when replacing GSK3B→CDK5 as tau kinase |
| 9 | AChE_MAOB | Hetero | Present | AD | Auxiliary | Sensitivity: cofactor (FAD) impact on reward signal. Boltz-2 affinity exploratory only |
| 10 | MEK1_MEK2 | Homo | Clean | Oncology | Auxiliary | MAPK pathway selectivity (~80% protein ID); trametinib approved 2013. Allosteric pocket test |
| 11 | FGFR1_FGFR2 | Homo | Clean | Oncology | Auxiliary | Highest in-set homology (~90% kinase domain ID); erdafitinib approved 2024. "Easy homo" oncology control |
| 12 | EZH2_EZH1 | Homo | Clean | Oncology | Auxiliary | PRC2 methyltransferase (94% SET domain ID); valemetostat dual EZH1/2 approved JP 2022. Epigenetic target test |
| 13 | BRD4_BRD2 | Homo | Clean | Oncology | Auxiliary | BET bromodomain (85–92% BD1 ID); JQ1 pan-BET inhibitor. Compact pair (~530 residues) |

### Core vs Auxiliary

**Core set (5 pairs)** — primary hypothesis test. Statistical conclusions are drawn from these pairs:
- Cover both homology levels (homo: CDK2_ERK2, AChE_BChE; hetero: BACE1_GSK3B, AChE_BACE1, GSK3B_CDK5)
- All cofactor-free → Boltz-2 affinity is considered a sufficiently reliable orthogonal metric given concordance with docking and ChEMBL validation
- Each protein participates in at most 2 core pairs (AChE in 2, BACE1 in 2, GSK3B in 2; CDK2, ERK2, BChE, CDK5 — in 1)

**Auxiliary set (7 pairs + 1 blocked)** — robustness and sensitivity analysis:
- EGFR_HER2: second homo Non-AD control (not in core since CDK2_ERK2 already covers this)
- BACE1_CDK5: tau kinase substitution test (overlaps with BACE1_GSK3B and GSK3B_CDK5)
- AChE_MAOB: cofactor-sensitive pair, docking + binary Boltz-2 only
- MEK1_MEK2: oncology MAPK pathway homo pair; allosteric pocket selectivity test
- FGFR1_FGFR2: highest-homology oncology pair (~90% kinase domain ID); "easy homo" control
- EZH2_EZH1: epigenetic (PRC2 methyltransferase) homo pair; dual inhibitor clinical precedent (valemetostat)
- BRD4_BRD2: BET bromodomain homo pair; compact size (~530 residues), pan-BET inhibitor precedent

**Blocked:** COX1_COX2 (non-human PDB structures: COX1 — Ovis aries, COX2 — Mus musculus). Excluded from both primary and auxiliary analyses.

> The primary hypothesis is tested on 5 pre-fixed core pairs; auxiliary pairs are used for robustness/sensitivity analyses. This division was fixed **before experiments began** to preclude cherry-picking.

### Target Overlap and Statistical Independence

The same proteins (AChE, BACE1, GSK3B) participate in multiple pairs.
This is useful for analyzing the effect of the second target, but a naive "N=13 pairs" does not constitute
13 independent observations.

- Primary statistical tests are conducted on the **core set** (5 pairs), where overlap is minimized
- Auxiliary pairs are used as **within-target sensitivity analysis**, not as additional independent observations
- The limitations section explicitly states: *"because core pairs share certain proteins, effect sizes are interpreted with caution"*

## Non-AD Controls (Pairs 1–3)

- **Kinase pairs** (CDK2/ERK2, EGFR/HER2): well-studied binding pockets, numerous known inhibitors, extensive docking benchmarks.
- **COX pair**: gold standard for the selectivity problem in medicinal chemistry. Heme in the active site is problematic for Boltz-2. Blocked (non-human PDB: COX1 — Ovis aries, COX2 — Mus musculus).

## AD Homologous Pair (Pair 4)

- **AChE_BChE**: formally AD-relevant (both targets are therapeutic targets for AD), but also serves as a **homology control** — the only AD pair with high structural homology. Acts as a bridge between the Non-AD homo controls (CDK2_ERK2, EGFR_HER2) and the AD hetero pairs (BACE1_GSK3B, etc.).

## AD-Specific Pairs (Pairs 5–9)

### BACE1_GSK3B — Best Pair

- **Biology**: GSK3B regulates BACE1 expression via NF-κB (Ly et al. 2013, J Clin Invest). Notopterol — a validated in vivo dual inhibitor (Jiang et al. 2020, APP/PS1 mice).
- **Structures**: BACE1 6EQM (1.35 Å, CNP520 IC50 11 nM) + GSK3B 5KPK (2.4 Å, BRD0209).
- **ChEMBL**: >8000 BACE1 records, >4000 GSK3B records.
- **Compute**: ~750 total residues — fast UniDock GPU, comfortable for Boltz-2.

### AChE_BACE1 — Classic Pair

- **Biology**: The most studied dual-target combination in AD. Stern et al. 2022 — from in silico to in vivo.
- **ChEMBL**: 3524 dual-compound records (maximum overlap for validation).
- **Compute**: ~943 total residues — borderline but workable.

### GSK3B_CDK5 — Compute-Optimal

- **Biology**: Dual tau kinases. GSK3B phosphorylates Ser396/Ser404, CDK5 — Ser202/Thr205 (AT8 epitope). Both cause NFT in AD.
- **Structures**: GSK3B 5KPK (2.4 Å) + CDK5 1UNL (2.2 Å, Roscovitine IC50 0.2 μM).
- **Compute**: ~642 total residues — **smallest pair**, fastest `double` run.

### BACE1_CDK5 — Robustness Test (Auxiliary)

- **Biology**: Amyloid (BACE1) + Tau (CDK5). The p25/CDK5 complex regulates BACE1 activity. Less studied pair — originality.
- **Compute**: ~692 total residues — fast.
- **Role**: Auxiliary pair for **robustness analysis** — tests whether the double-dock advantage persists when substituting the tau kinase GSK3B→CDK5. Overlaps with core pairs BACE1_GSK3B and GSK3B_CDK5, hence excluded from the core set.

### AChE_MAOB — With Caveats

- **Biology**: Clinical precedent — ladostigil (Phase II). Rivastigmine + rasagiline pharmacophores.
- **Caveat**: MAO-B contains a covalent FAD biologically. However, the receptor preparation protocol removes all HETATM (`removeHeterogens`), so UniDock docks to a **cofactor-free** MAO-B receptor. This means docking scores for MAO-B reflect binding to a stripped pocket and may not capture steric/electrostatic effects of FAD. Boltz-2 affinity for MAO-B is also **unreliable** (affinity head does not model cofactors) — interpret with caution; `affinity_probability_binary` is recommended.

## Oncology Kinase & Epigenetic Pairs (Pairs 10–13)

Four additional auxiliary pairs extending the design to oncology targets. All are homologous (Homo), cofactor-free (Clean), and serve as out-of-disease-area generalizability controls.

### MEK1_MEK2 — MAPK Selectivity

- **Biology**: Dual MAP kinase kinases in the RAS-RAF-MEK-ERK pathway. ~80% protein identity. Conserved allosteric pocket. Trametinib (MEK1/2 inhibitor) approved 2013 for melanoma.
- **Structures**: MEK1 7JUR + MEK2 3PP1.
- **Role**: Tests double-dock on allosteric (non-ATP) binding pockets.

### FGFR1_FGFR2 — Highest Homology

- **Biology**: Fibroblast growth factor receptors, ~90% kinase domain identity (highest in the auxiliary set). Erdafitinib (pan-FGFR) approved 2024 for bladder cancer.
- **Structures**: FGFR1 5EW8 + FGFR2 2PSQ.
- **Role**: "Easy homo" oncology control — double-dock should trivially outperform single when targets are near-identical.

### EZH2_EZH1 — Epigenetic Target

- **Biology**: PRC2 histone methyltransferases, 94% SET domain identity. EZH1 compensates EZH2 loss; valemetostat (dual EZH1/2) approved in Japan 2022 for ATL.
- **Structures**: Both use PDB 5LS6 (shared PRC2 architecture). WARNING: same PDB for both targets — binding pockets are structurally near-identical.
- **Role**: Tests double-dock on epigenetic (non-kinase) targets with extremely high homology.

### BRD4_BRD2 — BET Bromodomain

- **Biology**: BET family bromodomains, 85–92% BD1 identity, ~95% pocket conservation. JQ1 is a well-characterized pan-BET inhibitor; I-BET762 reached Phase I.
- **Structures**: BRD4 3MXF + BRD2 3ONI.
- **Role**: Compact pair (~530 total residues) — fastest compute; tests epigenetic reader domain targets.

## Boltz-2 Scoring Protocol

### Two Usage Modes

Boltz-2 is used as an **orthogonal scorer** (independent of the UniDock reward) in two modes:

| Mode | When Applied | Metric | Status |
|------|-------------|--------|--------|
| **Primary (quantitative)** | All cofactor-free pairs: 5 core + 6 auxiliary (11 of 12 active) | `affinity_pred_value` (log10 IC50-like, μM) | Primary endpoint |
| **Exploratory (binary)** | Cofactor-bearing pairs (AChE_MAOB) | `affinity_probability_binary` (hit/non-hit) | Exploratory only |

### Primary Mode: `affinity_pred_value`

For all pairs **without cofactors** in the binding pocket:
- Continuous metric: predicted log10(IC50-like, μM) — a surrogate trained on a mixture of Ki/Kd/IC50/EC50
- Used for molecule ranking and computing hr_both, enrichment@K
- Parameters are standardized in `boltz2.py` (`_BOLTZ_CLI_ARGS_BASE`):
  `recycling_steps: 3, sampling_steps: 50, diffusion_samples: 1,
   sampling_steps_affinity: 50, diffusion_samples_affinity: 1`
  (reduced from initial 200/4 for computational feasibility; quality impact
  is negligible for ranking — Boltz-2 is used as a relative comparator,
  not for absolute affinity prediction)

### Exploratory Mode: `affinity_probability_binary`

For cofactor-bearing pairs (AChE_MAOB — FAD):
- Binary metric: binding probability (hit/non-hit)
- **Not included in the primary endpoint** — qualitative analysis only
- Reason: Boltz-2 affinity head does not model cofactors (Passaro et al. 2025, Limitations)

> For AChE_MAOB the primary metric is the docking score (UniDock GPU). Boltz-2 is used
> only as an auxiliary binary indicator. Results for this pair
> **are not included in the primary statistical analysis**.

### In-Domain Bias and Interpretation

Boltz-2 was trained on large-scale datasets of protein structures and protein–ligand complexes.
The PDB structures used here (or their close analogues) were very likely
part of the training set.

Therefore:
- Boltz-2 metrics are interpreted as a **relative ranking** between strategies
  (single vs double), not as absolute predictions of experimental IC50
- The primary value of Boltz-2 in this project is its **orthogonality** to UniDock:
  two independent methods (physics-based docking vs learned affinity) should
  yield a concordant signal, thereby increasing confidence in the results
- For pairs with extensive ChEMBL data (BACE1, AChE, GSK3B), Tanimoto-similarity
  enrichment against known actives is additionally performed — this constitutes
  a third validation source, independent of both scorers

## Receptor Preparation Protocol

An **identical** preparation protocol is used for all targets,
eliminating hidden variability between pairs.

### Step 1: PDB → Fixed PDB (`pdbfixer_add_h.py`)

Executed inside a Docker container (`dualdocking/freedpp`) with PDBFixer + OpenMM:

1. `findMissingResidues()` — detect missing residues
2. `findNonstandardResidues()` + `replaceNonstandardResidues()` — replace non-standard amino acids
3. `removeHeterogens(keepWater=False)` — remove **all** HETATM records (water, ions, ligands, cofactors). **Important**: this means cofactors (HEM for COX, FAD for MAO-B) are also removed; docking is performed against a cofactor-free receptor for all targets without exception
4. `findMissingAtoms()` + `addMissingAtoms()` — rebuild missing atoms
5. `addMissingHydrogens(pH=7.0)` — protonation at pH 7.0

Post-processing for Meeko compatibility:
- HIS → HID/HIE/HIP depending on protonation state (HD1, HE2)
- CYS → CYX for disulfide bridges (SSBOND or SG–SG ≤ 2.3 Å)
- Removal of terminal atoms H2/H3/OXT

### Step 2: Fixed PDB → PDBQT (`mk_prepare_receptor.py`)

Meeko converts PDB to PDBQT format for AutoDock/UniDock:
- Assignment of partial charges (Gasteiger)
- Assignment of AutoDock atom types
- Flag `--skip_gpf` (grid parameter file not needed — box derived from co-ligand)

### Step 3: Binding Box

Automatically determined from the co-crystallized ligand (HETATM) of the **original PDB** (before `removeHeterogens`):
- Ligand coordinates are extracted from the original PDB file by `scoring/binding_box.py`
- Center = centroid of heavy atoms of the ligand
- Size = extent + `box_padding` (10.0 Å by default)
- Fallback: protein center of mass + 30×30×30 Å cube

> **Order of operations**: the binding box is determined from the original PDB → then `pdbfixer_add_h.py` cleans the PDB (removing HETATM) → Meeko converts the cleaned PDB to PDBQT. Thus, box coordinates are derived before ligand/cofactor removal.

### Validation: Re-docking

Re-docking of the co-crystallized ligand is performed for each target:
- Ligand is extracted from PDB; SMILES is obtained from RCSB
- Docking uses the same parameters as the main experiment
- RMSD ≤ 2.0 Å between the docked and crystal pose = PASS
- Script: `cli/s2_calibrate/validate_redocking.py`

> An identical preparation protocol for all targets is a critical guarantee
> that differences in results between pairs are not artifacts
> of different receptor processing.

## Common Properties of All Pairs

- PDB structures with co-crystallized ligands (except 1HCL — see ¹)
- Extensive activity data in ChEMBL
- Manageable protein sizes (< 600 residues per target)

> ¹ **1HCL (CDK2)** — apo structure (no ligand). The binding box is determined from ATP-binding pocket coordinates of PDB 4GCJ (CDK2 with ligand) or via fallback (protein center of mass + 30×30×30 Å cube). Impact on docking quality is minimal: the CDK2 ATP pocket is well-defined structurally, confirmed by re-docking validation (RMSD ≤ 2.0 Å, see `cli/s2_calibrate/validate_redocking.py`).

## Key References

- Ly et al. (2013) J Clin Invest — GSK3B regulates BACE1
- Stern et al. (2022) IJMS — AChE/BACE1 dual-target
- Ivanova et al. (2020) Molecules 25:1846 — AD multi-target
- Jiang et al. (2020) Clin Transl Med — Notopterol dual BACE1/GSK3B in vivo (DOI: 10.1002/ctm2.50)
- Boltz-2 (Passaro et al. 2025) — cofactor limitations
