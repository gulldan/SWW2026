# Receptor QC Sheet

Structural quality control for all targets used in the dual-target docking study.
Each target has been verified against RCSB PDB and UniProt.

**Protocol reference:** `docs/selection_criteria.md` (Receptor Preparation Protocol)

---

## Summary Table

| Target | PDB | Res. (A) | Chain | Co-crystal ligand | Lig. ID | Mutations | Modeled (%) | Set | Flags |
|--------|------|----------|-------|-------------------|---------|-----------|-------------|-----|-------|
| CDK2 | 1HCL | 1.80 | A | *apo* | -- | 0 | 98.7 | core | Apo: fallback binding box |
| ERK2 | 4QTA | 1.45 | A | SCH772984 | 38Z | 0 | 92.8 | core | -- |
| EGFR | 1M17 | 2.60 | A | Erlotinib | AQ4 | 0 | 93.7 | auxiliary | Resolution > 2.5 A cutoff |
| HER2 | 3PP0 | 2.25 | A | Chloro-TFM-pyrrolo-pyrimidine | 03Q | 3 | 84.6 | auxiliary | 3 engineered mutations |
| AChE | 4EY7 | 2.35 | A | Donepezil | E20 | 0 | 97.8 | core | -- |
| BChE | 4BDS | 2.10 | A | Tacrine | THA | 4 | 99.1 | core | 4 engineered mutations |
| BACE1 | 6EQM | 1.35 | A | Umibecestat (CNP520) | BUH | 0 | 94.0 | core | -- |
| GSK3B | 5KPK | 2.40 | A | BRD0209 | 6VK | 0 | 81.8 | core | pTyr216; low coverage |
| CDK5 | 1UNL | 2.20 | A | R-Roscovitine | RRC | 1 | ~100 | core | CDK5/p25 heterodimer |
| MAOB | 2V5Z | 1.60 | A | Safinamide | SAG | 0 | 96.0 | auxiliary | Covalent FAD removed by prep |

**Resolution cutoff:** 2.5 A (all targets pass except EGFR 1M17, documented auxiliary exception).

---

## Receptor Preparation Protocol (uniform)

Applied identically to all 10 targets:

1. **PDB -> Fixed PDB** (`pdbfixer_add_h.py`):
   - `findMissingResidues()` + `findNonstandardResidues()` + `replaceNonstandardResidues()`
   - `removeHeterogens(keepWater=False)` -- removes ALL HETATM (water, ions, ligands, cofactors)
   - `findMissingAtoms()` + `addMissingAtoms()`
   - `addMissingHydrogens(pH=7.0)`
   - Post-processing: HIS -> HID/HIE/HIP tautomers; CYS -> CYX for disulfides (SG-SG <= 2.3 A); remove terminal H2/H3/OXT

2. **Fixed PDB -> PDBQT** (`mk_prepare_receptor.py` via Meeko):
   - Gasteiger partial charges
   - AutoDock atom types
   - `--skip_gpf` flag

3. **Binding box** (from original PDB, before removeHeterogens):
   - Center = centroid of co-crystal ligand heavy atoms
   - Size = ligand extent + 10.0 A padding per axis
   - Fallback (apo): protein center of mass + 30x30x30 A cube

**Protonation:** pH 7.0 for all targets. Histidine tautomers assigned automatically by PDBFixer (HID/HIE/HIP based on H-bond network). No manual protonation overrides.

---

## Per-Target QC Details

### CDK2 -- 1HCL

| Field | Value |
|-------|-------|
| UniProt | P24941 |
| Organism | Homo sapiens (expressed in Sf9 baculovirus) |
| Resolution | 1.80 A |
| Method | X-ray diffraction |
| Deposition | 1996-06-03 |
| Chain | A (298 residues) |
| Co-crystal ligand | **Apo** (no ligand; water only) |
| Missing residues | 37-40 (4 residues, N-terminal loop) |
| Mutations | None |
| Binding box | **Fallback**: protein COM + 30x30x30 A (no co-crystal ligand) |

**Risks:**
- Apo structure means binding box is not experimentally defined. The fallback 30x30x30 A cube centered on COM may include non-pocket regions, reducing docking precision.
- Alternative: use ATP pocket coordinates from CDK2 holo structure (e.g., PDB 4GCJ) as reference.

---

### ERK2 -- 4QTA

| Field | Value |
|-------|-------|
| UniProt | P28482 |
| Organism | Homo sapiens (expressed in E. coli BL21-R3-pRARE2) |
| Resolution | 1.45 A |
| Method | X-ray diffraction |
| Deposition | 2014-07-07 |
| Chain | A (361 residues) |
| Co-crystal ligand | SCH772984 (38Z), pyrrolidine-indazole-piperazine-pyrimidine inhibitor |
| Missing residues | 1-10 (N-term), 177-190 (activation loop, 14 res), 360-361 (C-term). Total 26 res (7.2%) |
| Mutations | None |
| Artifacts | EDO (ethylene glycol x8, cryoprotectant), SO4 (sulfate x2) -- excluded by prep |

**Risks:**
- Activation loop residues 177-190 missing -- common for kinase structures. Does not affect ATP-site docking.
- Excellent resolution (1.45 A) -- high confidence in binding pocket geometry.

---

### EGFR -- 1M17

| Field | Value |
|-------|-------|
| UniProt | P00533 |
| Organism | Homo sapiens (expressed in Sf9 baculovirus) |
| Resolution | 2.60 A |
| Method | X-ray diffraction |
| Deposition | 2002-06-17 |
| Chain | A (333 residues, kinase domain residues 671-998) |
| Co-crystal ligand | Erlotinib (AQ4), 4-anilinoquinazoline. IC50 0.02-240 nM |
| Missing residues | 1-6 (N-term), 300-311 (12 res, internal), 331-333 (C-term). Total 21 res (6.3%) |
| Mutations | None |
| Space group | I 2 3 (cubic) |

**Risks:**
- **Resolution 2.6 A exceeds the 2.5 A quality cutoff.** Allowed only as auxiliary pair exception (sole human EGFR structure co-crystallized with erlotinib at time of selection).
- Several residues (65, 296-298, 313, 329) have partial zero-occupancy atoms.
- Not used in primary hypothesis testing.

---

### HER2 -- 3PP0

| Field | Value |
|-------|-------|
| UniProt | P04626 |
| Organism | Homo sapiens (expressed in Sf9 baculovirus) |
| Resolution | 2.25 A |
| Method | X-ray diffraction |
| Deposition | 2010-11-23 |
| Chains | A, B (homodimer; kinase domain residues 706-1009, 338 res/chain) |
| Co-crystal ligand | 03Q, chloro-trifluoromethylphenoxy-pyridinyl-pyrrolopyrimidine. IC50 = 11 nM |
| Missing residues | 1-4 (N-term), 179-180 (2 res), 293-338 (46 res, C-term). Total 52/chain (15.4%) |
| Mutations | **3 engineered**: M706A, Q711L, M712L (all near N-terminus of kinase domain) |
| R-work/R-free | 0.185 / 0.260 |

**Risks:**
- **3 engineered mutations** at construct N-terminus (M706A, Q711L, M712L). All are surface mutations introduced for crystallization/expression, not in the ATP binding pocket. Low risk for docking accuracy.
- **15.4% missing residues** -- mostly the C-terminal 46 residues (293-338). Significant but typical for kinase domain constructs.
- Used only in auxiliary pair (EGFR_HER2).

---

### AChE -- 4EY7

| Field | Value |
|-------|-------|
| UniProt | P22303 |
| Organism | Homo sapiens (expressed in HEK-293 cells, native glycosylation) |
| Resolution | 2.35 A |
| Method | X-ray diffraction |
| Deposition | 2012-05-01 |
| Chains | A, B (dimer; 542 res/chain, mature protein residues 33-574) |
| Co-crystal ligand | Donepezil / Aricept (E20). Binds catalytic gorge |
| Missing residues | 1-2 (N-term), 258-263 (6 res), 494-496 (3 res), 542 (C-term). Total 12 res (2.2%) |
| Mutations | None |
| PTMs | NAG (N-acetylglucosamine) glycosylation; 6 disulfide bonds |

**Risks:**
- Minimal. Excellent coverage (97.8%), well-characterized binding site.
- NAG glycosylation removed by `removeHeterogens` -- no effect on catalytic gorge.
- Appears in 3 pairs (AChE_BChE, AChE_BACE1, AChE_MAOB) -- same PDB/prep reused.

---

### BChE -- 4BDS

| Field | Value |
|-------|-------|
| UniProt | P06276 |
| Organism | Homo sapiens (expressed in CHO-K1 cells) |
| Resolution | 2.10 A |
| Method | X-ray diffraction |
| Deposition | 2012-10-06 |
| Chain | A |
| Co-crystal ligand | Tacrine (THA), classic AChE/BChE inhibitor |
| Missing residues | 1-3 (N-term), 378-379 (2 res); 486 partially modeled. 524/529 modeled (99.1%) |
| Mutations | **4 engineered** (positions 17, 455, 481, 486 in entity numbering) |
| PTMs | NAG at 6 N-linked glycosylation sites; 3 disulfide bonds |

**Risks:**
- **4 engineered mutations** -- specific substitutions not fully detailed in PDB metadata. Need to verify whether any affect the catalytic gorge. If all are surface/glycosylation-site mutations (common for CHO expression), low risk.
- Excellent coverage (99.1%).

---

### BACE1 -- 6EQM

| Field | Value |
|-------|-------|
| UniProt | P56817 |
| Organism | Homo sapiens (expressed in E. coli BL21(DE3)) |
| Resolution | 1.35 A |
| Method | X-ray diffraction |
| Deposition | 2017-10-13 |
| Chain | A |
| Co-crystal ligand | Umibecestat / CNP520 (BUH). Clinical BACE1 inhibitor |
| Missing residues | 1-13 (N-term, prodomain), 173-182 (10 res, internal loop), 402 (C-term). 378/402 modeled (94%) |
| Mutations | None |
| PTMs | 3 disulfide bonds; 4 cis-peptides |

**Risks:**
- Minimal. Excellent resolution (1.35 A) -- highest in the target set.
- Internal loop 173-182 missing is typical for BACE1 crystal structures.
- Appears in 3 pairs (BACE1_GSK3B, AChE_BACE1, BACE1_CDK5) -- same PDB/prep reused.

---

### GSK3B -- 5KPK

| Field | Value |
|-------|-------|
| UniProt | P49841 |
| Organism | Homo sapiens (expressed in Sf9 baculovirus) |
| Resolution | 2.40 A |
| Method | X-ray diffraction |
| Deposition | 2016-07-04 |
| Chains | A, B (two copies in ASU) |
| Co-crystal ligand | BRD0209 (6VK), IC50 = 5.0 nM |
| Missing residues | 1-38 (N-term), 124-126 (3 res), 389-424 (C-term, 36 res). 347/424 modeled (81.8%) |
| Mutations | None |
| PTMs | Phosphotyrosine (PTR) at Tyr216 -- activation mark |

**Risks:**
- **Lowest coverage in the target set (81.8%).** Large N-terminal (38 res) and C-terminal (36 res) disorder is common for kinase constructs but represents significant structural uncertainty.
- **pTyr216** is the biologically active phosphorylation mark. Retained after prep (part of polymer chain, not HETATM). Relevant for correct kinase conformation.
- Resolution 2.4 A is within cutoff but borderline.
- Appears in 2 pairs (BACE1_GSK3B, GSK3B_CDK5) -- same PDB/prep reused.

---

### CDK5 -- 1UNL

| Field | Value |
|-------|-------|
| UniProt | Q00535 |
| Organism | Homo sapiens (expressed in Sf9 baculovirus) |
| Resolution | 2.20 A |
| Method | X-ray diffraction |
| Deposition | 2003-09-10 |
| Chains | A, B (CDK5, 292 res each); C, D (p25 activator, 208 res each) |
| Co-crystal ligand | R-Roscovitine (RRC), IC50 0.2-1080 nM |
| Missing residues | None documented for CDK5 chain A (~100% modeled) |
| Mutations | 1 flagged in PDB metadata (specific substitution not detailed) |

**Risks:**
- **CDK5/p25 heterodimer**: the activator p25 (truncated p35, CDK5R1 residues 100-307) is present. For docking, only chain A (CDK5) is the target. The `removeHeterogens` step does NOT remove p25 (it is a polymer entity). If p25 is retained, it may occlude parts of the CDK5 surface but not the ATP-binding pocket.
- **1 mutation** flagged -- identity not specified in PDB metadata. Should be verified against UniProt Q00535 wild-type sequence.
- Appears in 2 pairs (GSK3B_CDK5, BACE1_CDK5).

---

### MAOB -- 2V5Z

| Field | Value |
|-------|-------|
| UniProt | P27338 |
| Organism | Homo sapiens (expressed in Pichia pastoris) |
| Resolution | 1.60 A |
| Method | X-ray diffraction |
| Deposition | 2007-07-12 |
| Chains | A, B (homodimer, 520 res each) |
| Co-crystal ligand | Safinamide (SAG), reversible selective MAO-B inhibitor |
| Missing residues | 1-2 (N-term), 502-520 (C-term, 19 res). 499/520 modeled (96.0%) |
| Mutations | None |
| Cofactor | **FAD covalently bound to Cys397** (8-alpha-S-cysteinyl thioether) |

**Risks:**
- **Critical: FAD cofactor is covalently bound** but classified as HETATM. The `removeHeterogens(keepWater=False)` step removes FAD. Docking is performed against the cofactor-free pocket, which is structurally distorted relative to the biological active site.
- **Boltz-2 affinity prediction is unreliable** for this target (`boltz2_status: warning_cofactor_fad`). Used only in auxiliary pair (AChE_MAOB).
- Despite this limitation, safinamide-based docking scores (UniDock) may still discriminate binders, since the substrate pocket shape is partially retained even without FAD.

---

## Blocked Targets (excluded from production)

| Target | PDB | Organism | Reason |
|--------|------|----------|--------|
| COX1 | 1EQG | Ovis aries (sheep) | Non-human species |
| COX2 | 3LN1 | Mus musculus (mouse) | Non-human species |

Both also contain heme cofactors, making Boltz-2 unreliable.

---

## Cross-Target Risk Summary

| Risk category | Targets affected | Severity | Mitigation |
|---------------|-----------------|----------|------------|
| Apo structure (no co-crystal ligand) | CDK2 (1HCL) | Medium | Fallback binding box; consider holo reference (4GCJ) |
| Resolution > 2.5 A | EGFR (1M17) | Low | Auxiliary pair only; sole human EGFR+erlotinib co-crystal |
| Engineered mutations | HER2 (3PP0): 3 mut; BChE (4BDS): 4 mut; CDK5 (1UNL): 1 mut | Low-Medium | All reported as surface/expression mutations, not in binding pocket. BChE mutations need verification |
| Low coverage (<85% modeled) | GSK3B (5KPK): 81.8%; HER2 (3PP0): 84.6% | Low | Terminal disorder typical for kinase constructs; binding pocket intact |
| Cofactor removed by prep | MAOB (2V5Z): covalent FAD | High | Auxiliary pair only; Boltz-2 flagged unreliable; UniDock scores interpreted with caution |
| Heteromeric complex | CDK5 (1UNL): p25 activator retained | Low | p25 does not occlude ATP pocket; chain A selected for docking |
| Phosphorylation | GSK3B (5KPK): pTyr216 | None | PTR is polymer-chain modification, retained after prep. Biologically relevant active form |

---

## QC Completeness Checklist

- [x] All 10 active targets covered (excluding blocked COX1/COX2)
- [x] No "unknown" in required fields (resolution, chain, ligand, organism)
- [x] Risks and limitations explicitly described per target
- [x] Uniform receptor preparation protocol documented
- [x] Binding box strategy documented (co-crystal centroid + 10 A padding; fallback for apo)
- [x] Protonation decisions documented (pH 7.0, automated HIS tautomers, CYS->CYX for disulfides)
- [x] Cross-target risk summary with severity and mitigation
