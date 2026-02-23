# MD Validation Protocol

Post-generation molecular dynamics validation for top candidate molecules.
Not part of the primary endpoint — used to strengthen Q1 publication
credibility by demonstrating binding pose stability under explicit solvent.

---

## Software & Force Field

| Component | Choice | Reference |
|-----------|--------|-----------|
| MD engine | GROMACS 2026.0 | [Abraham et al. 2015, SoftwareX 1-2:19-25](https://doi.org/10.1016/j.softx.2015.06.001) |
| Protein force field | AMBER ff14SB | [Maier et al. 2015, JCTC 11:3696-3713](https://doi.org/10.1021/acs.jctc.5b00255) |
| Ligand force field | GAFF2 + AM1-BCC charges | AmberTools (antechamber + acpype) |
| Water model | TIP3P | [Jorgensen et al. 1983, JCP 79:926-935](https://doi.org/10.1063/1.445869) |

**Note:** ff14SB was parameterized and validated with TIP3P; using a different
water model would introduce a force-field/water mismatch.

---

## System Preparation

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Box shape | Rhombic dodecahedron | ~29% fewer solvent molecules vs cubic at same image distance |
| Buffer distance | 1.2 nm | Within standard range (1.0-1.5 nm) |
| Ionic strength | 0.15 M NaCl | Physiological; charge-neutralized with Na+/Cl- counterions |
| Energy minimization | Steepest descent, 50,000 steps, emtol = 1000 kJ/mol/nm | Standard protocol to resolve steric clashes |

---

## Equilibration

### NVT (constant volume, temperature coupling)

| Parameter | Value |
|-----------|-------|
| Duration | 200 ps (configurable via `equil_time_ps`) |
| Thermostat | V-rescale, tau_t = 0.1 ps |
| Temperature | 310 K (physiological, 37 C) |
| Position restraints | `-DPOSRES -DPOSRES_LIG` (protein + ligand) |
| Velocity generation | gen-vel = yes |
| Coupling groups | `Protein_LIG` / `Water_and_ions` |

### NPT (constant pressure, temperature coupling)

| Parameter | Value |
|-----------|-------|
| Duration | 200 ps (configurable via `equil_time_ps`) |
| Barostat | **C-rescale**, tau_p = 1.0 ps ([Bernetti & Bussi 2020, JCP 153:114107](https://doi.org/10.1063/5.0020514)) |
| Thermostat | V-rescale, tau_t = 0.1 ps |
| Pressure | 1.0 bar, isotropic coupling |
| Position restraints | `-DPOSRES -DPOSRES_LIG` |
| Continuation | yes (from NVT) |

**Why C-rescale over Berendsen:** Berendsen does not sample the correct NPT
ensemble (wrong volume fluctuations). C-rescale retains Berendsen's fast
convergence while producing correct isobaric-isothermal statistics. This is
the recommended equilibration barostat in modern GROMACS.

---

## Production

| Parameter | Value |
|-----------|-------|
| Duration | 50 ns (configurable via `simulation_time_ns`) |
| Timestep | 2 fs (dt = 0.002 ps) |
| Output frequency | Every 10 ps (nstxout_compressed = 5000) |
| Barostat | **Parrinello-Rahman**, tau_p = 2.0 ps |
| Thermostat | V-rescale, tau_t = 1.0 ps |
| Coupling groups | `Protein_LIG` / `Water_and_ions` |
| Electrostatics | PME (coulombtype = PME, fourierspacing = 0.12, pme_order = 4) |
| Real-space cutoffs | rcoulomb = rvdw = 1.0 nm (AMBER-compatible in GROMACS) |
| VdW modifier | Potential-shift |
| Dispersion correction | EnerPres (energy + pressure LR correction) |
| Constraints | h-bonds (LINCS, lincs_order = 4) |
| Boundary conditions | pbc = xyz |
| Position restraints | None |
| Energy groups | None (incompatible with GPU NB kernels) |

**Note on cutoffs:** The native AMBER convention is 0.8 nm (8 A).
The 1.0 nm cutoff used here follows the widely adopted GROMACS convention
for AMBER force fields and is conservative (more direct-space interactions
before PME handoff). Both values are valid; 1.0 nm is standard in published
GROMACS + AMBER protocols.

**Why Parrinello-Rahman for production only:** Its second-order dynamics
cause large box oscillations when the system is far from equilibrium.
C-rescale handles equilibration; Parrinello-Rahman generates the correct
NPT ensemble for data collection.

---

## Analyses

### Backbone RMSD

- `gmx rms` (stdin: `Backbone/Backbone`), output in Angstrom (nm x 10)
- Time unit: ns (`-tu ns`)

### Ligand RMSD

- `gmx rms` (stdin: `LIG/LIG`), output in Angstrom
- **Stability criterion:** mean ligand RMSD < 3.0 A over the last 25 ns of
  production (configurable via `rmsd_threshold_a` and `stable_window_ns`)

### Per-residue RMSF

- `gmx rmsf -res` (stdin: `Backbone`), output in Angstrom
- Binding pocket proxy: mean RMSF across all backbone residues

### Radius of Gyration

- `gmx gyrate` (stdin: `Protein`), output in Angstrom
- Reports mean +/- std over trajectory; monitors protein compactness

### Hydrogen Bonds

- `gmx hbond-legacy` (stdin: `Protein/LIG`), with fallback to `gmx hbond`
  for GROMACS versions before 2024
- Reports mean protein-ligand H-bond count over trajectory

### SASA (Solvent-Accessible Surface Area)

- `gmx sasa` (stdin: `Protein`), time unit ns (`-tu ns`)
- Output converted from nm^2 to A^2 (x100)
- Reports mean +/- std; written to `sasa.csv` (columns: `time_ns`, `sasa_A2`)

### PCA (Principal Component Analysis)

- `gmx covar` (stdin: `Backbone/Backbone`)
- Outputs: covariance matrix, eigenvectors, eigenvalues
- Eigenvalue spectrum written to `eigenvalues.csv` (columns: `mode`, `eigenvalue`)
- Reports PC1 and PC2 explained variance (% of total eigenvalue sum)
- Useful for identifying dominant collective motions of the protein backbone

### Convergence Assessment

- Two-sample Kolmogorov-Smirnov test on first vs second half of backbone RMSD
- **Converged** when: p > 0.05 (alpha) **AND** |mean_second_half - mean_first_half| < 0.5 A
- Parameters configurable via `convergence_ks_alpha`

### MM-GBSA (Optional)

- Requires `gmx_MMPBSA` in PATH (bundled in Docker image)
- GB model: igb=5 (OBC-II), saltcon = 0.15 M
- Frame selection: last 20% of trajectory, every 10th frame (interval=10)
- Reports binding Delta-G in kcal/mol

---

## GROMACS Pipeline (24 commands)

| Phase | Count | Commands |
|-------|-------|----------|
| Setup | 14 | pdb2gmx -> editconf -> solvate -> grompp(ions) -> genion -> grompp(EM) -> mdrun(EM) -> make_ndx -> grompp(NVT) -> mdrun(NVT) -> grompp(NPT) -> mdrun(NPT) -> grompp(prod) -> mdrun(prod) |
| Preprocessing | 3 | trjconv (`-pbc whole`) -> trjconv (`-center -pbc mol`) -> trjconv (`-fit rot+trans`) |
| Analysis | 7 | gmx rms (backbone) -> gmx rms (ligand) -> gmx rmsf -> gmx gyrate -> gmx hbond-legacy -> gmx sasa -> gmx covar |

All analysis commands use the index file (`-n index.ndx`) for group selection.

---

## Literature Comparison

| Parameter | Literature Standard | This Pipeline | Match |
|-----------|-------------------|---------------|-------|
| **Software** | | | |
| MD engine | GROMACS 2020-2024 | GROMACS 2026.0 | Yes |
| Protein FF | AMBER ff14SB or CHARMM36m | AMBER ff14SB | Yes |
| Ligand FF | GAFF/GAFF2 + AM1-BCC | GAFF2 + AM1-BCC | Yes |
| Water | TIP3P or SPC/E | TIP3P | Yes |
| **System prep** | | | |
| Box | Cubic or dodecahedron | Dodecahedron | Yes |
| Buffer | 1.0-1.5 nm | 1.2 nm | Yes |
| Ions | 0.15 M NaCl, neutralized | 0.15 M NaCl, neutralized | Yes |
| EM | Steepest descent | Steepest descent | Yes |
| **Equilibration** | | | |
| NVT | 100 ps, V-rescale | 200 ps, V-rescale | Exceeds |
| NPT barostat | Berendsen (traditional) | C-rescale (modern) | Exceeds |
| Temperature | 300 K or 310 K | 310 K | Yes |
| Pressure | 1 bar | 1 bar | Yes |
| Position restraints | Protein + ligand | Protein + ligand | Yes |
| **Production** | | | |
| Barostat | Parrinello-Rahman | Parrinello-Rahman | Yes |
| Timestep | 2 fs | 2 fs | Yes |
| Cutoffs | 1.0-1.2 nm | 1.0 nm | Yes |
| Electrostatics | PME | PME | Yes |
| Constraints | h-bonds (LINCS) | h-bonds (LINCS) | Yes |
| Dispersion corr. | EnerPres | EnerPres | Yes |
| Duration | 10-100 ns (typically 50) | 50 ns (default) | Yes |
| **Analyses** | | | |
| Backbone RMSD | Required | Yes | Yes |
| Ligand RMSD | Required | Yes | Yes |
| RMSF | Required | Yes | Yes |
| Radius of gyration | Common | Yes | Yes |
| H-bonds | Common | Yes | Yes |
| SASA | Sometimes reported | Yes | Yes |
| PCA | Sometimes (long MD) | Yes | Yes |
| Convergence test | Sometimes (KS test) | Yes (KS test) | Yes |
| MM-GBSA | Common | Yes (optional) | Yes |

---

## Output Files

Per molecule x target in `output/md/{pair}/{mode}/mol_{id}/target_{A,B}/`:

| File | Columns / Content |
|------|-------------------|
| `rmsd_backbone.csv` | time_ns, rmsd_A |
| `rmsd_ligand.csv` | time_ns, rmsd_A |
| `rmsf.csv` | residue, rmsf_A |
| `radius_gyration.csv` | time_ps, rg_A |
| `hbonds.csv` | time_ns, n_hbonds |
| `sasa.csv` | time_ns, sasa_A2 |
| `eigenvalues.csv` | mode, eigenvalue |
| `trajectory.xtc` | PBC-corrected, centered, fitted trajectory |
| `md_result.json` | All summary metrics (25 fields) |
| `mmpbsa.csv` | MM-GBSA output (only with `--mmpbsa`) |

Global summary: `output/md/md_summary.parquet`.

---

## Limitations

1. **Single replica per molecule x target** — no ensemble averaging; variability
   from initial conditions is not captured.
2. **50 ns may be insufficient** for slow conformational transitions, unbinding
   events, or allosteric motions.
3. **GAFF2 parameterization** may produce suboptimal parameters for unusual
   functional groups in de novo generated molecules.
4. **MM-GBSA accuracy** is system-dependent; igb=5 (OBC-II) may not be optimal
   for all protein-ligand systems.
5. **PCA on 50 ns** may capture local backbone fluctuations rather than
   functionally relevant global conformational changes.
6. **RMSF binding pocket proxy** uses mean over all backbone residues, not a
   curated binding-site residue list.

---

## References

- Maier, J. A. et al. ff14SB: Improving the Accuracy of Protein Side Chain
  and Backbone Parameters from ff99SB. *J. Chem. Theory Comput.* **2015**,
  11, 3696-3713. [DOI:10.1021/acs.jctc.5b00255](https://doi.org/10.1021/acs.jctc.5b00255)
- Bernetti, M.; Bussi, G. Pressure control using stochastic cell rescaling.
  *J. Chem. Phys.* **2020**, 153, 114107.
  [DOI:10.1063/5.0020514](https://doi.org/10.1063/5.0020514)
- Jorgensen, W. L. et al. Comparison of simple potential functions for
  simulating liquid water. *J. Chem. Phys.* **1983**, 79, 926-935.
  [DOI:10.1063/1.445869](https://doi.org/10.1063/1.445869)
- Hess, B. et al. LINCS: A linear constraint solver for molecular simulations.
  *J. Comput. Chem.* **1997**, 18, 1463-1472.
  [DOI:10.1002/(SICI)1096-987X(199709)18:12<1463::AID-JCC4>3.0.CO;2-H](https://doi.org/10.1002/(SICI)1096-987X(199709)18:12%3C1463::AID-JCC4%3E3.0.CO;2-H)
- Abraham, M. J. et al. GROMACS: High performance molecular simulations
  through multi-level parallelism from laptops to supercomputers.
  *SoftwareX* **2015**, 1-2, 19-25.
  [DOI:10.1016/j.softx.2015.06.001](https://doi.org/10.1016/j.softx.2015.06.001)
