"""Molecular dynamics validation for top candidate molecules.

Selects the top-N molecules per generation mode (ranked by combined
Boltz-2 affinity), parameterizes each ligand with AmberTools + acpype,
builds a GROMACS topology (protein + ligand + solvent + ions), runs a
configurable-length MD simulation, and analyses binding stability (RMSD,
hydrogen bonds, interaction energy).  Optionally computes MM-GBSA binding
free energy via ``gmx_MMPBSA``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from loguru import logger

# ── Defaults ─────────────────────────────────────────────────────────
_DEFAULT_RMSD_THRESHOLD_A = 3.0  # Angstrom
_DEFAULT_STABLE_WINDOW_NS = 25.0
_DEFAULT_SIM_NS = 50.0
_DEFAULT_DT = 0.002  # ps
_DEFAULT_NSTXOUT = 5000  # write every 10 ps
_DEFAULT_REF_T = 310  # K (physiological)
_DEFAULT_REF_P = 1.0  # bar


@dataclass(frozen=True)
class MDConfig:
    """Configuration for an MD validation run."""

    pair_id: str
    top_n: int = 3
    modes: list[str] = field(default_factory=lambda: ["single_a", "single_b", "double"])
    seeds: list[int] = field(default_factory=lambda: [42, 137, 256, 512, 1024])
    simulation_time_ns: float = _DEFAULT_SIM_NS
    stable_window_ns: float = _DEFAULT_STABLE_WINDOW_NS
    rmsd_threshold_a: float = _DEFAULT_RMSD_THRESHOLD_A
    mmpbsa: bool = False
    output_dir: Path = Path("output/md")
    results_dir: Path = Path("output/results")
    targets_dir: Path = Path("output/data/targets")
    dt: float = _DEFAULT_DT
    ref_t: int = _DEFAULT_REF_T
    ref_p: float = _DEFAULT_REF_P
    nstxout_compressed: int = _DEFAULT_NSTXOUT


@dataclass(frozen=True)
class MDResult:
    """Result from a single molecule x target MD run."""

    pair: str
    mode: str
    mol_id: int
    target: str
    target_label: str
    smiles: str
    mean_rmsd: float
    max_rmsd: float
    stable: bool
    mean_hbonds: float
    mean_energy: float


# ── Utility: external command runner ─────────────────────────────────
def _run(
    cmd: list[str], *, cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run a shell command with logging."""
    logger.debug("$ {}", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)  # noqa: S603


# ── Ligand parameterization ──────────────────────────────────────────
def _parameterize_ligand(smiles: str, work_dir: Path) -> Path:
    """Generate GAFF2 parameters for a ligand via AmberTools + acpype.

    Returns:
        Path to the acpype output directory containing GROMACS-ready
        topology files (``*.itp``, ``*.gro``).

    Raises:
        RuntimeError: If parameterization fails at any stage.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        msg = f"Invalid SMILES: {smiles}"
        raise RuntimeError(msg)

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)

    mol2_path = work_dir / "ligand.mol2"
    Chem.MolToMolFile(mol, str(work_dir / "ligand.sdf"))

    # SDF -> MOL2 via obabel
    _run(["obabel", str(work_dir / "ligand.sdf"), "-O", str(mol2_path), "--gen3d"], cwd=work_dir)

    # Antechamber: AM1-BCC charges
    _run(
        [
            "antechamber",
            "-i", str(mol2_path),
            "-fi", "mol2",
            "-o", str(work_dir / "ligand_bcc.mol2"),
            "-fo", "mol2",
            "-c", "bcc",
            "-at", "gaff2",
            "-rn", "LIG",
        ],
        cwd=work_dir,
    )

    # Parmchk2: missing parameters
    _run(
        [
            "parmchk2",
            "-i", str(work_dir / "ligand_bcc.mol2"),
            "-f", "mol2",
            "-o", str(work_dir / "ligand.frcmod"),
            "-s", "gaff2",
        ],
        cwd=work_dir,
    )

    # Acpype: AMBER -> GROMACS conversion
    _run(
        [
            "acpype",
            "-i", str(work_dir / "ligand_bcc.mol2"),
            "-b", "LIG",
            "-c", "user",
            "-a", "gaff2",
        ],
        cwd=work_dir,
    )

    acpype_dir = work_dir / "LIG.acpype"
    if not acpype_dir.exists():
        msg = f"acpype output not found: {acpype_dir}"
        raise RuntimeError(msg)
    return acpype_dir


# ── GROMACS system setup ─────────────────────────────────────────────
def _setup_system(
    protein_pdb: Path,
    acpype_dir: Path,
    work_dir: Path,
    config: MDConfig,
) -> Path:
    """Build a solvated, charge-neutralized GROMACS system.

    Returns:
        Path to the production-ready TPR file.
    """
    # Protein topology
    _run(
        [
            "gmx", "pdb2gmx",
            "-f", str(protein_pdb),
            "-o", str(work_dir / "protein.gro"),
            "-p", str(work_dir / "topol.top"),
            "-ignh",
            "-ff", "amber99sb-ildn",
            "-water", "tip3p",
        ],
        cwd=work_dir,
    )

    # Merge protein + ligand coordinates
    _merge_gro(
        work_dir / "protein.gro",
        acpype_dir / "LIG_GMX.gro",
        work_dir / "complex.gro",
    )

    # Include ligand topology in topol.top
    _patch_topology(work_dir / "topol.top", acpype_dir / "LIG_GMX.itp")

    # Solvate
    _run(
        [
            "gmx", "editconf",
            "-f", str(work_dir / "complex.gro"),
            "-o", str(work_dir / "box.gro"),
            "-c", "-d", "1.2", "-bt", "dodecahedron",
        ],
        cwd=work_dir,
    )
    _run(
        [
            "gmx", "solvate",
            "-cp", str(work_dir / "box.gro"),
            "-cs", "spc216.gro",
            "-o", str(work_dir / "solv.gro"),
            "-p", str(work_dir / "topol.top"),
        ],
        cwd=work_dir,
    )

    # Add ions (neutralize + 0.15 M NaCl)
    _write_ions_mdp(work_dir / "ions.mdp")
    _run(
        [
            "gmx", "grompp",
            "-f", str(work_dir / "ions.mdp"),
            "-c", str(work_dir / "solv.gro"),
            "-p", str(work_dir / "topol.top"),
            "-o", str(work_dir / "ions.tpr"),
            "-maxwarn", "2",
        ],
        cwd=work_dir,
    )
    _run(
        [
            "gmx", "genion",
            "-s", str(work_dir / "ions.tpr"),
            "-o", str(work_dir / "solv_ions.gro"),
            "-p", str(work_dir / "topol.top"),
            "-pname", "NA",
            "-nname", "CL",
            "-neutral",
            "-conc", "0.15",
        ],
        cwd=work_dir,
    )

    # Energy minimization
    _write_em_mdp(work_dir / "em.mdp")
    _run(
        [
            "gmx", "grompp",
            "-f", str(work_dir / "em.mdp"),
            "-c", str(work_dir / "solv_ions.gro"),
            "-p", str(work_dir / "topol.top"),
            "-o", str(work_dir / "em.tpr"),
            "-maxwarn", "2",
        ],
        cwd=work_dir,
    )
    _run(["gmx", "mdrun", "-deffnm", str(work_dir / "em"), "-ntmpi", "1"], cwd=work_dir)

    # NVT equilibration (100 ps)
    _write_nvt_mdp(work_dir / "nvt.mdp", config)
    _run(
        [
            "gmx", "grompp",
            "-f", str(work_dir / "nvt.mdp"),
            "-c", str(work_dir / "em.gro"),
            "-r", str(work_dir / "em.gro"),
            "-p", str(work_dir / "topol.top"),
            "-o", str(work_dir / "nvt.tpr"),
            "-maxwarn", "2",
        ],
        cwd=work_dir,
    )
    _run(["gmx", "mdrun", "-deffnm", str(work_dir / "nvt"), "-ntmpi", "1"], cwd=work_dir)

    # NPT equilibration (100 ps)
    _write_npt_mdp(work_dir / "npt.mdp", config)
    _run(
        [
            "gmx", "grompp",
            "-f", str(work_dir / "npt.mdp"),
            "-c", str(work_dir / "nvt.gro"),
            "-r", str(work_dir / "nvt.gro"),
            "-p", str(work_dir / "topol.top"),
            "-t", str(work_dir / "nvt.cpt"),
            "-o", str(work_dir / "npt.tpr"),
            "-maxwarn", "2",
        ],
        cwd=work_dir,
    )
    _run(["gmx", "mdrun", "-deffnm", str(work_dir / "npt"), "-ntmpi", "1"], cwd=work_dir)

    # Production MD
    _write_prod_mdp(work_dir / "md.mdp", config)
    _run(
        [
            "gmx", "grompp",
            "-f", str(work_dir / "md.mdp"),
            "-c", str(work_dir / "npt.gro"),
            "-t", str(work_dir / "npt.cpt"),
            "-p", str(work_dir / "topol.top"),
            "-o", str(work_dir / "md.tpr"),
            "-maxwarn", "2",
        ],
        cwd=work_dir,
    )
    _run(["gmx", "mdrun", "-deffnm", str(work_dir / "md"), "-ntmpi", "1"], cwd=work_dir)

    return work_dir / "md.tpr"


# ── MDP writers ──────────────────────────────────────────────────────
def _write_ions_mdp(path: Path) -> None:
    """Write a minimal MDP for ion placement (grompp only)."""
    path.write_text("integrator = steep\nnsteps = 0\n")


def _write_em_mdp(path: Path) -> None:
    """Write energy minimization MDP."""
    path.write_text(
        "integrator = steep\nemtol = 1000.0\nemstep = 0.01\nnsteps = 50000\n"
        "nstlist = 10\ncoulombtype = PME\nrcoulomb = 1.2\nrvdw = 1.2\n"
        "pbc = xyz\n"
    )


def _write_equil_mdp(
    path: Path,
    config: MDConfig,
    *,
    pressure_block: str,
    velocity_block: str,
) -> None:
    """Write shared equilibration MDP content for NVT/NPT phases."""
    path.write_text(
        f"integrator = md\ndt = {config.dt}\nnsteps = 50000\n"
        f"nstxout-compressed = {config.nstxout_compressed}\n"
        "coulombtype = PME\nrcoulomb = 1.2\nrvdw = 1.2\n"
        "tcoupl = V-rescale\ntc-grps = Protein Non-Protein\ntau_t = 0.1 0.1\n"
        f"ref_t = {config.ref_t} {config.ref_t}\n"
        f"{pressure_block}"
        f"{velocity_block}"
        "constraints = h-bonds\n"
        "define = -DPOSRES\npbc = xyz\n"
    )


def _write_nvt_mdp(path: Path, config: MDConfig) -> None:
    """Write NVT equilibration MDP (100 ps, position restraints)."""
    _write_equil_mdp(
        path,
        config,
        pressure_block="pcoupl = no\n",
        velocity_block=f"gen-vel = yes\ngen-temp = {config.ref_t}\n",
    )


def _write_npt_mdp(path: Path, config: MDConfig) -> None:
    """Write NPT equilibration MDP (100 ps, position restraints)."""
    _write_equil_mdp(
        path,
        config,
        pressure_block=(
            "pcoupl = Parrinello-Rahman\n"
            "pcoupltype = isotropic\n"
            "tau_p = 2.0\n"
            f"ref_p = {config.ref_p}\n"
            "compressibility = 4.5e-5\n"
        ),
        velocity_block="gen-vel = no\ncontinuation = yes\n",
    )


def _write_prod_mdp(path: Path, config: MDConfig) -> None:
    """Write production MD MDP."""
    nsteps = int(config.simulation_time_ns * 1e3 / config.dt)
    path.write_text(
        f"integrator = md\ndt = {config.dt}\nnsteps = {nsteps}\n"
        f"nstxout-compressed = {config.nstxout_compressed}\n"
        f"coulombtype = PME\nrcoulomb = 1.2\nrvdw = 1.2\n"
        f"tcoupl = V-rescale\ntc-grps = Protein Non-Protein\ntau_t = 0.1 0.1\n"
        f"ref_t = {config.ref_t} {config.ref_t}\n"
        f"pcoupl = Parrinello-Rahman\npcoupltype = isotropic\ntau_p = 2.0\n"
        f"ref_p = {config.ref_p}\ncompressibility = 4.5e-5\n"
        "gen-vel = no\ncontinuation = yes\n"
        "constraints = h-bonds\n"
        "pbc = xyz\n"
    )


# ── Coordinate / topology helpers ────────────────────────────────────
def _merge_gro(protein_gro: Path, ligand_gro: Path, out_gro: Path) -> None:
    """Merge protein and ligand GRO files into a single complex."""
    prot_lines = protein_gro.read_text().splitlines()
    lig_lines = ligand_gro.read_text().splitlines()

    prot_natoms = int(prot_lines[1].strip())
    lig_natoms = int(lig_lines[1].strip())
    total = prot_natoms + lig_natoms

    merged = [prot_lines[0]]
    merged.append(f" {total}")
    merged.extend(prot_lines[2 : 2 + prot_natoms])
    merged.extend(lig_lines[2 : 2 + lig_natoms])
    merged.append(prot_lines[-1])  # box vector

    out_gro.write_text("\n".join(merged) + "\n")


def _patch_topology(topol_path: Path, lig_itp_path: Path) -> None:
    """Add ligand ITP include and molecule entry to the GROMACS topology."""
    content = topol_path.read_text()

    # Copy ligand ITP into topology directory (skip if already there)
    dest_itp = topol_path.parent / lig_itp_path.name
    if lig_itp_path.resolve() != dest_itp.resolve():
        shutil.copy2(lig_itp_path, dest_itp)

    # Insert ligand include directive before [ system ] section
    include_line = f'#include "{lig_itp_path.name}"\n'
    if include_line not in content:
        content = content.replace("[ system ]", f"{include_line}\n[ system ]")

    # Add LIG molecule to [ molecules ]
    if "\nLIG " not in content and not content.startswith("LIG "):
        content = content.rstrip() + "\nLIG              1\n"

    topol_path.write_text(content)


# ── Analysis ─────────────────────────────────────────────────────────
def _parse_xvg(path: Path, columns: list[str]) -> pd.DataFrame:
    """Parse a GROMACS XVG file into a DataFrame."""
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith(("#", "@")):
            continue
        parts = line.split()
        if len(parts) >= len(columns):
            rows.append([float(x) for x in parts[: len(columns)]])
    return pd.DataFrame(rows, columns=columns)


def _analyze_trajectory(work_dir: Path, out_dir: Path, config: MDConfig) -> dict:
    """Extract RMSD, hydrogen bonds, and interaction energy from trajectory.

    Returns:
        Dictionary with ``mean_rmsd``, ``max_rmsd``, ``stable``,
        ``mean_hbonds``, ``mean_energy``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ligand RMSD
    rmsd_xvg = work_dir / "rmsd_lig.xvg"
    _run(
        [
            "gmx", "rms",
            "-s", str(work_dir / "md.tpr"),
            "-f", str(work_dir / "md.xtc"),
            "-o", str(rmsd_xvg),
            "-tu", "ns",
        ],
        cwd=work_dir,
    )
    rmsd_df = _parse_xvg(rmsd_xvg, columns=["time_ns", "rmsd_nm"])
    rmsd_df["rmsd_A"] = rmsd_df["rmsd_nm"] * 10.0  # nm -> Angstrom
    rmsd_df[["time_ns", "rmsd_A"]].to_csv(out_dir / "rmsd_ligand.csv", index=False)

    # Hydrogen bonds
    hbond_xvg = work_dir / "hbond.xvg"
    _run(
        [
            "gmx", "hbond",
            "-s", str(work_dir / "md.tpr"),
            "-f", str(work_dir / "md.xtc"),
            "-num", str(hbond_xvg),
            "-tu", "ns",
        ],
        cwd=work_dir,
    )
    hbond_df = _parse_xvg(hbond_xvg, columns=["time_ns", "n_hbonds"])
    hbond_df.to_csv(out_dir / "hbonds.csv", index=False)

    # Interaction energy (short-range Coulomb + LJ)
    energy_xvg = work_dir / "interaction_energy.xvg"
    _run(
        [
            "gmx", "energy",
            "-f", str(work_dir / "md.edr"),
            "-o", str(energy_xvg),
        ],
        cwd=work_dir,
    )
    energy_df = _parse_xvg(energy_xvg, columns=["time_ns", "energy_kJmol"])
    energy_df.to_csv(out_dir / "interaction_energy.csv", index=False)

    # Copy trajectory
    xtc_src = work_dir / "md.xtc"
    if xtc_src.exists():
        shutil.copy2(xtc_src, out_dir / "trajectory.xtc")

    # Stability: RMSD < threshold over last N ns
    tail = rmsd_df[rmsd_df["time_ns"] >= (config.simulation_time_ns - config.stable_window_ns)]
    mean_rmsd = float(tail["rmsd_A"].mean()) if len(tail) > 0 else float("nan")
    max_rmsd = float(tail["rmsd_A"].max()) if len(tail) > 0 else float("nan")
    stable = bool(mean_rmsd < config.rmsd_threshold_a)
    mean_hbonds = float(hbond_df["n_hbonds"].mean()) if len(hbond_df) > 0 else float("nan")
    mean_energy = float(energy_df["energy_kJmol"].mean()) if len(energy_df) > 0 else float("nan")

    return {
        "mean_rmsd": round(mean_rmsd, 3),
        "max_rmsd": round(max_rmsd, 3),
        "stable": stable,
        "mean_hbonds": round(mean_hbonds, 2),
        "mean_energy": round(mean_energy, 2),
    }


# ── MM-GBSA (optional) ──────────────────────────────────────────────
def _run_mmpbsa(work_dir: Path, out_dir: Path) -> None:
    """Run gmx_MMPBSA for MM-GBSA binding free energy estimation."""
    if not shutil.which("gmx_MMPBSA"):
        logger.warning("gmx_MMPBSA not found, skipping MM-GBSA")
        return

    mmpbsa_in = work_dir / "mmpbsa.in"
    mmpbsa_in.write_text(
        "&general\n  startframe=2500\n  endframe=5000\n  interval=10\n/\n"
        "&gb\n  igb=5\n  saltcon=0.15\n/\n"
    )

    result = _run(
        [
            "gmx_MMPBSA",
            "-O",
            "-i", str(mmpbsa_in),
            "-cs", str(work_dir / "md.tpr"),
            "-ct", str(work_dir / "md.xtc"),
            "-cp", str(work_dir / "topol.top"),
        ],
        cwd=work_dir,
        check=False,
    )
    if result.returncode != 0:
        logger.warning("gmx_MMPBSA failed: {}", result.stderr[:500])
        return

    # Parse FINAL_RESULTS_MMPBSA.dat if available
    final = work_dir / "FINAL_RESULTS_MMPBSA.dat"
    if final.exists():
        shutil.copy2(final, out_dir / "mmpbsa.csv")


# ── Top-N molecule selection ─────────────────────────────────────────
def select_top_molecules(
    pair: str,
    mode: str,
    top_n: int,
    seeds: list[int],
    results_dir: Path,
) -> pd.DataFrame:
    """Select top-N molecules by combined Boltz-2 affinity across seeds.

    Returns:
        DataFrame with ``smiles``, ``combined_min``, ``seed`` columns,
        deduplicated by SMILES and sorted by best ``combined_min``.
    """
    from dualdocking.data.io_utils import read_df, stem_exists

    frames = []
    for seed in seeds:
        stem = results_dir / pair / mode / f"seed_{seed}" / "scored_joined"
        if not stem_exists(stem):
            continue
        df = read_df(stem)
        df["seed"] = seed
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)

    # Deduplicate by SMILES, keep best combined score
    score_col = "combined_min" if "combined_min" in all_df.columns else "affinity_a"
    best = all_df.sort_values(score_col).drop_duplicates(subset="smiles", keep="first")
    return best.head(top_n).reset_index(drop=True)


# ── Single molecule pipeline ────────────────────────────────────────
def run_single_molecule_md(
    mol_id: int,
    smiles: str,
    pair: str,
    mode: str,
    target_id: str,
    target_label: str,
    protein_pdb: Path,
    out_base: Path,
    config: MDConfig,
) -> MDResult | None:
    """Run the full MD pipeline for one molecule x one target.

    Returns:
        MDResult on success, or ``None`` on failure.
    """
    out_dir = out_base / f"target_{target_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already completed
    summary_file = out_dir / "md_result.json"
    if summary_file.exists():
        logger.info("  Skipping mol_{}/target_{} (already done)", mol_id, target_label)
        data = json.loads(summary_file.read_text())
        return MDResult(
            pair=data["pair"],
            mode=data["mode"],
            mol_id=data["mol_id"],
            target=data["target"],
            target_label=data["target_label"],
            smiles=data["smiles"],
            mean_rmsd=data["mean_rmsd"],
            max_rmsd=data["max_rmsd"],
            stable=data["stable"],
            mean_hbonds=data["mean_hbonds"],
            mean_energy=data["mean_energy"],
        )

    with tempfile.TemporaryDirectory(prefix=f"md_{pair}_{mode}_mol{mol_id}_{target_label}_") as tmp:
        work = Path(tmp)
        try:
            logger.info("  Parameterizing ligand...")
            acpype_dir = _parameterize_ligand(smiles, work)

            logger.info("  Setting up GROMACS system...")
            _setup_system(protein_pdb, acpype_dir, work, config)

            logger.info("  Analyzing trajectory...")
            analysis = _analyze_trajectory(work, out_dir, config)

            if config.mmpbsa:
                logger.info("  Running MM-GBSA...")
                _run_mmpbsa(work, out_dir)

        except Exception:
            logger.exception("  MD failed for mol_{}/target_{}", mol_id, target_label)
            return None

    result = MDResult(
        pair=pair,
        mode=mode,
        mol_id=mol_id,
        target=target_id,
        target_label=target_label,
        smiles=smiles,
        **analysis,
    )
    summary_file.write_text(json.dumps({
        "pair": result.pair,
        "mode": result.mode,
        "mol_id": result.mol_id,
        "target": result.target,
        "target_label": result.target_label,
        "smiles": result.smiles,
        "mean_rmsd": result.mean_rmsd,
        "max_rmsd": result.max_rmsd,
        "stable": result.stable,
        "mean_hbonds": result.mean_hbonds,
        "mean_energy": result.mean_energy,
    }, indent=2))
    return result


# ── GROMACS command generation (for testing) ─────────────────────────
def build_gromacs_commands() -> list[list[str]]:
    """Build the sequence of GROMACS commands for the MD pipeline.

    This is a pure function suitable for unit testing -- it does not
    execute any commands. The returned list contains the ``gmx`` commands
    in execution order.

    Returns:
        List of command argument lists.
    """
    # Placeholders for paths -- these get filled in at runtime
    w = Path("/work")

    commands: list[list[str]] = []

    # pdb2gmx
    commands.append([
        "gmx", "pdb2gmx",
        "-f", str(w / "protein.pdb"),
        "-o", str(w / "protein.gro"),
        "-p", str(w / "topol.top"),
        "-ignh", "-ff", "amber99sb-ildn", "-water", "tip3p",
    ])

    # editconf
    commands.append([
        "gmx", "editconf",
        "-f", str(w / "complex.gro"),
        "-o", str(w / "box.gro"),
        "-c", "-d", "1.2", "-bt", "dodecahedron",
    ])

    # solvate
    commands.append([
        "gmx", "solvate",
        "-cp", str(w / "box.gro"),
        "-cs", "spc216.gro",
        "-o", str(w / "solv.gro"),
        "-p", str(w / "topol.top"),
    ])

    # grompp for ions
    commands.append([
        "gmx", "grompp",
        "-f", str(w / "ions.mdp"),
        "-c", str(w / "solv.gro"),
        "-p", str(w / "topol.top"),
        "-o", str(w / "ions.tpr"),
        "-maxwarn", "2",
    ])

    # genion
    commands.append([
        "gmx", "genion",
        "-s", str(w / "ions.tpr"),
        "-o", str(w / "solv_ions.gro"),
        "-p", str(w / "topol.top"),
        "-pname", "NA", "-nname", "CL",
        "-neutral", "-conc", "0.15",
    ])

    # grompp for EM
    commands.append([
        "gmx", "grompp",
        "-f", str(w / "em.mdp"),
        "-c", str(w / "solv_ions.gro"),
        "-p", str(w / "topol.top"),
        "-o", str(w / "em.tpr"),
        "-maxwarn", "2",
    ])

    # mdrun EM
    commands.append(["gmx", "mdrun", "-deffnm", str(w / "em"), "-ntmpi", "1"])

    # grompp for NVT
    commands.append([
        "gmx", "grompp",
        "-f", str(w / "nvt.mdp"),
        "-c", str(w / "em.gro"),
        "-r", str(w / "em.gro"),
        "-p", str(w / "topol.top"),
        "-o", str(w / "nvt.tpr"),
        "-maxwarn", "2",
    ])

    # mdrun NVT
    commands.append(["gmx", "mdrun", "-deffnm", str(w / "nvt"), "-ntmpi", "1"])

    # grompp for NPT
    commands.append([
        "gmx", "grompp",
        "-f", str(w / "npt.mdp"),
        "-c", str(w / "nvt.gro"),
        "-r", str(w / "nvt.gro"),
        "-p", str(w / "topol.top"),
        "-t", str(w / "nvt.cpt"),
        "-o", str(w / "npt.tpr"),
        "-maxwarn", "2",
    ])

    # mdrun NPT
    commands.append(["gmx", "mdrun", "-deffnm", str(w / "npt"), "-ntmpi", "1"])

    # grompp for production
    commands.append([
        "gmx", "grompp",
        "-f", str(w / "md.mdp"),
        "-c", str(w / "npt.gro"),
        "-t", str(w / "npt.cpt"),
        "-p", str(w / "topol.top"),
        "-o", str(w / "md.tpr"),
        "-maxwarn", "2",
    ])

    # mdrun production (nsteps embedded in MDP, not in command line)
    commands.append(["gmx", "mdrun", "-deffnm", str(w / "md"), "-ntmpi", "1"])

    return commands


# ── Main entry point ─────────────────────────────────────────────────
def run_md_validation(
    config: MDConfig,
    pair_cfg: dict,
) -> list[MDResult]:
    """Run MD validation for a protein pair.

    Args:
        config: MD configuration.
        pair_cfg: Pair configuration dict from ``pairs.yaml`` (contains
            ``target_a`` and ``target_b`` with ``id`` and ``pdb`` fields).

    Returns:
        List of MDResult for each successful molecule x target run.
    """
    target_a_id = pair_cfg["target_a"]["id"]
    target_b_id = pair_cfg["target_b"]["id"]

    targets_dir = config.targets_dir / config.pair_id
    pdb_a = targets_dir / "A" / f"{pair_cfg['target_a']['pdb']}.pdb"
    pdb_b = targets_dir / "B" / f"{pair_cfg['target_b']['pdb']}.pdb"

    for pdb, tid in [(pdb_a, target_a_id), (pdb_b, target_b_id)]:
        if not pdb.exists():
            msg = f"Protein PDB not found for {tid}: {pdb}"
            raise FileNotFoundError(msg)

    all_results: list[MDResult] = []
    md_base = config.output_dir / config.pair_id

    for m in config.modes:
        logger.info("=== Mode: {} ===", m)
        top_df = select_top_molecules(
            config.pair_id, m, config.top_n, config.seeds, config.results_dir
        )
        if top_df.empty:
            logger.warning("No scored molecules found for {}/{}", config.pair_id, m)
            continue

        logger.info("Selected {} molecules for MD", len(top_df))

        for idx, row in top_df.iterrows():
            mol_idx = int(str(idx))
            mol_out = md_base / m / f"mol_{mol_idx}"
            logger.info("Processing mol_{}: {:.60s}...", mol_idx, row["smiles"])

            for target_id, target_label, pdb in [
                (target_a_id, "A", pdb_a),
                (target_b_id, "B", pdb_b),
            ]:
                result = run_single_molecule_md(
                    mol_id=mol_idx,
                    smiles=row["smiles"],
                    pair=config.pair_id,
                    mode=m,
                    target_id=target_id,
                    target_label=target_label,
                    protein_pdb=pdb,
                    out_base=mol_out,
                    config=config,
                )
                if result:
                    all_results.append(result)

    return all_results


def write_summary(results: list[MDResult], summary_path: Path) -> None:
    """Write or append MD results to a parquet summary file.

    Args:
        results: List of MDResult from MD runs.
        summary_path: Path to the summary parquet file.
    """
    if not results:
        return

    summary_df = pd.DataFrame([
        {
            "pair": r.pair,
            "mode": r.mode,
            "mol_id": r.mol_id,
            "target": r.target,
            "target_label": r.target_label,
            "smiles": r.smiles,
            "mean_rmsd": r.mean_rmsd,
            "max_rmsd": r.max_rmsd,
            "stable": r.stable,
            "mean_hbonds": r.mean_hbonds,
            "mean_energy": r.mean_energy,
        }
        for r in results
    ])

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if summary_path.exists():
        existing = pd.read_parquet(summary_path)
        summary_df = pd.concat([existing, summary_df], ignore_index=True)
        summary_df = summary_df.drop_duplicates(
            subset=["pair", "mode", "mol_id", "target"], keep="last"
        )

    summary_df.to_parquet(summary_path, index=False)
    logger.info("MD summary written: {} ({} rows)", summary_path, len(summary_df))
