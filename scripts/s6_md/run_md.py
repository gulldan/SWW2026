"""Run molecular dynamics validation for top candidate molecules.

Selects the top-N molecules per generation mode (ranked by combined
Boltz-2 affinity), parameterizes each ligand with AmberTools + acpype,
builds a GROMACS topology (protein + ligand + solvent + ions), runs a
50 ns MD simulation, and analyses binding stability (RMSD, hydrogen
bonds, interaction energy).  Optionally computes MM-GBSA binding free
energy via ``gmx_MMPBSA``.

Prerequisites:
    - GROMACS >= 2024 with GPU support (CUDA)
    - AmberTools + acpype (ligand parameterization)
    - gmx_MMPBSA (optional, for MM-GBSA rescoring)

Install::

    conda install -c conda-forge -c bioconda gromacs ambertools acpype

Outputs (per molecule × target)::

    output/md/{pair}/{mode}/mol_{id}/target_{A,B}/rmsd_ligand.csv
    output/md/{pair}/{mode}/mol_{id}/target_{A,B}/hbonds.csv
    output/md/{pair}/{mode}/mol_{id}/target_{A,B}/interaction_energy.csv
    output/md/{pair}/{mode}/mol_{id}/target_{A,B}/trajectory.xtc
    output/md/{pair}/{mode}/mol_{id}/target_{A,B}/mmpbsa.csv  (optional)
    output/md/md_summary.parquet
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import click
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Stability criterion (io_contracts.md) ────────────────────────────
RMSD_THRESHOLD_A = 3.0  # Å — ligand RMSD over last 25 ns must be below this
STABLE_WINDOW_NS = 25.0  # last N ns of the trajectory used for stability check
TOTAL_SIM_NS = 50.0  # total simulation length

# ── MD parameters ────────────────────────────────────────────────────
MD_PARAMS = {
    "dt": 0.002,  # ps
    "nsteps": int(TOTAL_SIM_NS * 1e3 / 0.002),  # 50 ns
    "nstxout_compressed": 5000,  # write every 10 ps
    "coulombtype": "PME",
    "rcoulomb": 1.2,  # nm
    "rvdw": 1.2,
    "tcoupl": "V-rescale",
    "ref_t": 310,  # K (physiological)
    "pcoupl": "Parrinello-Rahman",
    "ref_p": 1.0,  # bar
    "gen_temp": 310,
    "constraints": "h-bonds",
}


# ── Utility: external command runner ─────────────────────────────────
def _run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a shell command with logging.

    Args:
        cmd: Command and arguments.
        cwd: Working directory.
        check: Raise on non-zero exit code.

    Returns:
        Completed process result.
    """
    logger.debug("$ {}", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)  # noqa: S603


# ── Ligand parameterization ──────────────────────────────────────────
def _parameterize_ligand(smiles: str, work_dir: Path) -> Path:
    """Generate GAFF2 parameters for a ligand via AmberTools + acpype.

    Args:
        smiles: SMILES string of the ligand.
        work_dir: Directory for intermediate files.

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
def _setup_system(protein_pdb: Path, acpype_dir: Path, work_dir: Path) -> Path:
    """Build a solvated, charge-neutralized GROMACS system.

    Args:
        protein_pdb: Path to the cleaned protein PDB.
        acpype_dir: Path to acpype output with ligand topology.
        work_dir: Build directory.

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
    _write_nvt_mdp(work_dir / "nvt.mdp")
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
    _write_npt_mdp(work_dir / "npt.mdp")
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

    # Production MD (50 ns)
    _write_prod_mdp(work_dir / "md.mdp")
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


def _write_nvt_mdp(path: Path) -> None:
    """Write NVT equilibration MDP (100 ps, position restraints)."""
    path.write_text(
        f"integrator = md\ndt = {MD_PARAMS['dt']}\nnsteps = 50000\n"
        f"nstxout-compressed = {MD_PARAMS['nstxout_compressed']}\n"
        f"coulombtype = {MD_PARAMS['coulombtype']}\nrcoulomb = {MD_PARAMS['rcoulomb']}\nrvdw = {MD_PARAMS['rvdw']}\n"
        f"tcoupl = {MD_PARAMS['tcoupl']}\ntc-grps = Protein Non-Protein\ntau_t = 0.1 0.1\n"
        f"ref_t = {MD_PARAMS['ref_t']} {MD_PARAMS['ref_t']}\n"
        "pcoupl = no\ngen-vel = yes\n"
        f"gen-temp = {MD_PARAMS['gen_temp']}\n"
        f"constraints = {MD_PARAMS['constraints']}\n"
        "define = -DPOSRES\npbc = xyz\n"
    )


def _write_npt_mdp(path: Path) -> None:
    """Write NPT equilibration MDP (100 ps, position restraints)."""
    path.write_text(
        f"integrator = md\ndt = {MD_PARAMS['dt']}\nnsteps = 50000\n"
        f"nstxout-compressed = {MD_PARAMS['nstxout_compressed']}\n"
        f"coulombtype = {MD_PARAMS['coulombtype']}\nrcoulomb = {MD_PARAMS['rcoulomb']}\nrvdw = {MD_PARAMS['rvdw']}\n"
        f"tcoupl = {MD_PARAMS['tcoupl']}\ntc-grps = Protein Non-Protein\ntau_t = 0.1 0.1\n"
        f"ref_t = {MD_PARAMS['ref_t']} {MD_PARAMS['ref_t']}\n"
        f"pcoupl = {MD_PARAMS['pcoupl']}\npcoupltype = isotropic\ntau_p = 2.0\n"
        f"ref_p = {MD_PARAMS['ref_p']}\ncompressibility = 4.5e-5\n"
        "gen-vel = no\ncontinuation = yes\n"
        f"constraints = {MD_PARAMS['constraints']}\n"
        "define = -DPOSRES\npbc = xyz\n"
    )


def _write_prod_mdp(path: Path) -> None:
    """Write production MD MDP (50 ns, no restraints)."""
    path.write_text(
        f"integrator = md\ndt = {MD_PARAMS['dt']}\nnsteps = {MD_PARAMS['nsteps']}\n"
        f"nstxout-compressed = {MD_PARAMS['nstxout_compressed']}\n"
        f"coulombtype = {MD_PARAMS['coulombtype']}\nrcoulomb = {MD_PARAMS['rcoulomb']}\nrvdw = {MD_PARAMS['rvdw']}\n"
        f"tcoupl = {MD_PARAMS['tcoupl']}\ntc-grps = Protein Non-Protein\ntau_t = 0.1 0.1\n"
        f"ref_t = {MD_PARAMS['ref_t']} {MD_PARAMS['ref_t']}\n"
        f"pcoupl = {MD_PARAMS['pcoupl']}\npcoupltype = isotropic\ntau_p = 2.0\n"
        f"ref_p = {MD_PARAMS['ref_p']}\ncompressibility = 4.5e-5\n"
        "gen-vel = no\ncontinuation = yes\n"
        f"constraints = {MD_PARAMS['constraints']}\n"
        "pbc = xyz\n"
    )


# ── Coordinate / topology helpers ────────────────────────────────────
def _merge_gro(protein_gro: Path, ligand_gro: Path, out_gro: Path) -> None:
    """Merge protein and ligand GRO files into a single complex.

    Args:
        protein_gro: Protein coordinate file.
        ligand_gro: Ligand coordinate file from acpype.
        out_gro: Output merged coordinate file.
    """
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
    """Add ligand ITP include and molecule entry to the GROMACS topology.

    Args:
        topol_path: Path to the system topology file.
        lig_itp_path: Path to the ligand ITP file from acpype.
    """
    content = topol_path.read_text()

    # Copy ligand ITP into topology directory
    dest_itp = topol_path.parent / lig_itp_path.name
    shutil.copy2(lig_itp_path, dest_itp)

    # Insert ligand include directive before [ system ] section
    include_line = f'#include "{lig_itp_path.name}"\n'
    if include_line not in content:
        content = content.replace("[ system ]", f"{include_line}\n[ system ]")

    # Add LIG molecule to [ molecules ]
    if "LIG" not in content:
        content = content.rstrip() + "\nLIG              1\n"

    topol_path.write_text(content)


# ── Analysis ─────────────────────────────────────────────────────────
def _analyze_trajectory(work_dir: Path, out_dir: Path) -> dict:
    """Extract RMSD, hydrogen bonds, and interaction energy from trajectory.

    Args:
        work_dir: Directory containing MD output files.
        out_dir: Directory to write analysis CSV files.

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
    rmsd_df["rmsd_A"] = rmsd_df["rmsd_nm"] * 10.0  # nm -> Å
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

    # Stability: RMSD < 3.0 Å over last 25 ns
    tail = rmsd_df[rmsd_df["time_ns"] >= (TOTAL_SIM_NS - STABLE_WINDOW_NS)]
    mean_rmsd = float(tail["rmsd_A"].mean()) if len(tail) > 0 else float("nan")
    max_rmsd = float(tail["rmsd_A"].max()) if len(tail) > 0 else float("nan")
    stable = bool(mean_rmsd < RMSD_THRESHOLD_A)
    mean_hbonds = float(hbond_df["n_hbonds"].mean()) if len(hbond_df) > 0 else float("nan")
    mean_energy = float(energy_df["energy_kJmol"].mean()) if len(energy_df) > 0 else float("nan")

    return {
        "mean_rmsd": round(mean_rmsd, 3),
        "max_rmsd": round(max_rmsd, 3),
        "stable": stable,
        "mean_hbonds": round(mean_hbonds, 2),
        "mean_energy": round(mean_energy, 2),
    }


def _parse_xvg(path: Path, columns: list[str]) -> pd.DataFrame:
    """Parse a GROMACS XVG file into a DataFrame.

    Args:
        path: Path to the XVG file.
        columns: Column names to assign.

    Returns:
        DataFrame with the requested columns.
    """
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith(("#", "@")):
            continue
        parts = line.split()
        if len(parts) >= len(columns):
            rows.append([float(x) for x in parts[: len(columns)]])
    return pd.DataFrame(rows, columns=columns)


# ── MM-GBSA (optional) ──────────────────────────────────────────────
def _run_mmpbsa(work_dir: Path, out_dir: Path) -> None:
    """Run gmx_MMPBSA for MM-GBSA binding free energy estimation.

    Args:
        work_dir: Directory containing MD output files.
        out_dir: Directory to write ``mmpbsa.csv``.
    """
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
def _select_top_molecules(pair: str, mode: str, top_n: int, seeds: list[int]) -> pd.DataFrame:
    """Select top-N molecules by combined Boltz-2 affinity across seeds.

    Args:
        pair: Pair identifier.
        mode: Generation mode.
        top_n: Number of top molecules to select.
        seeds: List of seeds to aggregate across.

    Returns:
        DataFrame with ``smiles``, ``combined_min``, ``seed`` columns,
        deduplicated by SMILES and sorted by best ``combined_min``.
    """
    from dualdocking.data.io_utils import read_df, stem_exists

    frames = []
    for seed in seeds:
        stem = PROJECT_ROOT / "output" / "results" / pair / mode / f"seed_{seed}" / "scored_joined"
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


# ── Main pipeline ────────────────────────────────────────────────────
def _run_one_molecule(
    mol_id: int,
    smiles: str,
    pair: str,
    mode: str,
    target_id: str,
    target_label: str,
    protein_pdb: Path,
    out_base: Path,
    *,
    mmpbsa: bool = False,
) -> dict | None:
    """Run the full MD pipeline for one molecule × one target.

    Args:
        mol_id: Integer index of the molecule.
        smiles: SMILES string.
        pair: Pair identifier.
        mode: Generation mode.
        target_id: Target identifier (e.g. ``CDK2``).
        target_label: ``"A"`` or ``"B"``.
        protein_pdb: Path to the cleaned protein PDB.
        out_base: Base output directory for this molecule.
        mmpbsa: Whether to run gmx_MMPBSA.

    Returns:
        Summary dict, or ``None`` on failure.
    """
    out_dir = out_base / f"target_{target_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already completed
    summary_file = out_dir / "md_result.json"
    if summary_file.exists():
        logger.info("  Skipping mol_{}/target_{} (already done)", mol_id, target_label)
        return json.loads(summary_file.read_text())

    with tempfile.TemporaryDirectory(prefix=f"md_{pair}_{mode}_mol{mol_id}_{target_label}_") as tmp:
        work = Path(tmp)
        try:
            logger.info("  Parameterizing ligand...")
            acpype_dir = _parameterize_ligand(smiles, work)

            logger.info("  Setting up GROMACS system...")
            _setup_system(protein_pdb, acpype_dir, work)

            logger.info("  Analyzing trajectory...")
            result = _analyze_trajectory(work, out_dir)

            if mmpbsa:
                logger.info("  Running MM-GBSA...")
                _run_mmpbsa(work, out_dir)

        except Exception:
            logger.exception("  MD failed for mol_{}/target_{}", mol_id, target_label)
            return None

    summary = {
        "pair": pair,
        "mode": mode,
        "mol_id": mol_id,
        "target": target_id,
        "target_label": target_label,
        "smiles": smiles,
        **result,
    }
    summary_file.write_text(json.dumps(summary, indent=2))
    return summary


@click.command()
@click.option("--pair", required=True, help="Pair ID from configs/pairs.yaml")
@click.option("--mode", default=None, help="Generation mode (default: all configured modes)")
@click.option("--top-n", default=3, type=int, show_default=True, help="Top-N molecules per mode")
@click.option("--config", default="configs/pairs.yaml")
@click.option("--experiment-config", default="configs/experiment.yaml")
@click.option("--mmpbsa", is_flag=True, default=False, help="Run MM-GBSA via gmx_MMPBSA")
@click.option("--dry-run", is_flag=True, default=False, help="Print plan without executing")
def main(
    pair: str,
    mode: str | None,
    top_n: int,
    config: str,
    experiment_config: str,
    mmpbsa: bool,
    dry_run: bool,
) -> None:
    """Run MD validation for top candidate molecules.

    Selects the top-N molecules per mode from scored results, runs 50 ns
    GROMACS MD for each molecule against both targets, and writes per-frame
    RMSD, hydrogen bond, and interaction energy data.

    Stability criterion: ligand heavy-atom RMSD < 3.0 A over the last 25 ns.

    Args:
        pair: Protein pair identifier from ``configs/pairs.yaml``.
        mode: Generation mode.  If not specified, all modes from experiment
            config are processed.
        top_n: Number of top molecules to select per mode.
        config: Path to ``pairs.yaml``.
        experiment_config: Path to ``experiment.yaml``.
        mmpbsa: Whether to compute MM-GBSA binding free energy.
        dry_run: If ``True``, print what would be run without executing.
    """
    # Check prerequisites
    for tool in ("gmx", "antechamber", "acpype", "obabel"):
        if not shutil.which(tool):
            logger.error("{} not found in PATH. Install prerequisites first.", tool)
            sys.exit(1)

    from dualdocking.config import load_experiment, load_pairs

    pairs_cfg = load_pairs(PROJECT_ROOT / config)["pairs"]
    exp_cfg = load_experiment(PROJECT_ROOT / experiment_config)

    if pair not in pairs_cfg:
        logger.error("Pair '{}' not found", pair)
        sys.exit(1)

    pair_cfg = pairs_cfg[pair]
    seeds = exp_cfg.get("seeds", [42, 137, 256, 512, 1024])
    modes = [mode] if mode else exp_cfg.get("modes", ["single_a", "single_b", "double"])

    target_a_id = pair_cfg["target_a"]["id"]
    target_b_id = pair_cfg["target_b"]["id"]

    # Resolve protein PDB paths
    targets_dir = PROJECT_ROOT / "output" / "data" / "targets" / pair
    pdb_a = targets_dir / "A" / f"{pair_cfg['target_a']['pdb']}.pdb"
    pdb_b = targets_dir / "B" / f"{pair_cfg['target_b']['pdb']}.pdb"

    for pdb, tid in [(pdb_a, target_a_id), (pdb_b, target_b_id)]:
        if not pdb.exists():
            logger.error("Protein PDB not found for {}: {}", tid, pdb)
            sys.exit(1)

    all_summaries: list[dict] = []
    md_base = PROJECT_ROOT / "output" / "md" / pair

    for m in modes:
        logger.info("=== Mode: {} ===", m)
        top_df = _select_top_molecules(pair, m, top_n, seeds)
        if top_df.empty:
            logger.warning("No scored molecules found for {}/{}", pair, m)
            continue

        logger.info("Selected {} molecules for MD", len(top_df))

        if dry_run:
            for i, row in top_df.iterrows():
                logger.info("  [DRY RUN] mol_{}: {:.80s}", i, row["smiles"])
            continue

        for i, row in top_df.iterrows():
            mol_out = md_base / m / f"mol_{i}"
            logger.info("Processing mol_{}: {:.60s}...", i, row["smiles"])

            for target_id, target_label, pdb in [
                (target_a_id, "A", pdb_a),
                (target_b_id, "B", pdb_b),
            ]:
                result = _run_one_molecule(
                    mol_id=int(i),
                    smiles=row["smiles"],
                    pair=pair,
                    mode=m,
                    target_id=target_id,
                    target_label=target_label,
                    protein_pdb=pdb,
                    out_base=mol_out,
                    mmpbsa=mmpbsa,
                )
                if result:
                    all_summaries.append(result)

    if not dry_run and all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = PROJECT_ROOT / "output" / "md" / "md_summary.parquet"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        # Append if existing summary exists
        if summary_path.exists():
            existing = pd.read_parquet(summary_path)
            summary_df = pd.concat([existing, summary_df], ignore_index=True)
            summary_df = summary_df.drop_duplicates(subset=["pair", "mode", "mol_id", "target"], keep="last")

        summary_df.to_parquet(summary_path, index=False)
        logger.info("MD summary written: {} ({} rows)", summary_path, len(summary_df))

    logger.info("MD validation complete for pair {}", pair)


if __name__ == "__main__":
    main()
