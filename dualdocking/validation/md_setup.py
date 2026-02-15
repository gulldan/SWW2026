"""GROMACS system preparation: PDB repair, ligand parameterization, topology assembly, and simulation."""

from __future__ import annotations

import shutil
from pathlib import Path

from loguru import logger

from dualdocking.validation.md_config import MDConfig, _detect_gpu_flags, _run
from dualdocking.validation.md_mdp import (
    _write_em_mdp,
    _write_ions_mdp,
    _write_npt_mdp,
    _write_nvt_mdp,
    _write_prod_mdp,
)


# ── PDB repair ───────────────────────────────────────────────────────
def _repair_pdb(pdb_path: Path, out_path: Path) -> Path:
    """Fix missing side-chain atoms in a PDB file using PDBFixer.

    Crystal structures often have disordered side chains reported as missing
    atoms (REMARK 470).  ``pdb2gmx`` rejects such structures.  PDBFixer
    rebuilds missing heavy atoms so the structure is complete.

    Returns:
        *out_path* — the repaired PDB.
    """
    try:
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer  # conda-forge: pdbfixer
    except ImportError:
        logger.warning("pdbfixer not installed — skipping PDB repair")
        shutil.copy2(pdb_path, out_path)
        return out_path

    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()

    n_missing = sum(len(v) for v in fixer.missingAtoms.values())
    if n_missing:
        logger.info("Repairing PDB: adding {} missing atoms", n_missing)
        fixer.addMissingAtoms()

    with out_path.open("w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    return out_path


# ── Ligand parameterization ──────────────────────────────────────────
def _parameterize_ligand(smiles: str, work_dir: Path, *, cmd_timeout: int = 600) -> Path:
    """Generate GAFF2 parameters for a ligand via AmberTools + acpype.

    Also generates ligand position restraints (``posre_lig.itp``) and
    patches the GROMACS ITP with a ``#ifdef POSRES_LIG`` block so that
    equilibration can restrain the ligand alongside the protein.

    Returns:
        Path to the acpype output directory containing GROMACS-ready
        topology files (``*.itp``, ``*.gro``, ``posre_lig.itp``).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        msg = f"Invalid SMILES: {smiles}"
        raise RuntimeError(msg)

    mol = Chem.AddHs(mol)
    ret = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if ret != 0:
        msg = f"3D embedding failed for: {smiles}"
        raise RuntimeError(msg)

    opt_ret = AllChem.MMFFOptimizeMolecule(mol)
    if opt_ret != 0:
        logger.warning("MMFF optimization returned {} for {:.60s}", opt_ret, smiles)

    mol2_path = work_dir / "ligand.mol2"
    Chem.MolToMolFile(mol, str(work_dir / "ligand.sdf"))

    # SDF -> MOL2 via obabel (no --gen3d: RDKit already did 3D)
    _run(
        ["obabel", str(work_dir / "ligand.sdf"), "-O", str(mol2_path)],
        cwd=work_dir, timeout=cmd_timeout,
    )

    # Antechamber: AM1-BCC charges
    _run(
        [
            "antechamber",
            "-i", str(mol2_path), "-fi", "mol2",
            "-o", str(work_dir / "ligand_bcc.mol2"), "-fo", "mol2",
            "-c", "bcc", "-at", "gaff2", "-rn", "LIG",
        ],
        cwd=work_dir, timeout=cmd_timeout,
    )

    # Parmchk2: missing parameters
    _run(
        [
            "parmchk2",
            "-i", str(work_dir / "ligand_bcc.mol2"), "-f", "mol2",
            "-o", str(work_dir / "ligand.frcmod"), "-s", "gaff2",
        ],
        cwd=work_dir, timeout=cmd_timeout,
    )

    # Acpype: AMBER -> GROMACS conversion
    _run(
        [
            "acpype",
            "-i", str(work_dir / "ligand_bcc.mol2"),
            "-b", "LIG", "-c", "user", "-a", "gaff2",
        ],
        cwd=work_dir, timeout=cmd_timeout,
    )

    acpype_dir = work_dir / "LIG.acpype"
    if not acpype_dir.exists():
        msg = f"acpype output not found: {acpype_dir}"
        raise RuntimeError(msg)

    # Generate ligand position restraints
    _run(
        [
            "gmx", "genrestr",
            "-f", str(acpype_dir / "LIG_GMX.gro"),
            "-o", str(acpype_dir / "posre_lig.itp"),
            "-fc", "1000", "1000", "1000",
        ],
        cwd=work_dir,
        input_text="0\n",  # group 0 = System (all ligand atoms)
        timeout=cmd_timeout,
    )

    # Patch LIG ITP with position restraint ifdef
    lig_itp = acpype_dir / "LIG_GMX.itp"
    itp_content = lig_itp.read_text()
    posres_block = (
        "\n; Include ligand position restraints\n"
        "#ifdef POSRES_LIG\n"
        '#include "posre_lig.itp"\n'
        "#endif\n"
    )
    if "#ifdef POSRES_LIG" not in itp_content:
        lig_itp.write_text(itp_content + posres_block)

    return acpype_dir


# ── Index file creation ──────────────────────────────────────────────
def _make_index(tpr_path: Path, work_dir: Path, *, cmd_timeout: int = 600) -> Path:
    """Create a GROMACS index file with LIG and Protein_LIG groups.

    Returns:
        Path to the created ``index.ndx`` file.
    """
    ndx_path = work_dir / "index.ndx"
    _run(
        [
            "gmx", "make_ndx",
            "-f", str(tpr_path),
            "-o", str(ndx_path),
        ],
        cwd=work_dir,
        input_text="Protein | r LIG\nq\n",
        timeout=cmd_timeout,
    )
    return ndx_path


# ── Coordinate / topology helpers ────────────────────────────────────
def _merge_gro(protein_gro: Path, ligand_gro: Path, out_gro: Path) -> None:
    """Merge protein and ligand GRO files into a single complex.

    Inserts ligand atoms **before** any crystal-water residues (HOH / SOL)
    so the coordinate order matches the topology ``[ molecules ]`` section:
    Protein -> LIG -> SOL.  This prevents the GROMACS ``genion`` error
    "The solvent group SOL is not continuous".
    """
    prot_lines = protein_gro.read_text().splitlines()
    lig_lines = ligand_gro.read_text().splitlines()

    prot_natoms = int(prot_lines[1].strip())
    lig_natoms = int(lig_lines[1].strip())
    total = prot_natoms + lig_natoms

    prot_atoms = prot_lines[2 : 2 + prot_natoms]

    # Find first crystal-water line (HOH or SOL residue name in cols 5:10)
    water_resnames = {"HOH", "SOL", "WAT", "TIP", "SPC"}
    insert_idx = len(prot_atoms)  # default: after all protein atoms
    for i, line in enumerate(prot_atoms):
        resname = line[5:10].strip()
        if resname in water_resnames:
            insert_idx = i
            break

    merged = [prot_lines[0]]
    merged.append(f" {total}")
    merged.extend(prot_atoms[:insert_idx])
    merged.extend(lig_lines[2 : 2 + lig_natoms])
    merged.extend(prot_atoms[insert_idx:])
    merged.append(prot_lines[-1])  # box vector

    out_gro.write_text("\n".join(merged) + "\n")


def _patch_topology(topol_path: Path, lig_itp_path: Path) -> None:
    """Add ligand ITP include and molecule entry to the GROMACS topology."""
    content = topol_path.read_text()

    # Copy ligand ITP into topology directory (skip if already there)
    dest_itp = topol_path.parent / lig_itp_path.name
    if lig_itp_path.resolve() != dest_itp.resolve():
        shutil.copy2(lig_itp_path, dest_itp)

    # Copy ligand position restraint file if it exists
    posre_lig_src = lig_itp_path.parent / "posre_lig.itp"
    if posre_lig_src.exists():
        posre_lig_dst = topol_path.parent / "posre_lig.itp"
        if posre_lig_src.resolve() != posre_lig_dst.resolve():
            shutil.copy2(posre_lig_src, posre_lig_dst)

    # Insert ligand include directive right after the forcefield.itp include
    # (LIG_GMX.itp may contain [ atomtypes ] which must precede [ moleculetype ])
    include_line = f'#include "{lig_itp_path.name}"\n'
    if include_line not in content:
        lines = content.split("\n")
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#include") and "forcefield.itp" in stripped:
                insert_idx = i + 1
                break
        lines.insert(insert_idx, include_line.rstrip())
        content = "\n".join(lines)

    # Add LIG molecule to [ molecules ].  Insert BEFORE the first SOL
    # entry so crystal waters and bulk solvent remain one contiguous block
    # (genion requires "SOL is continuous").
    if "\nLIG " not in content and not content.startswith("LIG "):
        lines = content.split("\n")
        in_molecules = False
        insert_pos = len(lines)  # fallback: end of file
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "[ molecules ]":
                in_molecules = True
                continue
            if in_molecules and stripped.startswith("["):
                # Reached next section — insert before it
                insert_pos = i
                break
            if in_molecules and stripped.startswith("SOL"):
                insert_pos = i
                break
        else:
            # No SOL found — append at end
            insert_pos = len(lines)
        lines.insert(insert_pos, "LIG              1")
        content = "\n".join(lines)

    topol_path.write_text(content)


# ── GROMACS system setup ─────────────────────────────────────────────
def _setup_system(
    protein_pdb: Path,
    acpype_dir: Path,
    work_dir: Path,
    config: MDConfig,
) -> Path:
    """Build a solvated, charge-neutralized GROMACS system and run MD.

    Uses amber14sb force field (Maier et al. 2015, JCTC 11:3696) with
    TIP3P water and 1.0 nm cutoffs (AMBER standard).

    Returns:
        Path to the production-ready TPR file.
    """
    gpu_flags = _detect_gpu_flags() if config.gpu_auto else []
    # Energy minimization only supports -nb gpu (no PME/bonded/update GPU)
    em_gpu_flags = ["-nb", "gpu"] if gpu_flags else []

    # Repair PDB (fix missing side-chain atoms from crystal disorder)
    repaired_pdb = work_dir / "protein_repaired.pdb"
    _repair_pdb(protein_pdb, repaired_pdb)

    # Protein topology — amber14sb (Maier et al. 2015)
    _run(
        [
            "gmx", "pdb2gmx",
            "-f", str(repaired_pdb),
            "-o", str(work_dir / "protein.gro"),
            "-p", str(work_dir / "topol.top"),
            "-ignh", "-ff", "amber14sb", "-water", "tip3p",
        ],
        cwd=work_dir, timeout=config.cmd_timeout,
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
        cwd=work_dir, timeout=config.cmd_timeout,
    )
    _run(
        [
            "gmx", "solvate",
            "-cp", str(work_dir / "box.gro"),
            "-cs", "spc216.gro",
            "-o", str(work_dir / "solv.gro"),
            "-p", str(work_dir / "topol.top"),
        ],
        cwd=work_dir, timeout=config.cmd_timeout,
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
            "-maxwarn", "0",
        ],
        cwd=work_dir, timeout=config.cmd_timeout,
    )
    _run(
        [
            "gmx", "genion",
            "-s", str(work_dir / "ions.tpr"),
            "-o", str(work_dir / "solv_ions.gro"),
            "-p", str(work_dir / "topol.top"),
            "-pname", "NA", "-nname", "CL",
            "-neutral", "-conc", "0.15",
        ],
        cwd=work_dir,
        input_text="SOL\n",
        timeout=config.cmd_timeout,
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
            "-maxwarn", "0",
        ],
        cwd=work_dir, timeout=config.cmd_timeout,
    )
    _run(
        ["gmx", "mdrun", "-deffnm", str(work_dir / "em"), "-ntmpi", "1", *em_gpu_flags],
        cwd=work_dir, timeout=config.mdrun_timeout,
    )

    # Create index file after EM
    ndx = _make_index(work_dir / "em.tpr", work_dir, cmd_timeout=config.cmd_timeout)

    # NVT equilibration
    _write_nvt_mdp(work_dir / "nvt.mdp", config)
    _run(
        [
            "gmx", "grompp",
            "-f", str(work_dir / "nvt.mdp"),
            "-c", str(work_dir / "em.gro"),
            "-r", str(work_dir / "em.gro"),
            "-p", str(work_dir / "topol.top"),
            "-n", str(ndx),
            "-o", str(work_dir / "nvt.tpr"),
            "-maxwarn", "1",
        ],
        cwd=work_dir, timeout=config.cmd_timeout,
    )
    _run(
        ["gmx", "mdrun", "-deffnm", str(work_dir / "nvt"), "-ntmpi", "1", *gpu_flags],
        cwd=work_dir, timeout=config.mdrun_timeout,
    )

    # NPT equilibration
    _write_npt_mdp(work_dir / "npt.mdp", config)
    _run(
        [
            "gmx", "grompp",
            "-f", str(work_dir / "npt.mdp"),
            "-c", str(work_dir / "nvt.gro"),
            "-r", str(work_dir / "nvt.gro"),
            "-p", str(work_dir / "topol.top"),
            "-n", str(ndx),
            "-t", str(work_dir / "nvt.cpt"),
            "-o", str(work_dir / "npt.tpr"),
            "-maxwarn", "1",
        ],
        cwd=work_dir, timeout=config.cmd_timeout,
    )
    _run(
        ["gmx", "mdrun", "-deffnm", str(work_dir / "npt"), "-ntmpi", "1", *gpu_flags],
        cwd=work_dir, timeout=config.mdrun_timeout,
    )

    # Production MD
    _write_prod_mdp(work_dir / "md.mdp", config)
    _run(
        [
            "gmx", "grompp",
            "-f", str(work_dir / "md.mdp"),
            "-c", str(work_dir / "npt.gro"),
            "-t", str(work_dir / "npt.cpt"),
            "-p", str(work_dir / "topol.top"),
            "-n", str(ndx),
            "-o", str(work_dir / "md.tpr"),
            "-maxwarn", "1",
        ],
        cwd=work_dir, timeout=config.cmd_timeout,
    )
    _run(
        ["gmx", "mdrun", "-deffnm", str(work_dir / "md"), "-ntmpi", "1", *gpu_flags],
        cwd=work_dir, timeout=config.mdrun_timeout,
    )

    return work_dir / "md.tpr"
