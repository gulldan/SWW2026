"""Shared molecule preparation: SMILES validation, 3D embedding, neutralization, PDBQT.

Standalone module with **zero imports from dualdocking** — safe to COPY into
the FREED++ Docker image (``/opt/mol_prep.py``).

Dependencies: ``rdkit``, ``meeko`` (both available in Docker).
"""

from __future__ import annotations

import logging
from contextlib import suppress
from pathlib import Path

from rdkit.Chem import Mol

logger = logging.getLogger(__name__)


def parse_smiles(smiles: str) -> Mol | None:
    """Parse a SMILES string into an RDKit ``Mol``.

    Args:
        smiles: SMILES string.

    Returns:
        RDKit ``Mol`` object, or ``None`` on parse failure.
    """
    from rdkit import Chem

    return Chem.MolFromSmiles(smiles)


def check_ligand_size(
    mol: Mol,
    *,
    max_heavy: int = 70,
    max_atoms: int | None = None,
    max_rotatable: int | None = 12,
) -> bool:
    """Check whether *mol* passes ligand size gates.

    Args:
        mol: RDKit ``Mol`` (should **not** have explicit H added).
        max_heavy: ``GetNumHeavyAtoms`` threshold (docking default 70).
        max_atoms: ``GetNumAtoms`` threshold (Boltz-2 default 128).
            ``None`` to skip.
        max_rotatable: ``NumRotatableBonds`` threshold (docking default 12).
            ``None`` to skip.

    Returns:
        ``True`` if all enabled checks pass.
    """
    if mol.GetNumHeavyAtoms() > max_heavy:
        return False
    if max_atoms is not None and mol.GetNumAtoms() > max_atoms:
        return False
    if max_rotatable is not None:
        from rdkit.Chem import Descriptors

        if Descriptors.NumRotatableBonds(mol) > max_rotatable:
            return False
    return True


def neutralize_smiles(smiles: str) -> str:
    """Remove formal charges via ``rdMolStandardize.Uncharger``.

    Boltz-2 v2.2.1 crashes on charged species (``[NH2+]``, ``[O-]``).
    Strips protonation states while preserving molecular identity.

    Args:
        smiles: Input SMILES (potentially charged).

    Returns:
        Neutralized SMILES, or the original string if neutralization
        fails or RDKit is unavailable.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError:
        return smiles

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles

    try:
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
    except Exception:
        logger.debug("Neutralization failed for SMILES: %s", smiles)
        return smiles

    result = Chem.MolToSmiles(mol)
    return result or smiles


def embed_mol_3d(
    mol: Mol,
    *,
    seed: int = 0,
    max_uff_iters: int = 200,
) -> Mol | None:
    """Generate a 3D conformer: ``AddHs`` + ETKDGv3 + UFF optimise.

    Args:
        mol: RDKit ``Mol`` (implicit H; explicit H will be added).
        seed: Random seed for the embedding algorithm.
        max_uff_iters: Maximum UFF optimisation iterations.

    Returns:
        Mol with explicit H and a 3D conformer, or ``None`` on failure.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed) if int(seed) >= 0 else 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None

    with suppress(Exception):
        AllChem.UFFOptimizeMolecule(mol, maxIters=max_uff_iters)

    return mol


def mol_to_pdbqt(mol: Mol, out_path: str | Path) -> bool:
    """Write a 3D ``Mol`` to PDBQT via Meeko.

    Args:
        mol: RDKit ``Mol`` with a 3D conformer.
        out_path: Destination PDBQT file path.

    Returns:
        ``True`` on success.
    """
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    try:
        setups = MoleculePreparation().prepare(mol)
    except (ValueError, RuntimeError):
        return False
    if not setups:
        return False
    pdbqt_text, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        logger.debug("Meeko write failed: %s", err)
        return False

    with Path(out_path).open("w", encoding="utf-8") as f:
        f.write(pdbqt_text)
    return True


def prepare_ligand_pdbqt(
    smiles: str,
    out_path: str | Path,
    *,
    seed: int = 0,
    max_heavy: int = 70,
    max_atoms: int | None = None,
    max_rotatable: int | None = 12,
) -> bool:
    """Full pipeline: parse SMILES -> size check -> embed 3D -> write PDBQT.

    Args:
        smiles: Input SMILES string.
        out_path: Destination PDBQT file path.
        seed: Random seed for 3D embedding.
        max_heavy: Heavy-atom count threshold.
        max_atoms: Total-atom count threshold (``None`` to skip).
        max_rotatable: Rotatable-bond threshold (``None`` to skip).

    Returns:
        ``True`` on success, ``False`` on any preparation failure.
    """
    mol = parse_smiles(smiles)
    if mol is None:
        return False

    if not check_ligand_size(
        mol,
        max_heavy=max_heavy,
        max_atoms=max_atoms,
        max_rotatable=max_rotatable,
    ):
        return False

    mol_3d = embed_mol_3d(mol, seed=seed)
    if mol_3d is None:
        return False

    return mol_to_pdbqt(mol_3d, out_path)
