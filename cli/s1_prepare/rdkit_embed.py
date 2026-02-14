"""Generate 3D coordinates from SMILES using RDKit and write SDF.

Designed to run inside the FREED++ docker image (RDKit preinstalled).
Reads a SMILES string, computes a 3D conformation via ETKDGv3, optimises
with UFF, and writes the result as an SDF file.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    """Parse arguments, embed SMILES in 3D, and write SDF.

    Returns:
        Exit code: 0 on success, 2 on RDKit import failure, 3 on SMILES
        parse failure, 4 on 3D embedding failure.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--sdf", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=200)
    args = parser.parse_args()

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except Exception as exc:
        sys.stderr.write(f"RDKit import failed: {exc}\n")
        return 2

    mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        sys.stderr.write("RDKit failed to parse SMILES\n")
        return 3

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = args.seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        sys.stderr.write("RDKit 3D embedding failed\n")
        return 4

    AllChem.UFFOptimizeMolecule(mol, maxIters=args.max_iters)

    writer = Chem.SDWriter(args.sdf)
    writer.write(mol)
    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
