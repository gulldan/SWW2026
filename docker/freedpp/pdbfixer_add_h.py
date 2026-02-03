"""Add missing atoms and hydrogens to a PDB using PDBFixer."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path


def _fix_pdb_for_meeko(pdb_text: str) -> str:
    """Normalize a PDB to match Meeko residue templates.

    - Rename HIS to HID/HIE/HIP based on protonation state.
    - Drop terminal-only atoms (H2/H3/OXT) that aren't in standard templates.
    """
    lines = pdb_text.splitlines(keepends=True)

    # First pass: collect which hydrogens each HIS residue has
    his_protons: dict[tuple[str, int], set[str]] = {}  # (chain, resnum) -> {atom_names}
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resname = line[17:20].strip()
        if resname != "HIS":
            continue
        chain = line[21:22]
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        atom_name = line[12:16].strip()
        key = (chain, resnum)
        if key not in his_protons:
            his_protons[key] = set()
        his_protons[key].add(atom_name)

    # Determine correct name for each HIS
    his_names: dict[tuple[str, int], str] = {}
    for key, atoms in his_protons.items():
        has_hd1 = "HD1" in atoms
        has_he2 = "HE2" in atoms
        if has_hd1 and has_he2:
            his_names[key] = "HIP"
        elif has_hd1:
            his_names[key] = "HID"
        elif has_he2:
            his_names[key] = "HIE"
        else:
            his_names[key] = "HID"  # default fallback

    # Second pass: rename residues
    new_lines = []
    for raw_line in lines:
        line = raw_line
        if line.startswith(("ATOM", "HETATM")):
            atom_name = line[12:16].strip()
            if atom_name in {"H2", "H3", "OXT"}:
                continue
            if line[17:20].strip() == "HIS":
                chain = line[21:22]
                try:
                    resnum = int(line[22:26])
                    key = (chain, resnum)
                    if key in his_names:
                        new_name = his_names[key]
                        line = line[:17] + new_name + line[20:]
                except ValueError:
                    pass
        new_lines.append(line)

    return "".join(new_lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True, help="Input PDB file")
    parser.add_argument("--out", required=True, help="Output PDB file with hydrogens")
    parser.add_argument("--ph", type=float, default=7.0, help="pH for protonation (default: 7.0)")
    args = parser.parse_args()

    try:
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise SystemExit(f"PDBFixer/OpenMM import failed: {exc}") from exc

    pdb_path = Path(args.pdb)
    out_path = Path(args.out)
    if not pdb_path.exists():
        raise SystemExit(f"Input PDB not found: {pdb_path}")

    try:
        fixer = PDBFixer(filename=str(pdb_path))
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=args.ph)
    except Exception as exc:  # pragma: no cover - runtime dependency
        detail = traceback.format_exc()
        raise SystemExit(f"PDBFixer failed: {exc}\n{detail}") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to string first, then rename HIS for Meeko compatibility
    from io import StringIO

    buffer = StringIO()
    PDBFile.writeFile(fixer.topology, fixer.positions, buffer, keepIds=True)
    pdb_text = _fix_pdb_for_meeko(buffer.getvalue())

    with out_path.open("w") as f:
        f.write(pdb_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
