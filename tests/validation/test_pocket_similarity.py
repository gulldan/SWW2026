"""Tests for pocket-sequence similarity from co-crystal PDBs."""

from __future__ import annotations

from pathlib import Path

import pytest

from dualdocking.validation.pocket_similarity import compute_pocket_similarity_from_pdb_paths


def _pdb_line(
    record: str,
    serial: int,
    atom_name: str,
    resname: str,
    chain: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    return (
        f"{record:<6}{serial:>5d} {atom_name:<4}{' ':1}{resname:>3} {chain:1}{resseq:>4d}{' ':1}"
        f"   {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00 20.00          {element:>2}\n"
    )


def _write_test_pdb(path: Path, pocket_residues: list[str]) -> None:
    lines: list[str] = []
    serial = 1
    for idx, resname in enumerate(pocket_residues, start=1):
        # Place these residues close to the ligand (origin)
        lines.append(_pdb_line("ATOM", serial, "CA", resname, "A", idx, float(idx), 0.0, 0.0, "C"))
        serial += 1
    # Add one distant residue that should not be counted in the pocket
    lines.append(_pdb_line("ATOM", serial, "CA", "VAL", "A", 999, 100.0, 100.0, 100.0, "C"))
    serial += 1
    # Co-crystallized ligand near the pocket residues
    lines.append(_pdb_line("HETATM", serial, "C1", "LIG", "A", 900, 0.0, 0.0, 0.0, "C"))
    serial += 1
    lines.append(_pdb_line("HETATM", serial, "C2", "LIG", "A", 900, 0.2, 0.2, 0.0, "C"))
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


def test_pocket_similarity_identical(tmp_path: Path) -> None:
    pdb_a = tmp_path / "a.pdb"
    pdb_b = tmp_path / "b.pdb"
    _write_test_pdb(pdb_a, ["ALA", "GLY"])
    _write_test_pdb(pdb_b, ["ALA", "GLY"])

    report = compute_pocket_similarity_from_pdb_paths(pdb_a, pdb_b, pdb_id_a="AAAA", pdb_id_b="BBBB")
    assert report.identity_pct == 100.0
    assert report.pocket_len_a == 2
    assert report.pocket_len_b == 2
    assert report.matches == 2


def test_pocket_similarity_mismatch(tmp_path: Path) -> None:
    pdb_a = tmp_path / "a.pdb"
    pdb_b = tmp_path / "b.pdb"
    _write_test_pdb(pdb_a, ["ALA", "GLY"])
    _write_test_pdb(pdb_b, ["ALA", "SER"])

    report = compute_pocket_similarity_from_pdb_paths(pdb_a, pdb_b, pdb_id_a="AAAA", pdb_id_b="BBBB")
    assert report.identity_pct == 50.0
    assert report.matches == 1
    assert report.aligned_length == 2


def test_pocket_similarity_requires_ligand(tmp_path: Path) -> None:
    pdb_a = tmp_path / "a.pdb"
    pdb_b = tmp_path / "b.pdb"
    pdb_a.write_text(_pdb_line("ATOM", 1, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0, "C") + "END\n", encoding="utf-8")
    _write_test_pdb(pdb_b, ["ALA"])

    with pytest.raises(ValueError, match="No ligand detected"):
        compute_pocket_similarity_from_pdb_paths(pdb_a, pdb_b)

