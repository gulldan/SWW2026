"""Tests for binding_box module."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import numpy as np

from dualdocking.binding_box import (
    compute_box,
    extract_binding_box,
    extract_ligand_coords,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_pdb(tmp_path: Path, content: str, name: str = "test.pdb") -> Path:
    pdb_path = tmp_path / name
    pdb_path.write_text(textwrap.dedent(content))
    return pdb_path


# --- extract_ligand_coords ---


def test_extract_ligand_coords_with_ligand(tmp_path: Path) -> None:
    """PDB с лигандом → корректные координаты."""
    pdb = _write_pdb(
        tmp_path,
        """\
        ATOM      1  CA  ALA A   1      10.000  20.000  30.000  1.00 10.00           C
        HETATM  100  C1  LIG A 200      1.000   2.000   3.000  1.00  5.00           C
        HETATM  101  C2  LIG A 200      4.000   5.000   6.000  1.00  5.00           C
        HETATM  102  H1  LIG A 200      1.500   2.500   3.500  1.00  5.00           H
        END
        """,
    )
    coords = extract_ligand_coords(pdb)
    assert coords is not None
    assert coords.shape == (2, 3)  # 2 heavy atoms (H excluded)
    np.testing.assert_allclose(coords[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(coords[1], [4.0, 5.0, 6.0])


def test_extract_ligand_coords_water_only(tmp_path: Path) -> None:
    """PDB без лиганда (только вода) → None."""
    pdb = _write_pdb(
        tmp_path,
        """\
        ATOM      1  CA  ALA A   1      10.000  20.000  30.000  1.00 10.00           C
        HETATM  100  O   HOH A 300       5.000   5.000   5.000  1.00  5.00           O
        END
        """,
    )
    coords = extract_ligand_coords(pdb)
    assert coords is None


def test_exclude_ions(tmp_path: Path) -> None:
    """Ионы (NA, CL, SO4) исключаются."""
    pdb = _write_pdb(
        tmp_path,
        """\
        ATOM      1  CA  ALA A   1      10.000  20.000  30.000  1.00 10.00           C
        HETATM  100  NA  NA  A 400       1.000   1.000   1.000  1.00  5.00          NA
        HETATM  101  CL  CL  A 401       2.000   2.000   2.000  1.00  5.00          CL
        HETATM  102  S   SO4 A 402       3.000   3.000   3.000  1.00  5.00           S
        END
        """,
    )
    coords = extract_ligand_coords(pdb)
    assert coords is None


# --- compute_box ---


def test_compute_box_padding() -> None:
    """Padding расчёт: extent = max-min + 2*padding."""
    coords = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
    center_str, size_str = compute_box(coords, padding=5.0)

    center = [float(x) for x in center_str.split(",")]
    np.testing.assert_allclose(center, [5.0, 10.0, 15.0])

    size = [float(x) for x in size_str.split(",")]
    # extent: [10, 20, 30] + 2*5 = [20, 30, 40]
    np.testing.assert_allclose(size, [20.0, 30.0, 40.0])


# --- extract_binding_box ---


def test_extract_binding_box_with_ligand(tmp_path: Path) -> None:
    """Лиганд есть → box по лиганду."""
    pdb = _write_pdb(
        tmp_path,
        """\
        ATOM      1  CA  ALA A   1      50.000  50.000  50.000  1.00 10.00           C
        HETATM  100  C1  LIG A 200       0.000   0.000   0.000  1.00  5.00           C
        HETATM  101  C2  LIG A 200      10.000  10.000  10.000  1.00  5.00           C
        END
        """,
    )
    center_str, _size_str = extract_binding_box(pdb, padding=5.0)
    center = [float(x) for x in center_str.split(",")]
    np.testing.assert_allclose(center, [5.0, 5.0, 5.0])


def test_extract_binding_box_fallback(tmp_path: Path) -> None:
    """Нет лиганда → fallback на центр белка + куб 30×30×30."""
    pdb = _write_pdb(
        tmp_path,
        """\
        ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C
        ATOM      2  CA  ALA A   2      20.000  20.000  20.000  1.00 10.00           C
        END
        """,
    )
    center_str, size_str = extract_binding_box(pdb)
    center = [float(x) for x in center_str.split(",")]
    np.testing.assert_allclose(center, [10.0, 10.0, 10.0])
    assert size_str == "30.00,30.00,30.00"
