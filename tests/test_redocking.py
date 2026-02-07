"""Tests for re-docking validation utilities."""

import textwrap

import numpy as np
import pytest


class TestSymmetricRmsd:
    def test_identical_coords(self):
        from scripts.validate_redocking import symmetric_rmsd

        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert symmetric_rmsd(coords, coords) == pytest.approx(0.0, abs=1e-10)

    def test_known_displacement(self):
        from scripts.validate_redocking import symmetric_rmsd

        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        # Each atom displaced by 1.0 A
        assert symmetric_rmsd(a, b) == pytest.approx(1.0, abs=1e-10)

    def test_shuffled_atoms(self):
        from scripts.validate_redocking import symmetric_rmsd

        a = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        # Same atoms but shuffled order
        b = np.array([[0.0, 10.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        assert symmetric_rmsd(a, b) == pytest.approx(0.0, abs=1e-10)

    def test_atom_count_mismatch_raises(self):
        from scripts.validate_redocking import symmetric_rmsd

        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="Atom count mismatch"):
            symmetric_rmsd(a, b)


class TestCentroidDistance:
    def test_same_centroid(self):
        from scripts.validate_redocking import centroid_distance

        a = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        b = np.array([[0.5, 0.5, 0.0], [1.5, -0.5, 0.0]])
        assert centroid_distance(a, b) == pytest.approx(0.0, abs=1e-10)

    def test_known_distance(self):
        from scripts.validate_redocking import centroid_distance

        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[3.0, 4.0, 0.0]])
        assert centroid_distance(a, b) == pytest.approx(5.0)


class TestParseSdfCoords:
    def test_simple_sdf(self):
        from scripts.validate_redocking import parse_sdf_coords

        sdf = textwrap.dedent("""\
            mol
                 RDKit

              3  2  0  0  0  0  0  0  0  0999 V2000
                1.0000    2.0000    3.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
                4.0000    5.0000    6.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
                0.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
              1  2  1  0
              1  3  1  0
            M  END
        """)
        coords = parse_sdf_coords(sdf)
        assert coords is not None
        assert coords.shape == (2, 3)  # H excluded
        np.testing.assert_allclose(coords[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(coords[1], [4.0, 5.0, 6.0])

    def test_empty_sdf(self):
        from scripts.validate_redocking import parse_sdf_coords

        assert parse_sdf_coords("") is None
        assert parse_sdf_coords("short\ntext") is None


class TestExtractLigandInfo:
    def test_finds_ligand(self, tmp_path):
        from dualdocking.binding_box import extract_ligand_info

        pdb = textwrap.dedent("""\
            ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C
            HETATM    2  C1  LIG B   1      10.000  20.000  30.000  1.00  5.00           C
            HETATM    3  O1  LIG B   1      11.000  21.000  31.000  1.00  5.00           O
            HETATM    4  O   HOH C   1      50.000  50.000  50.000  1.00 20.00           O
            END
        """)
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text(pdb)

        info = extract_ligand_info(pdb_path)
        assert info is not None
        assert info.resname == "LIG"
        assert info.chain == "B"
        assert info.coords.shape == (2, 3)
        np.testing.assert_allclose(info.coords[0], [10.0, 20.0, 30.0])

    def test_no_ligand(self, tmp_path):
        from dualdocking.binding_box import extract_ligand_info

        pdb = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00           C\nEND\n"
        pdb_path = tmp_path / "empty.pdb"
        pdb_path.write_text(pdb)

        assert extract_ligand_info(pdb_path) is None
