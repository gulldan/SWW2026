"""Tests for dualdocking.analysis.calibration.redocking."""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from dualdocking.analysis.calibration.redocking import (
    CSV_COLUMNS,
    RMSD_THRESHOLD,
    RedockResult,
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIPPED_APO,
    STATUS_SKIPPED_BLOCKED,
    centroid_distance,
    parse_sdf_coords,
    parse_unidock_output_sdf,
    resolve_pdb_for_target,
    result_to_dict,
    symmetric_rmsd,
    write_csv,
)


class TestSymmetricRmsd:
    def test_identical_coords(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert symmetric_rmsd(coords, coords) == pytest.approx(0.0, abs=1e-10)

    def test_known_displacement(self):
        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        assert symmetric_rmsd(a, b) == pytest.approx(1.0, abs=1e-10)

    def test_shuffled_atoms(self):
        a = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        b = np.array([[0.0, 10.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        assert symmetric_rmsd(a, b) == pytest.approx(0.0, abs=1e-10)

    def test_atom_count_mismatch_raises(self):
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="Atom count mismatch"):
            symmetric_rmsd(a, b)


class TestCentroidDistance:
    def test_same_centroid(self):
        a = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        b = np.array([[0.5, 0.5, 0.0], [1.5, -0.5, 0.0]])
        assert centroid_distance(a, b) == pytest.approx(0.0, abs=1e-10)

    def test_known_distance(self):
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[3.0, 4.0, 0.0]])
        assert centroid_distance(a, b) == pytest.approx(5.0)


class TestParseSdfCoords:
    def test_simple_sdf(self):
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
        assert coords.shape == (2, 3)
        np.testing.assert_allclose(coords[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(coords[1], [4.0, 5.0, 6.0])

    def test_empty_sdf(self):
        assert parse_sdf_coords("") is None
        assert parse_sdf_coords("short\ntext") is None


class TestParseUnidockOutputSdf:
    def test_nonexistent_file(self, tmp_path):
        score, coords = parse_unidock_output_sdf(tmp_path / "nonexistent.sdf")
        assert score is None
        assert coords is None


class TestResolvePdbForTarget:
    def test_returns_path(self, tmp_path):
        # When no pair dirs exist, falls back to base path
        (tmp_path / "output" / "data" / "targets").mkdir(parents=True)
        result = resolve_pdb_for_target("CDK2", "1HCL", tmp_path)
        assert result == tmp_path / "output" / "data" / "targets" / "1hcl.pdb"


class TestRedockResult:
    def test_creation(self):
        r = RedockResult(
            target_id="CDK2",
            pdb_id="1HCL",
            ligand_resname="?",
            ligand_smiles="",
            n_crystal_atoms=0,
            n_docked_atoms=0,
            docking_score=None,
            rmsd=None,
            centroid_dist=None,
            passed=False,
            status=STATUS_SKIPPED_APO,
        )
        assert r.target_id == "CDK2"
        assert r.status == "skipped:apo"

    def test_result_to_dict(self):
        r = RedockResult(
            target_id="ERK2",
            pdb_id="4QTA",
            ligand_resname="38Z",
            ligand_smiles="CCC",
            n_crystal_atoms=20,
            n_docked_atoms=20,
            docking_score=-8.5,
            rmsd=1.2,
            centroid_dist=0.8,
            passed=True,
            status=STATUS_PASS,
        )
        d = result_to_dict(r)
        assert d["target_id"] == "ERK2"
        assert d["status"] == "pass"
        assert d["rmsd"] == 1.2


class TestWriteCsv:
    def test_csv_output(self, tmp_path):
        import csv as csv_mod

        results = [
            RedockResult(
                target_id="ERK2",
                pdb_id="4QTA",
                ligand_resname="38Z",
                ligand_smiles="CCC",
                n_crystal_atoms=20,
                n_docked_atoms=20,
                docking_score=-8.5,
                rmsd=1.2,
                centroid_dist=0.8,
                passed=True,
                status=STATUS_PASS,
            ),
        ]

        csv_path = tmp_path / "test.csv"
        write_csv(results, csv_path)

        with csv_path.open() as f:
            reader = list(csv_mod.DictReader(f))

        assert len(reader) == 1
        assert reader[0]["target_id"] == "ERK2"
        assert reader[0]["status"] == STATUS_PASS


class TestStatusConstants:
    def test_values(self):
        assert STATUS_PASS == "pass"
        assert STATUS_FAIL == "fail"
        assert STATUS_SKIPPED_APO == "skipped:apo"
        assert STATUS_SKIPPED_BLOCKED == "skipped:blocked"
        assert STATUS_ERROR == "error"

    def test_rmsd_threshold(self):
        assert RMSD_THRESHOLD == 2.0

    def test_csv_columns(self):
        assert "target_id" in CSV_COLUMNS
        assert "rmsd" in CSV_COLUMNS
        assert "status" in CSV_COLUMNS
