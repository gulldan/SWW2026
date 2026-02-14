"""Tests for dualdocking.analysis.calibration.enrichment."""

from __future__ import annotations

from pathlib import Path

import pytest

from dualdocking.analysis.calibration.enrichment import (
    CSV_COLUMNS,
    load_decoy_scores,
)


class TestCsvColumns:
    def test_expected_columns(self):
        assert "target_id" in CSV_COLUMNS
        assert "auroc" in CSV_COLUMNS
        assert "bedroc_alpha20" in CSV_COLUMNS
        assert "status" in CSV_COLUMNS
        assert "passed" in CSV_COLUMNS


class TestLoadDecoyScores:
    def test_loads_csv(self, tmp_path):
        import pandas as pd

        df = pd.DataFrame({"score": [-5.0, -6.0, -7.0]})
        df.to_csv(tmp_path / "CDK2.csv", index=False)

        scores = load_decoy_scores(tmp_path / "CDK2")
        assert len(scores) == 3
        assert scores[0] == pytest.approx(-5.0)

    def test_missing_returns_empty(self, tmp_path):
        scores = load_decoy_scores(tmp_path / "MISSING")
        assert scores == []

    def test_loads_parquet(self, tmp_path):
        import pandas as pd

        df = pd.DataFrame({"score": [-4.0, -5.5]})
        df.to_parquet(tmp_path / "ERK2.parquet")

        scores = load_decoy_scores(tmp_path / "ERK2")
        assert len(scores) == 2
