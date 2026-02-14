"""Tests for dualdocking.analysis.calibration.normalization."""

from __future__ import annotations

import numpy as np
import pytest

from dualdocking.analysis.calibration.normalization import (
    boltz_version,
    docking_version_fallback,
    load_scores,
)


class TestBoltzVersion:
    def test_returns_string(self):
        v = boltz_version()
        assert isinstance(v, str)
        # Either a version number or "unknown"
        assert len(v) > 0


class TestDockingVersionFallback:
    def test_returns_unknown(self):
        assert docking_version_fallback("unidock") == "unknown"
        assert docking_version_fallback("/usr/local/bin/vina") == "unknown"


class TestLoadScores:
    def test_loads_csv(self, tmp_path):
        import pandas as pd

        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        df = pd.DataFrame({"score": [-5.0, -6.0, -7.0, -4.5]})
        df.to_csv(scores_dir / "CDK2.csv", index=False)

        result = load_scores(scores_dir, "CDK2")
        assert isinstance(result, np.ndarray)
        assert len(result) == 4
        np.testing.assert_allclose(result, [-5.0, -6.0, -7.0, -4.5])

    def test_loads_parquet(self, tmp_path):
        import pandas as pd

        scores_dir = tmp_path / "scores"
        scores_dir.mkdir()
        df = pd.DataFrame({"score": [-5.0, -6.0, -7.0]})
        df.to_parquet(scores_dir / "ERK2.parquet")

        result = load_scores(scores_dir, "ERK2")
        assert len(result) == 3

    def test_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Scores not found"):
            load_scores(tmp_path, "MISSING_TARGET")
