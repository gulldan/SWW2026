"""Tests for dualdocking.reporting.plots.distributions module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dualdocking.reporting.plots.distributions import (
    PlotConfig,
    plot_all_pairs,
    plot_score_comparison,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    data = {
        "pair": ["CDK2_ERK2"] * 100 + ["AChE_BChE"] * 100,
        "mode": (["single_a"] * 50 + ["double"] * 50) * 2,
        "affinity_a": np.random.normal(-6, 1, 200),
        "affinity_b": np.random.normal(-5, 1.5, 200),
        "combined_min": np.random.normal(-5.5, 1.2, 200),
    }
    return pd.DataFrame(data)


class TestPlotConfig:
    """Test PlotConfig dataclass."""

    def test_frozen(self, tmp_path: Path) -> None:
        """PlotConfig should be frozen."""
        config = PlotConfig(
            data_path=tmp_path / "data.parquet",
            output_dir=tmp_path / "figs",
            modes=["single_a", "double"],
        )
        with pytest.raises(AttributeError, match="cannot assign to field"):
            config.modes = ["other"]  # type: ignore[misc]

    def test_default_objective_columns(self, tmp_path: Path) -> None:
        """objective_columns defaults to None."""
        config = PlotConfig(
            data_path=tmp_path / "data.parquet",
            output_dir=tmp_path / "figs",
            modes=["single_a"],
        )
        assert config.objective_columns is None


class TestPlotScoreComparison:
    """Test plot_score_comparison function."""

    def test_creates_output_files(self, sample_data: pd.DataFrame, tmp_path: Path) -> None:
        """Should create histogram/ECDF and boxplot PNG files."""
        df_pair = sample_data[sample_data["pair"] == "CDK2_ERK2"]
        plot_score_comparison(
            df_pair,
            "CDK2_ERK2",
            tmp_path,
            modes=["single_a", "double"],
        )
        # Should create at least one histogram/ECDF pair and one boxplot
        pngs = list(tmp_path.glob("*.png"))
        assert len(pngs) >= 4  # affinity_a, affinity_b, combined_min + boxplot


class TestPlotAllPairs:
    """Test plot_all_pairs function."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError if input file missing."""
        config = PlotConfig(
            data_path=tmp_path / "nonexistent.parquet",
            output_dir=tmp_path / "figs",
            modes=["single_a"],
        )
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            plot_all_pairs(config)

    def test_creates_plots_for_all_pairs(self, sample_data: pd.DataFrame, tmp_path: Path) -> None:
        """Should generate plots for each unique pair."""
        data_file = tmp_path / "data.parquet"
        sample_data.to_parquet(data_file)

        config = PlotConfig(
            data_path=data_file,
            output_dir=tmp_path / "figs",
            modes=["single_a", "double"],
        )
        plot_all_pairs(config)

        # Should create plots for CDK2_ERK2 and AChE_BChE
        cdk2_files = list((tmp_path / "figs").glob("CDK2_ERK2_*.png"))
        ache_files = list((tmp_path / "figs").glob("AChE_BChE_*.png"))
        assert len(cdk2_files) > 0
        assert len(ache_files) > 0
