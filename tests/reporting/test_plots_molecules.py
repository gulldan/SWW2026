"""Tests for dualdocking.reporting.plots.molecules module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dualdocking.reporting.plots.molecules import (
    MoleculeRenderConfig,
    is_hit,
    load_target_names,
    load_thresholds,
    render_all_pairs,
    render_molecule_grid,
)


class TestMoleculeRenderConfig:
    """Test MoleculeRenderConfig dataclass."""

    def test_frozen(self) -> None:
        """MoleculeRenderConfig should be frozen."""
        config = MoleculeRenderConfig(modes=["single_a"])
        with pytest.raises(AttributeError, match="cannot assign to field"):
            config.modes = ["other"]  # type: ignore[misc]

    def test_defaults(self) -> None:
        """Should have sensible defaults."""
        config = MoleculeRenderConfig()
        assert config.modes == ["single_a", "single_b", "double"]
        assert config.primary_col == "combined_min"
        assert config.mode_top_n == 25
        assert config.top_n == 5
        assert config.dpi == 150


class TestIsHit:
    """Test is_hit function."""

    def test_both_below_threshold(self) -> None:
        """Should return True if both scores below thresholds."""
        assert is_hit(-7.0, -6.0, -5.0, -5.0) is True

    def test_one_above_threshold(self) -> None:
        """Should return False if one score above threshold."""
        assert is_hit(-4.0, -6.0, -5.0, -5.0) is False
        assert is_hit(-7.0, -4.0, -5.0, -5.0) is False

    def test_equal_to_threshold(self) -> None:
        """Should return True if scores equal thresholds."""
        assert is_hit(-5.0, -5.0, -5.0, -5.0) is True


class TestLoadTargetNames:
    """Test load_target_names function."""

    def test_loads_from_yaml(self, tmp_path: Path) -> None:
        """Should load target names from pairs.yaml."""
        yaml_content = """
pairs:
  CDK2_ERK2:
    target_a:
      id: CDK2
    target_b:
      id: ERK2
  AChE_BChE:
    target_a:
      id: AChE
    target_b:
      id: BChE
"""
        yaml_file = tmp_path / "pairs.yaml"
        yaml_file.write_text(yaml_content)

        names = load_target_names(yaml_file)
        assert names == {
            "CDK2_ERK2": ("CDK2", "ERK2"),
            "AChE_BChE": ("AChE", "BChE"),
        }


class TestLoadThresholds:
    """Test load_thresholds function."""

    def test_loads_from_yaml(self, tmp_path: Path) -> None:
        """Should load thresholds from metrics.yaml."""
        yaml_content = """
thresholds:
  CDK2_ERK2:
    target_a: -6.5
    target_b: -5.5
  AChE_BChE:
    target_a: -7.0
    target_b: -6.0
"""
        yaml_file = tmp_path / "metrics.yaml"
        yaml_file.write_text(yaml_content)

        thresh = load_thresholds(yaml_file)
        assert thresh == {
            "CDK2_ERK2": (-6.5, -5.5),
            "AChE_BChE": (-7.0, -6.0),
        }


class TestRenderMoleculeGrid:
    """Test render_molecule_grid function."""

    @pytest.fixture
    def sample_molecules(self) -> pd.DataFrame:
        """Create sample molecule DataFrame."""
        return pd.DataFrame({
            "smiles": ["CCO", "CCC", "CCCC"],
            "affinity_a": [-7.0, -6.5, -6.0],
            "affinity_b": [-6.5, -6.0, -5.5],
            "combined_min": [-6.5, -6.0, -5.5],
        })

    def test_creates_output_file(self, sample_molecules: pd.DataFrame, tmp_path: Path) -> None:
        """Should create PNG file."""
        output = tmp_path / "test.png"
        render_molecule_grid(
            sample_molecules,
            output,
            "Test Title",
            "CDK2",
            "ERK2",
            -6.0,
            -6.0,
        )
        assert output.exists()

    def test_empty_dataframe(self, tmp_path: Path) -> None:
        """Should handle empty DataFrame gracefully."""
        df_empty = pd.DataFrame(columns=["smiles", "affinity_a", "affinity_b"])
        output = tmp_path / "empty.png"
        render_molecule_grid(
            df_empty,
            output,
            "Empty Test",
            "A",
            "B",
            0.0,
            0.0,
        )
        # Should not create file for empty DataFrame
        assert not output.exists()


class TestRenderAllPairs:
    """Test render_all_pairs function."""

    @pytest.fixture
    def sample_data_with_smiles(self) -> pd.DataFrame:
        """Create sample DataFrame with SMILES."""
        np.random.seed(42)
        return pd.DataFrame({
            "pair": ["CDK2_ERK2"] * 10,
            "mode": ["single_a"] * 5 + ["double"] * 5,
            "smiles": ["CCO"] * 10,
            "affinity_a": np.random.normal(-6, 1, 10),
            "affinity_b": np.random.normal(-5, 1, 10),
            "combined_min": np.random.normal(-5.5, 1, 10),
        })

    def test_creates_output_files(self, sample_data_with_smiles: pd.DataFrame, tmp_path: Path) -> None:
        """Should create molecule grid PNGs."""
        target_names = {"CDK2_ERK2": ("CDK2", "ERK2")}
        thresholds = {"CDK2_ERK2": (-6.0, -5.0)}

        render_all_pairs(
            sample_data_with_smiles,
            tmp_path,
            target_names,
            thresholds,
        )

        # Should create mode-specific and top-hits files
        files = list(tmp_path.glob("CDK2_ERK2_*.png"))
        assert len(files) > 0

    def test_mode_grid_is_capped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mode-specific grids should render only top-N molecules."""
        n = 60
        df = pd.DataFrame({
            "pair": ["CDK2_ERK2"] * n,
            "mode": ["single_a"] * n,
            "smiles": ["CCO"] * n,
            "affinity_a": np.linspace(-9.0, -3.0, n),
            "affinity_b": np.linspace(-8.5, -2.5, n),
            "combined_min": np.linspace(-8.0, -2.0, n),
        })
        target_names = {"CDK2_ERK2": ("CDK2", "ERK2")}
        thresholds = {"CDK2_ERK2": (-6.0, -5.0)}
        seen_sizes: list[int] = []

        import dualdocking.reporting.plots.molecules as mod

        def _capture(df_in: pd.DataFrame, *_args: object, **_kwargs: object) -> None:
            seen_sizes.append(len(df_in))

        monkeypatch.setattr(mod, "render_molecule_grid", _capture)

        render_all_pairs(
            df,
            tmp_path,
            target_names,
            thresholds,
            config=MoleculeRenderConfig(modes=["single_a"], mode_top_n=12, top_n=5),
        )

        assert seen_sizes
        assert seen_sizes[0] == 12
