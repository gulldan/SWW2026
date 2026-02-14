"""Tests for dualdocking.analysis.metrics.correlation module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dualdocking.analysis.metrics.correlation import (
    CorrelationConfig,
    CorrelationResult,
    bootstrap_correlation,
    load_boltz_scores,
    load_docking_scores,
    run_correlation,
)


# ---------------------------------------------------------------------------
# bootstrap_correlation
# ---------------------------------------------------------------------------


class TestBootstrapCorrelation:
    def test_perfect_positive(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        rho, ci_lo, ci_hi = bootstrap_correlation(x, y, method="spearman", n_bootstrap=500)
        assert abs(rho - 1.0) < 1e-10
        assert ci_lo > 0.8

    def test_perfect_negative(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        rho, ci_lo, ci_hi = bootstrap_correlation(x, y, method="spearman", n_bootstrap=500)
        assert abs(rho - (-1.0)) < 1e-10
        assert ci_hi < -0.5

    def test_pearson(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.1, 2.1, 2.9, 4.1, 5.0])
        r, ci_lo, ci_hi = bootstrap_correlation(x, y, method="pearson", n_bootstrap=500)
        assert r > 0.95

    def test_too_few_returns_nan(self) -> None:
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        rho, ci_lo, ci_hi = bootstrap_correlation(x, y)
        assert np.isnan(rho)
        assert np.isnan(ci_lo)
        assert np.isnan(ci_hi)

    def test_deterministic_with_seed(self) -> None:
        rng = np.random.default_rng(99)
        x = rng.normal(0, 1, 20)
        y = 0.5 * x + rng.normal(0, 0.5, 20)
        r1 = bootstrap_correlation(x, y, n_bootstrap=200, rng_seed=42)
        r2 = bootstrap_correlation(x, y, n_bootstrap=200, rng_seed=42)
        assert r1 == r2


# ---------------------------------------------------------------------------
# load_boltz_scores
# ---------------------------------------------------------------------------


class TestLoadBoltzScores:
    def test_basic(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "smiles_canonical": ["CCO", "CCN", "CCC"],
                "affinity_a": [-1.0, 0.5, -0.5],
                "affinity_b": [-0.8, -1.2, 0.3],
                "extra_col": [1, 2, 3],
            }
        )
        stem = tmp_path / "scored_joined"
        df.to_parquet(stem.with_suffix(".parquet"), index=False)

        result = load_boltz_scores(stem)
        assert list(result.columns) == ["smiles_canonical", "affinity_a", "affinity_b"]
        assert len(result) == 3

    def test_legacy_columns(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "smiles_canonical": ["CCO", "CCN"],
                "score_a": [-1.0, 0.5],
                "score_b": [-0.8, -1.2],
            }
        )
        stem = tmp_path / "scored_joined"
        df.to_parquet(stem.with_suffix(".parquet"), index=False)

        result = load_boltz_scores(stem)
        assert "affinity_a" in result.columns
        assert "affinity_b" in result.columns


# ---------------------------------------------------------------------------
# load_docking_scores
# ---------------------------------------------------------------------------


class TestLoadDockingScores:
    def test_no_workspace_returns_empty(self, tmp_path: Path) -> None:
        result = load_docking_scores(tmp_path)
        assert result.empty

    def test_single_mode(self, tmp_path: Path) -> None:
        mols_dir = tmp_path / "freedpp_workspace" / "output" / "run" / "mols"
        mols_dir.mkdir(parents=True)

        df = pd.DataFrame(
            {
                "Smiles": ["CCO", "CCN", "CCC"],
                "DockingScoreProperty": [-5.0, -3.0, -4.0],
            }
        )
        df.to_csv(mols_dir / "train_0.csv", index=False)

        result = load_docking_scores(tmp_path)
        assert not result.empty
        assert "smiles_canonical" in result.columns
        assert "docking_a" in result.columns
        assert len(result) == 3

    def test_double_mode(self, tmp_path: Path) -> None:
        mols_dir = tmp_path / "freedpp_workspace" / "output" / "run" / "mols"
        mols_dir.mkdir(parents=True)

        df = pd.DataFrame(
            {
                "smiles": ["CCO", "CCN"],
                "docking_a": [-5.0, -3.0],
                "docking_b": [-4.0, -6.0],
            }
        )
        df.to_csv(mols_dir / "train_0.csv", index=False)

        result = load_docking_scores(tmp_path)
        assert "docking_a" in result.columns
        assert "docking_b" in result.columns

    def test_deduplication_keeps_last(self, tmp_path: Path) -> None:
        mols_dir = tmp_path / "freedpp_workspace" / "output" / "run" / "mols"
        mols_dir.mkdir(parents=True)

        # Epoch 0
        df0 = pd.DataFrame(
            {"Smiles": ["CCO", "CCN"], "DockingScoreProperty": [-5.0, -3.0]}
        )
        df0.to_csv(mols_dir / "train_0.csv", index=False)

        # Epoch 1: CCO appears again with different score
        df1 = pd.DataFrame(
            {"Smiles": ["CCO"], "DockingScoreProperty": [-7.0]}
        )
        df1.to_csv(mols_dir / "train_1.csv", index=False)

        result = load_docking_scores(tmp_path)
        cco_row = result[result["smiles_canonical"] == "CCO"]
        assert len(cco_row) == 1
        assert cco_row["docking_a"].iloc[0] == -7.0  # last occurrence


# ---------------------------------------------------------------------------
# run_correlation (integration)
# ---------------------------------------------------------------------------


class TestRunCorrelation:
    def test_basic_correlation(self, tmp_path: Path) -> None:
        """End-to-end test with mock data."""
        project = tmp_path / "project"
        project.mkdir()

        run_dir = project / "output" / "results" / "P1" / "double" / "seed_42"
        run_dir.mkdir(parents=True)

        # Scored data
        scored_df = pd.DataFrame(
            {
                "smiles_canonical": [f"C{'C' * i}O" for i in range(20)],
                "affinity_a": np.linspace(-2.0, 1.0, 20),
                "affinity_b": np.linspace(-1.5, 0.5, 20),
            }
        )
        scored_df.to_parquet(run_dir / "scored_joined.parquet", index=False)

        # Docking data (correlated with affinities)
        mols_dir = run_dir / "freedpp_workspace" / "output" / "run" / "mols"
        mols_dir.mkdir(parents=True)
        dock_df = pd.DataFrame(
            {
                "Smiles": [f"C{'C' * i}O" for i in range(20)],
                "DockingScoreProperty": np.linspace(-8.0, -2.0, 20),
            }
        )
        dock_df.to_csv(mols_dir / "train_0.csv", index=False)

        cfg = CorrelationConfig(
            project_root=project,
            pairs={"P1": {"target_a": {"id": "T1"}, "target_b": {"id": "T2"}}},
            modes=["double"],
            seeds=[42],
            n_bootstrap=100,
        )
        result = run_correlation(cfg)

        assert isinstance(result, CorrelationResult)
        assert len(result.rows) > 0
        # Should have at least target_a correlation
        target_a_row = [r for r in result.rows if r["target"] == "target_a"]
        assert len(target_a_row) == 1
        assert target_a_row[0]["spearman_rho"] > 0.5  # should be strongly correlated

        # Summary
        assert "summary" in result.json_out
        assert result.json_out["summary"]["n_pair_mode_target_combos"] > 0

    def test_no_docking_data(self, tmp_path: Path) -> None:
        """When no docking data exists, result is empty."""
        project = tmp_path / "project"
        project.mkdir()

        run_dir = project / "output" / "results" / "P1" / "double" / "seed_42"
        run_dir.mkdir(parents=True)

        scored_df = pd.DataFrame(
            {
                "smiles_canonical": ["CCO", "CCN"],
                "affinity_a": [-1.0, 0.5],
                "affinity_b": [-0.8, -1.2],
            }
        )
        scored_df.to_parquet(run_dir / "scored_joined.parquet", index=False)

        cfg = CorrelationConfig(
            project_root=project,
            pairs={"P1": {"target_a": {"id": "T1"}, "target_b": {"id": "T2"}}},
            modes=["double"],
            seeds=[42],
        )
        result = run_correlation(cfg)
        assert len(result.rows) == 0
        assert np.isnan(result.json_out["summary"]["median_spearman_rho"])
