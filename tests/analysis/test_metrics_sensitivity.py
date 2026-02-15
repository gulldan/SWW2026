"""Tests for dualdocking.analysis.metrics.sensitivity module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dualdocking.analysis.metrics.sensitivity import (
    SensitivityConfig,
    SensitivityResult,
    _sem,
    compute_hr_both,
    run_sensitivity,
)


# ---------------------------------------------------------------------------
# compute_hr_both
# ---------------------------------------------------------------------------


class TestComputeHrBoth:
    def test_all_hit(self) -> None:
        df = pd.DataFrame({"affinity_a": [-1.0, -2.0], "affinity_b": [-0.5, -1.5]})
        assert compute_hr_both(df, thr=0.0) == 1.0

    def test_none_hit(self) -> None:
        df = pd.DataFrame({"affinity_a": [1.0, 2.0], "affinity_b": [0.5, 1.5]})
        assert compute_hr_both(df, thr=0.0) == 0.0

    def test_partial_hit(self) -> None:
        df = pd.DataFrame({"affinity_a": [-1.0, 2.0], "affinity_b": [-0.5, -1.5]})
        # Only first row hits both
        assert abs(compute_hr_both(df, thr=0.0) - 0.5) < 1e-10

    def test_nan_excluded(self) -> None:
        df = pd.DataFrame(
            {"affinity_a": [-1.0, float("nan")], "affinity_b": [-0.5, -1.5]}
        )
        # Only first row is valid
        assert compute_hr_both(df, thr=0.0) == 1.0

    def test_empty_returns_nan(self) -> None:
        df = pd.DataFrame({"affinity_a": [], "affinity_b": []})
        assert np.isnan(compute_hr_both(df, thr=0.0))

    def test_threshold_sensitivity(self) -> None:
        df = pd.DataFrame({"affinity_a": [-0.5, -1.5], "affinity_b": [-0.3, -2.0]})
        # At thr=0.0: both rows hit
        assert compute_hr_both(df, thr=0.0) == 1.0
        # At thr=-1.0: only row 1 hits both
        assert abs(compute_hr_both(df, thr=-1.0) - 0.5) < 1e-10
        # At thr=-3.0: none hit
        assert compute_hr_both(df, thr=-3.0) == 0.0


# ---------------------------------------------------------------------------
# _sem
# ---------------------------------------------------------------------------


class TestSem:
    def test_matches_expected(self) -> None:
        vals = [1.0, 2.0, 3.0]
        expected = float(np.std(vals, ddof=1) / np.sqrt(3))
        assert abs(_sem(vals) - expected) < 1e-10

    def test_single_returns_zero(self) -> None:
        assert _sem([5.0]) == 0.0


# ---------------------------------------------------------------------------
# run_sensitivity (integration)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_project(tmp_path: Path) -> Path:
    """Create minimal project structure for sensitivity analysis."""
    project = tmp_path / "project"
    project.mkdir()

    pairs = {"P1": {"target_a": {"id": "T1"}, "target_b": {"id": "T2"}, "set": "core"}}

    for mode in ["single_a", "single_b", "double"]:
        for seed in [42, 137]:
            run_dir = project / "output" / "results" / "P1" / mode / f"seed_{seed}"
            run_dir.mkdir(parents=True)

            rng = np.random.default_rng(seed)
            n = 50
            if mode == "double":
                affs_a = rng.normal(-0.5, 0.5, n)
                affs_b = rng.normal(-0.5, 0.5, n)
            elif mode == "single_a":
                affs_a = rng.normal(-0.8, 0.5, n)
                affs_b = rng.normal(0.5, 0.5, n)
            else:
                affs_a = rng.normal(0.5, 0.5, n)
                affs_b = rng.normal(-0.8, 0.5, n)

            df = pd.DataFrame(
                {
                    "smiles_canonical": [f"C{'C' * i}O" for i in range(n)],
                    "affinity_a": affs_a,
                    "affinity_b": affs_b,
                }
            )
            df.to_parquet(run_dir / "scored_joined.parquet", index=False)

            # Fairness check: generated.parquet needs ~1000 molecules
            n_gen = 1000
            gen_df = pd.DataFrame({"smiles": [f"C{'C' * i}O" for i in range(n_gen)]})
            gen_df.to_parquet(run_dir / "generated.parquet", index=False)

            # Quality gate metadata under freedpp_workspace/
            ws_dir = run_dir / "freedpp_workspace"
            ws_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "total_calls": 100,
                "invalid_reward_fraction_a": 0.01,
                "invalid_reward_fraction_b": 0.01,
                "timeout_fraction_a": 0.01,
                "timeout_fraction_b": 0.01,
            }
            (ws_dir / "reward_metadata.json").write_text(json.dumps(meta))

    return project


def test_run_sensitivity_basic(mock_project: Path) -> None:
    """Sensitivity analysis runs without error and produces structured output."""
    cfg = SensitivityConfig(
        project_root=mock_project,
        pairs={"P1": {"target_a": {"id": "T1"}, "target_b": {"id": "T2"}, "set": "core"}},
        modes=["single_a", "single_b", "double"],
        seeds=[42, 137],
        metrics_cfg={},
        exp_cfg={},
        quality_gate={},
        thresholds=[-1.0, 0.0, 1.0],
        n_bootstrap=100,  # small for speed
    )
    result = run_sensitivity(cfg)

    assert isinstance(result, SensitivityResult)
    assert "thresholds_tested" in result.json_out
    assert result.json_out["thresholds_tested"] == [-1.0, 0.0, 1.0]
    assert "results" in result.json_out
    assert "robustness_summary" in result.json_out

    # Each threshold should have P1
    for thr_key in ["-1.0", "0.0", "1.0"]:
        assert thr_key in result.json_out["results"]
        assert "P1" in result.json_out["results"][thr_key]

    # Rows should be populated
    assert len(result.rows) > 0


def test_run_sensitivity_robustness_summary(mock_project: Path) -> None:
    """Robustness summary correctly counts core pairs."""
    cfg = SensitivityConfig(
        project_root=mock_project,
        pairs={"P1": {"target_a": {"id": "T1"}, "target_b": {"id": "T2"}, "set": "core"}},
        modes=["single_a", "single_b", "double"],
        seeds=[42, 137],
        metrics_cfg={},
        exp_cfg={},
        thresholds=[0.0],
        n_bootstrap=100,
    )
    result = run_sensitivity(cfg)

    summary = result.json_out["robustness_summary"]
    assert "0.0" in summary
    assert summary["0.0"]["n_core_total"] == 1


def test_run_sensitivity_defaults_missing_set_to_core(mock_project: Path) -> None:
    """Pairs without 'set' should default to core in robustness summary."""
    cfg = SensitivityConfig(
        project_root=mock_project,
        pairs={"P1": {"target_a": {"id": "T1"}, "target_b": {"id": "T2"}}},
        modes=["single_a", "single_b", "double"],
        seeds=[42, 137],
        metrics_cfg={},
        exp_cfg={},
        thresholds=[0.0],
        n_bootstrap=100,
    )
    result = run_sensitivity(cfg)
    summary = result.json_out["robustness_summary"]["0.0"]
    assert summary["n_core_total"] == 1
