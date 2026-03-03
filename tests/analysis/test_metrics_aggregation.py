"""Tests for dualdocking.analysis.metrics.aggregation module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dualdocking.analysis.metrics.aggregation import AggregateConfig, aggregate


@pytest.fixture()
def mock_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a minimal project structure with scored results."""
    project = tmp_path / "project"
    project.mkdir()

    # Create results for one pair, one mode, one seed
    run_dir = project / "output" / "results" / "P1" / "double" / "seed_42"
    run_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC"],
            "affinity_a": [-1.0, 0.5, -0.5],
            "affinity_b": [-0.8, -1.2, 0.3],
            "score_a": [-5.0, -3.0, -4.0],
            "score_b": [-4.0, -6.0, -2.0],
        }
    )
    df.to_parquet(run_dir / "scored_joined.parquet", index=False)

    # Write reward_metadata.json (quality gate) in freedpp_workspace
    import json

    ws_dir = run_dir / "freedpp_workspace"
    ws_dir.mkdir(parents=True)
    meta = {
        "total_calls": 100,
        "invalid_reward_fraction_a": 0.02,
        "invalid_reward_fraction_b": 0.01,
        "timeout_fraction_a": 0.01,
        "timeout_fraction_b": 0.01,
    }
    (ws_dir / "reward_metadata.json").write_text(json.dumps(meta))

    return project


def test_aggregate_single_run(mock_project: Path) -> None:
    """Test aggregation with a single (pair, mode, seed) combination."""
    output_path = mock_project / "report" / "data" / "all_results.parquet"

    cfg = AggregateConfig(
        project_root=mock_project,
        pairs={"P1": {"target_a": {"id": "T1"}, "target_b": {"id": "T2"}}},
        modes=["double"],
        seeds=[42],
        quality_gate={},
        output_path=output_path,
    )

    result_path = aggregate(cfg)
    assert result_path == output_path
    assert result_path.exists()

    result_df = pd.read_parquet(result_path)
    assert len(result_df) == 3
    assert "pair" in result_df.columns
    assert "mode" in result_df.columns
    assert "seed" in result_df.columns
    assert result_df["pair"].iloc[0] == "P1"
    assert result_df["mode"].iloc[0] == "double"
    assert result_df["seed"].iloc[0] == 42
    # Filter columns added
    assert "passes_all_filters" in result_df.columns


def test_aggregate_missing_run(mock_project: Path) -> None:
    """Missing runs are skipped with a warning, not an error."""
    output_path = mock_project / "report" / "data" / "all_results.parquet"

    cfg = AggregateConfig(
        project_root=mock_project,
        pairs={"P1": {"target_a": {"id": "T1"}, "target_b": {"id": "T2"}}},
        modes=["double"],
        seeds=[42, 137],  # seed 137 does not exist
        quality_gate={},
        output_path=output_path,
    )

    result_path = aggregate(cfg)
    result_df = pd.read_parquet(result_path)
    # Only seed 42 data should be present
    assert len(result_df) == 3
    assert all(result_df["seed"] == 42)


def test_aggregate_no_results_exits(tmp_path: Path) -> None:
    """When no results are found at all, sys.exit(1) is called."""
    cfg = AggregateConfig(
        project_root=tmp_path,
        pairs={"P1": {"target_a": {"id": "T1"}, "target_b": {"id": "T2"}}},
        modes=["double"],
        seeds=[42],
        quality_gate={},
        output_path=tmp_path / "out.parquet",
    )

    with pytest.raises(SystemExit, match="1"):
        aggregate(cfg)


def test_aggregate_quality_gate_filters(mock_project: Path) -> None:
    """Runs failing quality gate are excluded."""
    import json

    # Overwrite metadata to fail quality gate
    run_dir = mock_project / "output" / "results" / "P1" / "double" / "seed_42"
    ws_dir = run_dir / "freedpp_workspace"
    meta = {
        "total_calls": 100,
        "invalid_reward_fraction_a": 0.50,  # far above default 0.10
        "invalid_reward_fraction_b": 0.30,
        "timeout_fraction_a": 0.30,
        "timeout_fraction_b": 0.30,
    }
    (ws_dir / "reward_metadata.json").write_text(json.dumps(meta))

    cfg = AggregateConfig(
        project_root=mock_project,
        pairs={"P1": {"target_a": {"id": "T1"}, "target_b": {"id": "T2"}}},
        modes=["double"],
        seeds=[42],
        quality_gate={},
        output_path=mock_project / "report" / "data" / "out.parquet",
    )

    with pytest.raises(SystemExit, match="1"):
        aggregate(cfg)
