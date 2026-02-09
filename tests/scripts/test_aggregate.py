"""Tests for scripts/aggregate.py."""

import pandas as pd
import scripts.analysis.aggregate as mod
import yaml
from click.testing import CliRunner
from scripts.analysis.aggregate import main


def _pairs_cfg():
    return {"pairs": {"pair1": {"target_a": {"id": "A", "organism": "Homo sapiens"}, "target_b": {"id": "B", "organism": "Homo sapiens"}}}}


def _experiment_cfg():
    return {"modes": ["single_a", "single_b", "double"], "seeds": [42]}


def _make_scored_csv(path, n=3):
    df = pd.DataFrame({
        "smiles": ["CCO", "c1ccccc1", "CC(=O)O"][:n],
        "score_a": [-8.0 + i for i in range(n)],
        "score_b": [-7.0 + i for i in range(n)],
        "combined_min": [-7.5 + i for i in range(n)],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_reward_metadata(run_dir, *, run_valid=True):
    """Write a valid reward_metadata.json for quality gate."""
    ws = run_dir / "freedpp_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    import json

    metadata = {
        "total_calls": 100,
        "invalid_count_a": 2,
        "invalid_count_b": 1,
        "invalid_reward_fraction_a": 0.02,
        "invalid_reward_fraction_b": 0.01,
        "timeout_count_a": 1,
        "timeout_count_b": 0,
        "timeout_fraction_a": 0.01,
        "timeout_fraction_b": 0.00,
        "invalid_z_penalty": 4.0,
        "primary_target": "A",
        "reward_mode": "single_a",
        "run_valid": run_valid,
    }
    (ws / "reward_metadata.json").write_text(json.dumps(metadata, indent=2))


def test_main_aggregates(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pairs.yaml").write_text(yaml.dump(_pairs_cfg()))
    (tmp_path / "configs" / "experiment.yaml").write_text(yaml.dump(_experiment_cfg()))

    for m in ("single_a", "single_b", "double"):
        run_dir = tmp_path / "results" / "pair1" / m / "seed_42"
        _make_scored_csv(run_dir / "scored_joined.csv")
        _write_reward_metadata(run_dir)

    runner = CliRunner()
    result = runner.invoke(main, ["--config", "configs/pairs.yaml", "--experiment-config", "configs/experiment.yaml", "--output", "report/data/all_results.parquet"])
    assert result.exit_code == 0, result.output

    out = tmp_path / "report" / "data" / "all_results.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df["mode"].unique()) == {"single_a", "single_b", "double"}
    assert len(df) == 9  # 3 rows * 3 modes


def test_quality_gate_skips_invalid_run(tmp_path, monkeypatch):
    """Run with invalid metadata is excluded from parquet."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    (tmp_path / "configs").mkdir()
    # Use 2 seeds so one can be invalid and one valid
    exp = {"modes": ["single_a"], "seeds": [42, 99]}
    (tmp_path / "configs" / "pairs.yaml").write_text(yaml.dump(_pairs_cfg()))
    (tmp_path / "configs" / "experiment.yaml").write_text(yaml.dump(exp))

    # seed_42: valid
    run_42 = tmp_path / "results" / "pair1" / "single_a" / "seed_42"
    _make_scored_csv(run_42 / "scored_joined.csv")
    _write_reward_metadata(run_42, run_valid=True)

    # seed_99: invalid (high invalid fraction)
    run_99 = tmp_path / "results" / "pair1" / "single_a" / "seed_99"
    _make_scored_csv(run_99 / "scored_joined.csv")
    import json as json_mod

    ws99 = run_99 / "freedpp_workspace"
    ws99.mkdir(parents=True, exist_ok=True)
    (ws99 / "reward_metadata.json").write_text(
        json_mod.dumps(
            {
                "total_calls": 100,
                "invalid_count_a": 50,
                "invalid_count_b": 0,
                "invalid_reward_fraction_a": 0.50,
                "invalid_reward_fraction_b": 0.00,
                "timeout_count_a": 0,
                "timeout_count_b": 0,
                "timeout_fraction_a": 0.00,
                "timeout_fraction_b": 0.00,
                "invalid_z_penalty": 4.0,
                "primary_target": "A",
                "reward_mode": "single_a",
                "run_valid": False,
            }
        )
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--config",
            "configs/pairs.yaml",
            "--experiment-config",
            "configs/experiment.yaml",
            "--output",
            "report/data/all_results.parquet",
        ],
    )
    assert result.exit_code == 0, result.output

    out = tmp_path / "report" / "data" / "all_results.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    # Only seed_42 should be included (3 rows), seed_99 excluded
    assert len(df) == 3
    assert set(df["seed"].unique()) == {42}


def test_main_no_results(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pairs.yaml").write_text(yaml.dump(_pairs_cfg()))
    (tmp_path / "configs" / "experiment.yaml").write_text(yaml.dump(_experiment_cfg()))

    runner = CliRunner()
    result = runner.invoke(main, ["--config", "configs/pairs.yaml", "--experiment-config", "configs/experiment.yaml"])
    assert result.exit_code == 1
