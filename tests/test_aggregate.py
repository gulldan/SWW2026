"""Tests for scripts/aggregate.py."""

import pandas as pd
import scripts.aggregate as mod
import yaml
from click.testing import CliRunner
from scripts.aggregate import main


def _pairs_cfg():
    return {"pairs": {"pair1": {"target_a": {"id": "A"}, "target_b": {"id": "B"}}}}


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


def test_main_aggregates(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pairs.yaml").write_text(yaml.dump(_pairs_cfg()))
    (tmp_path / "configs" / "experiment.yaml").write_text(yaml.dump(_experiment_cfg()))

    for m in ("single_a", "single_b", "double"):
        _make_scored_csv(tmp_path / "results" / "pair1" / m / "seed_42" / "scored_joined.csv")

    runner = CliRunner()
    result = runner.invoke(main, ["--config", "configs/pairs.yaml", "--experiment-config", "configs/experiment.yaml", "--output", "report/data/all_results.parquet"])
    assert result.exit_code == 0, result.output

    out = tmp_path / "report" / "data" / "all_results.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(df["mode"].unique()) == {"single_a", "single_b", "double"}
    assert len(df) == 9  # 3 rows * 3 modes


def test_main_no_results(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pairs.yaml").write_text(yaml.dump(_pairs_cfg()))
    (tmp_path / "configs" / "experiment.yaml").write_text(yaml.dump(_experiment_cfg()))

    runner = CliRunner()
    result = runner.invoke(main, ["--config", "configs/pairs.yaml", "--experiment-config", "configs/experiment.yaml"])
    assert result.exit_code == 1
