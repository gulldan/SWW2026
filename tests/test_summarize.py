"""Tests for scripts/summarize.py."""

import json

import pandas as pd
import yaml
from click.testing import CliRunner
from scripts.summarize import _build_report, _score_distribution_block, _top_molecules_block, main


def _make_df(n: int = 5, pair: str = "pair1", mode: str = "single") -> pd.DataFrame:
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "C=CC"][:n]
    return pd.DataFrame({
        "smiles": smiles,
        "score_a": [-8.0 + i for i in range(n)],
        "score_b": [-7.0 + i for i in range(n)],
        "combined_min": [-7.5 + i for i in range(n)],
        "pair": [pair] * n,
        "mode": [mode] * n,
    })


def _pairs_cfg() -> dict:
    return {
        "pairs": {
            "pair1": {
                "target_a": {"id": "TargetA"},
                "target_b": {"id": "TargetB"},
                "description": "Test pair",
            }
        }
    }


def _thresholds_cfg() -> dict:
    return {
        "thresholds": {
            "pair1": {"target_a": -7.0, "target_b": -6.5},
        }
    }


# --- unit tests ---


def test_score_distribution_block():
    scores = pd.Series([-8.0, -6.0, -7.0, -5.0, -9.0])
    text = _score_distribution_block(scores, "TargetA")
    assert "median" in text
    assert "min" in text
    assert "max" in text


def test_score_distribution_block_empty():
    scores = pd.Series([], dtype=float)
    text = _score_distribution_block(scores, "TargetA")
    assert "нет данных" in text


def test_top_molecules_block():
    df = _make_df(5)
    text = _top_molecules_block(df, "A", "B", n=3)
    # Top-3 by combined_min ascending: CCO, c1ccccc1, CC(=O)O
    assert "CCO" in text
    assert "c1ccccc1" in text
    assert "CC(=O)O" in text


def test_build_report_structure():
    df = pd.concat([_make_df(3, mode="single"), _make_df(3, mode="double")])
    metrics = {"pair1": {"single": {"hit_rate_a": 0.5, "hit_rate_b": 0.3, "dual_hit_rate": 0.1}}}
    stats = {"pair1": {"combined_min": {"mannwhitney_pvalue": 0.03, "ks_pvalue": 0.04, "rank_biserial_r": 0.2}}}

    report = _build_report(df, metrics, stats, _pairs_cfg(), _thresholds_cfg())

    assert "СВОДНЫЙ ОТЧЁТ" in report
    assert "ОБЩАЯ ОЦЕНКА" in report
    assert "pair1" in report
    assert "SINGLE" in report
    assert "DOUBLE" in report


def test_main_cli(tmp_path, monkeypatch):
    # Prepare fixture data
    (tmp_path / "report" / "data").mkdir(parents=True)
    (tmp_path / "configs").mkdir()

    df = pd.concat([_make_df(3, mode="single"), _make_df(3, mode="double")])
    df.to_parquet(tmp_path / "report" / "data" / "all_results.parquet")

    (tmp_path / "report" / "metrics.json").write_text(json.dumps({}))
    (tmp_path / "report" / "stats.json").write_text(json.dumps({}))
    (tmp_path / "configs" / "pairs.yaml").write_text(yaml.dump(_pairs_cfg()))
    (tmp_path / "configs" / "metrics.yaml").write_text(yaml.dump(_thresholds_cfg()))

    import scripts.summarize as mod
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    runner = CliRunner()
    result = runner.invoke(main, [
        "--input", "report/data/all_results.parquet",
        "--metrics", "report/metrics.json",
        "--stats", "report/stats.json",
        "--config", "configs/pairs.yaml",
        "--thresholds", "configs/metrics.yaml",
        "--output", "report/summary.txt",
    ])
    assert result.exit_code == 0, result.output

    out = tmp_path / "report" / "summary.txt"
    assert out.exists()
    content = out.read_text()
    assert "СВОДНЫЙ ОТЧЁТ" in content
