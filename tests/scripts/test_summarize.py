"""Tests for cli/summarize.py."""

import json

import pandas as pd
import yaml
from click.testing import CliRunner

from cli.s5_analyze.summarize import _build_report, _score_distribution_block, _top_molecules_block, main


def _make_df(n: int = 5, pair: str = "pair1", mode: str = "single_a") -> pd.DataFrame:
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "C=CC"][:n]
    return pd.DataFrame(
        {
            "smiles": smiles,
            "score_a": [-8.0 + i for i in range(n)],
            "score_b": [-7.0 + i for i in range(n)],
            "combined_min": [-7.5 + i for i in range(n)],
            "combined_harmonic": [-7.3 + i for i in range(n)],
            "combined_geometric": [-7.4 + i for i in range(n)],
            "combined_weighted_sum_0.5": [-7.2 + i for i in range(n)],
            "combined_weighted_product_0.5": [-7.1 + i for i in range(n)],
            "combined_tchebycheff_0.5": [0.5 + i for i in range(n)],
            "pair": [pair] * n,
            "mode": [mode] * n,
        }
    )


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
    assert "no data" in text


def test_top_molecules_block():
    df = _make_df(5)
    text = _top_molecules_block(df, "A", "B", n=3, primary_col="combined_min")
    # Top-3 by combined_min ascending: CCO, c1ccccc1, CC(=O)O
    assert "CCO" in text
    assert "c1ccccc1" in text
    assert "CC(=O)O" in text


def test_build_report_structure():
    df = pd.concat(
        [
            _make_df(3, mode="single_a"),
            _make_df(3, mode="single_b"),
            _make_df(3, mode="double"),
        ]
    )
    metrics = {
        "pair1": {
            "single_a": {"hr_a_mean": 0.5, "hr_both_mean": 0.1, "balance_mean": 1.0},
            "single_b": {"hr_a_mean": 0.4, "hr_both_mean": 0.2, "balance_mean": 1.1},
            "double": {"hr_a_mean": 0.6, "hr_both_mean": 0.3, "balance_mean": 0.9},
        }
    }
    stats = {
        "pair1": {
            "H1a": {
                "metric": "hr_both",
                "observed_diff": 0.1,
                "p_value_adjusted": 0.04,
                "cliffs_delta": 0.5,
                "significant": True,
            },
        }
    }

    obj_columns = ["combined_min", "combined_harmonic", "combined_geometric"]
    report = _build_report(
        df,
        metrics,
        stats,
        _pairs_cfg(),
        _thresholds_cfg(),
        ["single_a", "single_b", "double"],
        objective_columns=obj_columns,
        primary_col="combined_min",
    )

    assert "SUMMARY REPORT" in report
    assert "pair1" in report
    assert "SINGLE_A" in report
    assert "SINGLE_B" in report
    assert "DOUBLE" in report
    assert "combined_min" in report
    assert "combined_harmonic" in report
    assert "combined_geometric" in report


def test_main_cli(tmp_path, monkeypatch):
    # Prepare fixture data
    (tmp_path / "report" / "data").mkdir(parents=True)
    (tmp_path / "configs").mkdir()

    df = pd.concat(
        [
            _make_df(3, mode="single_a"),
            _make_df(3, mode="single_b"),
            _make_df(3, mode="double"),
        ]
    )
    df.to_parquet(tmp_path / "report" / "data" / "all_results.parquet")

    (tmp_path / "report" / "metrics.json").write_text(json.dumps({}))
    (tmp_path / "report" / "stats.json").write_text(json.dumps({}))
    (tmp_path / "configs" / "pairs.yaml").write_text(yaml.dump(_pairs_cfg()))
    (tmp_path / "configs" / "metrics.yaml").write_text(yaml.dump(_thresholds_cfg()))
    exp = {
        "modes": ["single_a", "single_b", "double"],
        "objectives": {
            "baseline": "min",
            "alternatives": [{"kind": "harmonic"}],
        },
    }
    (tmp_path / "configs" / "experiment.yaml").write_text(yaml.dump(exp))

    import cli.s5_analyze.summarize as mod

    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--input",
            "report/data/all_results.parquet",
            "--metrics",
            "report/metrics.json",
            "--stats",
            "report/stats.json",
            "--config",
            "configs/pairs.yaml",
            "--thresholds",
            "configs/metrics.yaml",
            "--output",
            "report/summary.txt",
        ],
    )
    assert result.exit_code == 0, result.output

    out = tmp_path / "report" / "summary.txt"
    assert out.exists()
    content = out.read_text()
    assert "SUMMARY REPORT" in content
    assert "combined_min" in content
    assert "combined_harmonic" in content


def test_legacy_csv_fallback(tmp_path, monkeypatch):
    """DataFrame with only combined_min → no crash when objectives request more."""
    df = pd.DataFrame(
        {
            "smiles": ["CCO"],
            "score_a": [-8.0],
            "score_b": [-7.0],
            "combined_min": [-7.5],
            "pair": ["pair1"],
            "mode": ["double"],
        }
    )
    report = _build_report(
        df,
        {},
        {},
        _pairs_cfg(),
        _thresholds_cfg(),
        ["double"],
        objective_columns=["combined_min", "combined_harmonic"],
        primary_col="combined_min",
    )
    assert "combined_min" in report
    # combined_harmonic not in df → should not crash, just not appear in distribution block
