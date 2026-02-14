"""Tests for cli/plot_results.py."""

import numpy as np
import pandas as pd
from click.testing import CliRunner

import cli.s6_report.plot_results as mod
from cli.s6_report.plot_results import main, plot_score_comparison


def _make_df(n=5):
    rng = np.random.default_rng(42)
    rows = []
    for m in ("single_a", "single_b", "double"):
        rows.append(
            pd.DataFrame(
                {
                    "smiles": [f"C{'C' * i}" for i in range(n)],
                    "score_a": rng.uniform(-10, -3, n),
                    "score_b": rng.uniform(-10, -3, n),
                    "combined_min": rng.uniform(-10, -3, n),
                    "combined_harmonic": rng.uniform(-10, -3, n),
                    "combined_geometric": rng.uniform(-10, -3, n),
                    "combined_weighted_sum_0.5": rng.uniform(-10, -3, n),
                    "combined_weighted_product_0.5": rng.uniform(-10, -3, n),
                    "combined_tchebycheff_0.5": rng.uniform(0, 3, n),
                    "pair": ["pair1"] * n,
                    "mode": [m] * n,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_plot_score_comparison(tmp_path):
    import matplotlib

    matplotlib.use("Agg")

    df = _make_df()
    obj_cols = [
        ("combined_min", "combined_min"),
        ("combined_harmonic", "combined_harmonic"),
        ("combined_geometric", "combined_geometric"),
        ("combined_weighted_sum_0.5", "combined_weighted_sum_0.5"),
        ("combined_weighted_product_0.5", "combined_weighted_product_0.5"),
        ("combined_tchebycheff_0.5", "combined_tchebycheff_0.5"),
    ]
    plot_score_comparison(df, "pair1", tmp_path, ["single_a", "single_b", "double"], objective_columns=obj_cols)

    # 2 (score_a, score_b) + 6 objectives = 8 histogram/ECDF PNGs + 1 boxplot = 9
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 9, f"Expected 9 PNGs, got {len(pngs)}: {[p.name for p in pngs]}"


def test_main_cli(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")

    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    parquet_dir = tmp_path / "report" / "data"
    parquet_dir.mkdir(parents=True)
    df = _make_df()
    df.to_parquet(parquet_dir / "all_results.parquet", index=False)

    # experiment config with objectives
    import yaml

    exp = {
        "modes": ["single_a", "single_b", "double"],
        "objectives": {
            "baseline": "min",
            "alternatives": [
                {"kind": "harmonic"},
                {"kind": "geometric"},
                {"kind": "weighted_sum", "weight_a": 0.5},
                {"kind": "weighted_product", "weight_a": 0.5},
                {"kind": "tchebycheff", "weight_a": 0.5},
            ],
        },
    }
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "experiment.yaml").write_text(yaml.dump(exp))

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--input",
            "report/data/all_results.parquet",
            "--output-dir",
            "report/figures",
        ],
    )
    assert result.exit_code == 0, result.output

    fig_dir = tmp_path / "report" / "figures"
    assert fig_dir.exists()
    pngs = list(fig_dir.glob("*.png"))
    # 8 histogram/ECDF + 1 boxplot = 9
    assert len(pngs) == 9, f"Expected 9 PNGs, got {len(pngs)}: {[p.name for p in pngs]}"


def test_missing_column_skipped(tmp_path):
    """Legacy DataFrame missing a column → no crash."""
    import matplotlib

    matplotlib.use("Agg")

    rng = np.random.default_rng(42)
    n = 5
    df = pd.DataFrame(
        {
            "smiles": [f"C{i}" for i in range(n)],
            "score_a": rng.uniform(-10, -3, n),
            "score_b": rng.uniform(-10, -3, n),
            "combined_min": rng.uniform(-10, -3, n),
            "pair": ["pair1"] * n,
            "mode": ["double"] * n,
        }
    )
    obj_cols = [
        ("combined_min", "combined_min"),
        ("combined_nonexistent", "combined_nonexistent"),
    ]
    # Should not crash even with missing column
    plot_score_comparison(df, "pair1", tmp_path, ["double"], objective_columns=obj_cols)
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) >= 1
