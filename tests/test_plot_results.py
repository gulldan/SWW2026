"""Tests for scripts/plot_results.py."""

import numpy as np
import pandas as pd
import scripts.plot_results as mod
from click.testing import CliRunner
from scripts.plot_results import main, plot_score_comparison


def _make_df(n=5):
    rng = np.random.default_rng(42)
    rows = []
    for m in ("single_a", "single_b", "double"):
        rows.append(pd.DataFrame({
            "smiles": [f"C{'C' * i}" for i in range(n)],
            "score_a": rng.uniform(-10, -3, n),
            "score_b": rng.uniform(-10, -3, n),
            "combined_min": rng.uniform(-10, -3, n),
            "pair": ["pair1"] * n,
            "mode": [m] * n,
        }))
    return pd.concat(rows, ignore_index=True)


def test_plot_score_comparison(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    df = _make_df()
    plot_score_comparison(df, "pair1", tmp_path, ["single_a", "single_b", "double"])

    expected = [
        "pair1_scorea_modes.png",
        "pair1_scoreb_modes.png",
        "pair1_combinedmin_modes.png",
        "pair1_boxplot.png",
    ]
    for fname in expected:
        assert (tmp_path / fname).exists(), f"Missing {fname}"


def test_main_cli(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")

    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    parquet_dir = tmp_path / "report" / "data"
    parquet_dir.mkdir(parents=True)
    df = _make_df()
    df.to_parquet(parquet_dir / "all_results.parquet", index=False)

    # experiment config for modes
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "experiment.yaml").write_text("modes: [single_a, single_b, double]\\n")

    runner = CliRunner()
    result = runner.invoke(main, [
        "--input", "report/data/all_results.parquet",
        "--output-dir", "report/figures",
    ])
    assert result.exit_code == 0, result.output

    fig_dir = tmp_path / "report" / "figures"
    assert fig_dir.exists()
    pngs = list(fig_dir.glob("*.png"))
    assert len(pngs) == 4
