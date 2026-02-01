"""Tests for scripts/render_molecules.py."""

from pathlib import Path

import pandas as pd
import yaml
from click.testing import CliRunner
from scripts.render_molecules import _is_hit, _load_target_names, _load_thresholds, _render_grid, main


def _pairs_yaml(path: Path) -> Path:
    cfg = {
        "pairs": {
            "pair1": {
                "target_a": {"id": "TargetA"},
                "target_b": {"id": "TargetB"},
            }
        }
    }
    f = path / "pairs.yaml"
    f.write_text(yaml.dump(cfg))
    return f


def _metrics_yaml(path: Path) -> Path:
    cfg = {
        "thresholds": {
            "pair1": {"target_a": -7.0, "target_b": -6.5},
        }
    }
    f = path / "metrics.yaml"
    f.write_text(yaml.dump(cfg))
    return f


def _make_df(n: int = 3, pair: str = "pair1", mode: str = "single") -> pd.DataFrame:
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "C=CC"][:n]
    return pd.DataFrame({
        "smiles": smiles,
        "score_a": [-8.0 + i for i in range(n)],
        "score_b": [-7.0 + i for i in range(n)],
        "combined_min": [-7.5 + i for i in range(n)],
        "pair": [pair] * n,
        "mode": [mode] * n,
    })


# --- unit tests ---


def test_load_target_names(tmp_path):
    f = _pairs_yaml(tmp_path)
    result = _load_target_names(f)
    assert result == {"pair1": ("TargetA", "TargetB")}


def test_load_thresholds(tmp_path):
    f = _metrics_yaml(tmp_path)
    result = _load_thresholds(f)
    assert result == {"pair1": (-7.0, -6.5)}


def test_is_hit_true():
    assert _is_hit(-8.0, -7.0, th_a=-7.0, th_b=-6.5) is True


def test_is_hit_false():
    assert _is_hit(-5.0, -7.0, th_a=-7.0, th_b=-6.5) is False


def test_render_grid_creates_png(tmp_path):
    df = _make_df(3)
    out = tmp_path / "grid.png"
    _render_grid(df, out, "Test Grid", "A", "B", -7.0, -6.5)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_grid_empty_df(tmp_path):
    df = _make_df(3).iloc[:0]
    out = tmp_path / "empty.png"
    _render_grid(df, out, "Empty", "A", "B", -7.0, -6.5)
    assert not out.exists()


def test_main_cli(tmp_path, monkeypatch):
    # Prepare fixture data under tmp_path mimicking PROJECT_ROOT layout
    (tmp_path / "report" / "data").mkdir(parents=True)
    (tmp_path / "report" / "figures").mkdir(parents=True)
    (tmp_path / "configs").mkdir()

    df = pd.concat([_make_df(3, mode="single"), _make_df(3, mode="double")])
    df.to_parquet(tmp_path / "report" / "data" / "all_results.parquet")
    _pairs_yaml(tmp_path / "configs")
    _metrics_yaml(tmp_path / "configs")

    import scripts.render_molecules as mod
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    runner = CliRunner()
    result = runner.invoke(main, [
        "--input", "report/data/all_results.parquet",
        "--output-dir", "report/figures",
        "--config", "configs/pairs.yaml",
        "--thresholds", "configs/metrics.yaml",
        "--top-n", "2",
    ])
    assert result.exit_code == 0, result.output

    figs = list((tmp_path / "report" / "figures").glob("*.png"))
    assert len(figs) >= 2
