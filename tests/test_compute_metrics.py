"""Tests for scripts/compute_metrics.py."""

import json

import numpy as np
import pandas as pd
import scripts.compute_metrics as mod
import yaml
from click.testing import CliRunner
from scripts.compute_metrics import main


def _metrics_cfg():
    return {"thresholds": {"pair1": {"target_a": -7.0, "target_b": -6.5}}}


def _make_parquet(path, n=5):
    rng = np.random.default_rng(42)
    rows = []
    for m in ("single", "double"):
        rows.append(pd.DataFrame({
            "smiles": [f"C{'C' * i}" for i in range(n)],
            "score_a": rng.uniform(-10, -3, n),
            "score_b": rng.uniform(-10, -3, n),
            "combined_min": rng.uniform(-10, -3, n),
            "pair": ["pair1"] * n,
            "mode": [m] * n,
        }))
    df = pd.concat(rows, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_main_computes(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "metrics.yaml").write_text(yaml.dump(_metrics_cfg()))

    parquet_path = tmp_path / "report" / "data" / "all_results.parquet"
    _make_parquet(parquet_path)

    runner = CliRunner()
    result = runner.invoke(main, [
        "--input", "report/data/all_results.parquet",
        "--metrics-config", "configs/metrics.yaml",
    ])
    assert result.exit_code == 0, result.output

    metrics = json.loads((tmp_path / "report" / "metrics.json").read_text())
    stats = json.loads((tmp_path / "report" / "stats.json").read_text())
    assert "pair1" in metrics
    assert "pair1" in stats


def test_main_missing_input(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    runner = CliRunner()
    result = runner.invoke(main, ["--input", "report/data/missing.parquet"])
    assert result.exit_code == 1
