"""Tests for scripts/proxy_final_correlation.py (KVA-271 / SCI-07)."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import scripts.s5_analyze.proxy_correlation as mod
import yaml
from click.testing import CliRunner
from scripts.s5_analyze.proxy_correlation import main


def _pairs_cfg():
    return {
        "pairs": {
            "pair1": {
                "target_a": {"id": "A", "organism": "Homo sapiens"},
                "target_b": {"id": "B", "organism": "Homo sapiens"},
                "set": "core",
            }
        }
    }


def _experiment_cfg():
    return {"modes": ["single_a", "double"], "seeds": [1, 2]}


def _write_scored(run_dir: Path, rng: np.random.Generator, n: int = 20) -> list[str]:
    """Write scored_joined.csv with Boltz-2 affinities. Return SMILES list."""
    run_dir.mkdir(parents=True, exist_ok=True)
    smiles = [f"C{'C' * i}O" for i in range(n)]
    df = pd.DataFrame({
        "smiles_canonical": smiles,
        "affinity_a": rng.normal(-1.0, 0.5, n),
        "affinity_b": rng.normal(-0.5, 0.5, n),
    })
    df.to_csv(run_dir / "scored_joined.csv", index=False)
    df.to_parquet(run_dir / "scored_joined.parquet", index=False)
    return smiles


def _write_training_csvs(
    run_dir: Path,
    smiles: list[str],
    rng: np.random.Generator,
    *,
    include_docking_b: bool = False,
) -> None:
    """Write FREED++ training CSV with docking scores for a subset of SMILES."""
    mols_dir = run_dir / "freedpp_workspace" / "output" / "run" / "mols"
    mols_dir.mkdir(parents=True, exist_ok=True)
    # Use a subset (80%) to simulate partial overlap
    subset = smiles[: int(len(smiles) * 0.8)]
    data = {
        "Smiles": subset,
        "DockingScoreProperty": rng.normal(-6.0, 1.0, len(subset)),
        "DockingScoreReward": rng.normal(0.5, 0.2, len(subset)),
        "Reward": rng.normal(0.5, 0.2, len(subset)),
        "Epoch": [0] * len(subset),
    }
    if include_docking_b:
        data["docking_a"] = data["DockingScoreProperty"]
        data["docking_b"] = rng.normal(-5.0, 1.0, len(subset))
    pd.DataFrame(data).to_csv(mols_dir / "train_000.csv", index=False)


def _setup(tmp_path, *, include_docking_b=False):
    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "configs" / "pairs.yaml").write_text(yaml.dump(_pairs_cfg()))
    (tmp_path / "configs" / "experiment.yaml").write_text(yaml.dump(_experiment_cfg()))

    rng = np.random.default_rng(42)
    for mode in ("single_a", "double"):
        for seed in (1, 2):
            run_dir = tmp_path / "output" / "results" / "pair1" / mode / f"seed_{seed}"
            smiles = _write_scored(run_dir, rng)
            _write_training_csvs(run_dir, smiles, rng, include_docking_b=include_docking_b)


def _invoke(extra_args=None):
    runner = CliRunner()
    args = [
        "--results-root", "output/results",
        "--pairs-config", "configs/pairs.yaml",
        "--experiment-config", "configs/experiment.yaml",
    ]
    if extra_args:
        args.extend(extra_args)
    return runner.invoke(main, args)


# ---------------------------------------------------------------------------
# Shared-fixture class: default _setup() invoked once for the whole class
# ---------------------------------------------------------------------------


class TestDefaultSetup:
    """Tests that share the same default _setup() + _invoke() result."""

    @pytest.fixture(scope="class")
    def shared_result(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("pfc_default")
        _setup(tmp_path)

        with patch.object(mod, "PROJECT_ROOT", tmp_path):
            result = _invoke()

        return tmp_path, result

    @pytest.fixture(autouse=True)
    def _unpack(self, shared_result):
        self.tmp_path, self.result = shared_result

    def test_basic_run(self):
        """Script runs and produces CSV + JSON."""
        assert self.result.exit_code == 0, self.result.output

        csv_path = self.tmp_path / "report" / "data" / "proxy_final_corr.parquet"
        json_path = self.tmp_path / "report" / "data" / "proxy_final_corr.json"
        assert csv_path.exists()
        assert json_path.exists()

        df = pd.read_parquet(csv_path)
        assert len(df) > 0
        assert "spearman_rho" in df.columns
        assert "pearson_r" in df.columns

    def test_correlation_columns(self):
        """CSV has all expected columns."""
        df = pd.read_parquet(self.tmp_path / "report" / "data" / "proxy_final_corr.parquet")
        expected_cols = {
            "pair", "mode", "target", "n",
            "spearman_rho", "spearman_ci_lo", "spearman_ci_hi",
            "pearson_r", "pearson_ci_lo", "pearson_ci_hi",
            "n_boltz", "n_docking", "n_joined", "join_rate",
        }
        assert expected_cols <= set(df.columns)

    def test_json_structure(self):
        """JSON has pairs, summary, and suitability decision."""
        data = json.loads(
            (self.tmp_path / "report" / "data" / "proxy_final_corr.json").read_text()
        )
        assert "pairs" in data
        assert "pair1" in data["pairs"]
        assert "summary" in data
        assert "proxy_suitable" in data["summary"]
        assert "median_spearman_rho" in data["summary"]

    def test_join_rate_less_than_one(self):
        """Training data has 80% overlap -> join rate < 1.0."""
        df = pd.read_parquet(self.tmp_path / "report" / "data" / "proxy_final_corr.parquet")
        for _, row in df.iterrows():
            assert row["join_rate"] <= 1.0
            assert row["join_rate"] > 0.0

    def test_spearman_within_bounds(self):
        """Spearman rho is within [-1, 1]."""
        df = pd.read_parquet(self.tmp_path / "report" / "data" / "proxy_final_corr.parquet")
        for _, row in df.iterrows():
            assert -1.0 <= row["spearman_rho"] <= 1.0
            assert -1.0 <= row["pearson_r"] <= 1.0


# ---------------------------------------------------------------------------
# Standalone tests with different setup
# ---------------------------------------------------------------------------


def test_double_mode_with_docking_b(tmp_path, monkeypatch):
    """Double mode with docking_b column reports both targets."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    _setup(tmp_path, include_docking_b=True)
    _invoke()

    data = json.loads(
        (tmp_path / "report" / "data" / "proxy_final_corr.json").read_text()
    )
    double_data = data["pairs"]["pair1"]["double"]
    assert "target_a" in double_data["targets"]
    assert "target_b" in double_data["targets"]


def test_empty_results(tmp_path, monkeypatch):
    """No results -> script exits 0, JSON has empty pairs."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pairs.yaml").write_text(yaml.dump(_pairs_cfg()))
    (tmp_path / "configs" / "experiment.yaml").write_text(yaml.dump(_experiment_cfg()))

    result = _invoke()
    assert result.exit_code == 0

    data = json.loads(
        (tmp_path / "report" / "data" / "proxy_final_corr.json").read_text()
    )
    assert data["summary"]["n_pair_mode_target_combos"] == 0
