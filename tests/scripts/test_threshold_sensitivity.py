"""Tests for scripts/threshold_sensitivity.py (KVA-270 / SCI-06)."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import scripts.s5_analyze.threshold_sensitivity as mod
import yaml
from click.testing import CliRunner
from scripts.s5_analyze.threshold_sensitivity import main


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
    return {"modes": ["single_a", "single_b", "double"], "seeds": [1, 2, 3, 4, 5]}


def _metrics_cfg():
    return {
        "thresholds": {"pair1": {"target_a": 0.0, "target_b": 0.0}},
        "sensitivity_thresholds": [-1.0, 0.0, 1.0],
    }


def _write_run(
    run_dir: Path, rng: np.random.Generator, *, mean: float = -1.0
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "generated.csv").write_text("smiles\nCCO\n")
    df = pd.DataFrame(
        {
            "smiles_canonical": [f"C{i}" for i in range(10)],
            "affinity_a": rng.normal(mean, 0.5, 10),
            "affinity_b": rng.normal(mean, 0.5, 10),
        }
    )
    df.to_csv(run_dir / "scored_joined.csv", index=False)
    df.to_parquet(run_dir / "scored_joined.parquet", index=False)


def _setup(tmp_path, *, double_mean=-1.5, single_mean=-0.5):
    """Set up configs + runs with controllable means for double vs single."""
    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "configs" / "pairs.yaml").write_text(yaml.dump(_pairs_cfg()))
    (tmp_path / "configs" / "experiment.yaml").write_text(yaml.dump(_experiment_cfg()))
    (tmp_path / "configs" / "metrics.yaml").write_text(yaml.dump(_metrics_cfg()))

    rng = np.random.default_rng(42)
    for mode in ("single_a", "single_b", "double"):
        mean = double_mean if mode == "double" else single_mean
        for seed in [1, 2, 3, 4, 5]:
            run_dir = tmp_path / "output" / "results" / "pair1" / mode / f"seed_{seed}"
            _write_run(run_dir, rng, mean=mean)


def _invoke(extra_args=None):
    runner = CliRunner()
    args = [
        "--results-root", "output/results",
        "--pairs-config", "configs/pairs.yaml",
        "--experiment-config", "configs/experiment.yaml",
        "--metrics-config", "configs/metrics.yaml",
    ]
    if extra_args:
        args.extend(extra_args)
    return runner.invoke(main, args)


# ---------------------------------------------------------------------------
# Shared-fixture class: default _setup() invoked once for the whole class
# ---------------------------------------------------------------------------


class TestDefaultSetup:
    """Tests that share the same _setup(default) + _invoke() result."""

    @pytest.fixture(scope="class")
    def shared_result(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("ts_default")
        _setup(tmp_path)

        import dualdocking.analysis.fairness as fairness_mod

        with (
            patch.object(mod, "PROJECT_ROOT", tmp_path),
            patch.object(fairness_mod, "validate_fairness", return_value={"budget_consistent": True, "issues": []}),
            patch.object(fairness_mod, "validate_reward_quality", return_value={"run_valid": True, "issues": []}),
        ):
            result = _invoke()

        return tmp_path, result

    @pytest.fixture(autouse=True)
    def _unpack(self, shared_result):
        self.tmp_path, self.result = shared_result

    def test_basic_run(self):
        """Script runs and produces both CSV and JSON outputs."""
        assert self.result.exit_code == 0, self.result.output

        csv_path = self.tmp_path / "report" / "data" / "threshold_sensitivity.parquet"
        json_path = self.tmp_path / "report" / "data" / "threshold_sensitivity.json"
        assert csv_path.exists()
        assert json_path.exists()

        df = pd.read_parquet(csv_path)
        assert len(df) > 0
        assert "threshold" in df.columns
        assert "pair" in df.columns
        assert "hr_both_mean" in df.columns

    def test_all_thresholds_present(self):
        """All 3 pre-registered thresholds appear in output."""
        assert self.result.exit_code == 0, self.result.output

        data = json.loads(
            (self.tmp_path / "report" / "data" / "threshold_sensitivity.json").read_text()
        )
        assert set(data["thresholds_tested"]) == {-1.0, 0.0, 1.0}
        for thr in [-1.0, 0.0, 1.0]:
            assert str(thr) in data["results"]

    def test_robustness_summary_present(self):
        """JSON output includes robustness_summary with pass/fail per threshold."""
        assert self.result.exit_code == 0, self.result.output

        data = json.loads(
            (self.tmp_path / "report" / "data" / "threshold_sensitivity.json").read_text()
        )
        assert "robustness_summary" in data
        for thr_key in ["-1.0", "0.0", "1.0"]:
            s = data["robustness_summary"][thr_key]
            assert "n_core_significant" in s
            assert "n_core_total" in s
            assert "passes_success_criterion" in s

    def test_csv_has_all_modes(self):
        """CSV rows cover all 3 modes for each pair-threshold combo."""
        df = pd.read_parquet(self.tmp_path / "report" / "data" / "threshold_sensitivity.parquet")
        for thr in [-1.0, 0.0, 1.0]:
            modes_present = set(df[df["threshold"] == thr]["mode"].unique())
            assert {"single_a", "single_b", "double"} == modes_present

    def test_statistical_tests_in_json(self):
        """H1a and H1b tests are present per pair per threshold."""
        data = json.loads(
            (self.tmp_path / "report" / "data" / "threshold_sensitivity.json").read_text()
        )
        for thr_key in ["-1.0", "0.0", "1.0"]:
            pair_data = data["results"][thr_key]["pair1"]
            assert "H1a" in pair_data["stats"]
            assert "H1b" in pair_data["stats"]
            for test_id in ("H1a", "H1b"):
                s = pair_data["stats"][test_id]
                assert "observed_diff" in s
                assert "p_value_adjusted" in s
                assert "cliffs_delta" in s
                assert "significant" in s


# ---------------------------------------------------------------------------
# Standalone tests with different setup
# ---------------------------------------------------------------------------


def test_hit_rate_varies_with_threshold(tmp_path, monkeypatch):
    """Stricter threshold (-1.0) yields lower hr_both than lenient (+1.0)."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    _setup(tmp_path, double_mean=-0.5)

    import dualdocking.analysis.fairness as fairness_mod

    monkeypatch.setattr(
        fairness_mod, "validate_fairness",
        lambda _: {"budget_consistent": True, "issues": []},
    )
    monkeypatch.setattr(
        fairness_mod, "validate_reward_quality",
        lambda *_a, **_kw: {"run_valid": True, "issues": []},
    )

    _invoke()

    data = json.loads(
        (tmp_path / "report" / "data" / "threshold_sensitivity.json").read_text()
    )
    hr_strict = data["results"]["-1.0"]["pair1"]["metrics"]["double"]["hr_both_mean"]
    hr_lenient = data["results"]["1.0"]["pair1"]["metrics"]["double"]["hr_both_mean"]
    assert hr_strict <= hr_lenient


def test_empty_results(tmp_path, monkeypatch):
    """No results dir -> script exits 0, JSON has empty results."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pairs.yaml").write_text(yaml.dump(_pairs_cfg()))
    (tmp_path / "configs" / "experiment.yaml").write_text(yaml.dump(_experiment_cfg()))
    (tmp_path / "configs" / "metrics.yaml").write_text(yaml.dump(_metrics_cfg()))

    result = _invoke()
    assert result.exit_code == 0

    data = json.loads(
        (tmp_path / "report" / "data" / "threshold_sensitivity.json").read_text()
    )
    assert data["thresholds_tested"] == [-1.0, 0.0, 1.0]
