"""Tests for scripts/compute_metrics.py (new pipeline)."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

import scripts.compute_metrics as mod
from scripts.compute_metrics import main


def _pairs_cfg():
    return {"pairs": {"pair1": {"target_a": {"id": "A", "organism": "Homo sapiens"}, "target_b": {"id": "B", "organism": "Homo sapiens"}}}}


def _experiment_cfg():
    return {"modes": ["single_a", "single_b", "double"], "seeds": [1, 2, 3, 4, 5]}


def _metrics_cfg():
    return {"thresholds": {"pair1": {"target_a": 0.0, "target_b": 0.0}}}


def _experiment_cfg_with_objectives():
    return {
        "modes": ["single_a", "single_b", "double"],
        "seeds": [1, 2, 3, 4, 5],
        "objectives": {
            "baseline": "min",
            "alternatives": [
                {"kind": "harmonic"},
            ],
        },
    }


def _write_run(run_dir: Path, rng: np.random.Generator) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    # minimal generated.csv for fairness gate (monkeypatched in test)
    (run_dir / "generated.csv").write_text("smiles\nCCO\n")

    df = pd.DataFrame(
        {
            "smiles_canonical": [f"C{i}" for i in range(10)],
            "affinity_a": rng.normal(-1.0, 0.1, 10),
            "affinity_b": rng.normal(-1.0, 0.1, 10),
            "is_hit_both": [True] * 10,
            "combined_min": rng.normal(-1.0, 0.1, 10),
            "combined_harmonic": rng.normal(-1.0, 0.1, 10),
        }
    )
    df.to_csv(run_dir / "scored_joined.csv", index=False)


def _setup_full(tmp_path, seeds=None, missing_seeds=None):
    """Helper: set up configs + runs, optionally omit some seeds."""
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]
    if missing_seeds is None:
        missing_seeds = []

    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "configs" / "pairs.yaml").write_text(yaml.dump(_pairs_cfg()))
    (tmp_path / "configs" / "experiment.yaml").write_text(
        yaml.dump({"modes": ["single_a", "single_b", "double"], "seeds": seeds})
    )
    (tmp_path / "configs" / "metrics.yaml").write_text(yaml.dump(_metrics_cfg()))

    rng = np.random.default_rng(42)
    for mode in ("single_a", "single_b", "double"):
        for seed in seeds:
            if seed in missing_seeds:
                continue
            run_dir = tmp_path / "results" / "pair1" / mode / f"seed_{seed}"
            _write_run(run_dir, rng)


def _invoke(extra_args=None):
    runner = CliRunner()
    args = [
        "--results-root", "results",
        "--pairs-config", "configs/pairs.yaml",
        "--experiment-config", "configs/experiment.yaml",
        "--metrics-config", "configs/metrics.yaml",
    ]
    if extra_args:
        args.extend(extra_args)
    return runner.invoke(main, args)


# ---------------------------------------------------------------------------
# Shared-fixture class: default _setup_full() invoked once for the whole class
# ---------------------------------------------------------------------------


class TestDefaultFullSetup:
    """Tests that share the same _setup_full(default seeds) + _invoke() result."""

    @pytest.fixture(scope="class")
    def shared_result(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("cm_default")
        _setup_full(tmp_path)

        import dualdocking.fairness as fairness_mod

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

    def test_main_computes(self):
        assert self.result.exit_code == 0, self.result.output

        metrics = json.loads((self.tmp_path / "report" / "metrics.json").read_text())
        stats = json.loads((self.tmp_path / "report" / "stats.json").read_text())
        assert "pair1" in metrics
        assert "pair1" in stats

        # hr_b must be present
        for mode in ("single_a", "single_b", "double"):
            assert "hr_b_mean" in metrics["pair1"][mode]
            assert "hr_b_sem" in metrics["pair1"][mode]

        # completeness fields in stats
        assert "_completeness" in stats
        assert stats["_completeness"]["n_valid"] == stats["_completeness"]["n_expected"]

    def test_hr_b_computed(self):
        """hr_b_mean / hr_b_sem present and non-None for each mode."""
        assert self.result.exit_code == 0, self.result.output

        metrics = json.loads((self.tmp_path / "report" / "metrics.json").read_text())
        for mode in ("single_a", "single_b", "double"):
            m = metrics["pair1"][mode]
            assert m["hr_b_mean"] is not None
            assert m["hr_b_sem"] is not None
            assert isinstance(m["hr_b_mean"], float)

    def test_degradation_guard_in_stats(self):
        """_degradation block in stats.json contains per-target Cliff's delta."""
        assert self.result.exit_code == 0, self.result.output

        stats = json.loads((self.tmp_path / "report" / "stats.json").read_text())
        assert "_degradation" in stats
        deg = stats["_degradation"]
        assert "pair1" in deg

        pair_deg = deg["pair1"]
        assert "hr_a_double_vs_single_a" in pair_deg
        assert "hr_b_double_vs_single_b" in pair_deg

        for check in pair_deg.values():
            assert "cliffs_delta" in check
            assert "degraded" in check
            assert isinstance(check["cliffs_delta"], float)
            assert isinstance(check["degraded"], bool)

    def test_is_hit_a_b_columns_created(self):
        """_load_scored_csv adds is_hit_a and is_hit_b columns."""
        assert self.result.exit_code == 0, self.result.output

        # Verify by reading a scored CSV and checking the load function
        run_dir = self.tmp_path / "results" / "pair1" / "double" / "seed_1"
        df = mod._load_scored_csv(run_dir / "scored_joined.csv", 0.0, 0.0)
        assert "is_hit_a" in df.columns
        assert "is_hit_b" in df.columns
        assert "is_hit_both" in df.columns


# ---------------------------------------------------------------------------
# Standalone tests with different setups
# ---------------------------------------------------------------------------


def test_quality_gate_skips_invalid_run(tmp_path, monkeypatch):
    """One seed fails quality gate -> skipped, metrics computed from remaining."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    _setup_full(tmp_path)

    import dualdocking.fairness as fairness_mod

    monkeypatch.setattr(fairness_mod, "validate_fairness", lambda _: {"budget_consistent": True, "issues": []})

    # Seed 1 fails quality gate in all modes
    call_count = {"n": 0}

    def _mock_qg(results_dir, **kwargs):
        call_count["n"] += 1
        if "seed_1" in str(results_dir):
            return {"run_valid": False, "issues": ["test: invalid fraction too high"]}
        return {"run_valid": True, "issues": []}

    monkeypatch.setattr(fairness_mod, "validate_reward_quality", _mock_qg)

    result = _invoke()
    assert result.exit_code == 0, result.output

    metrics = json.loads((tmp_path / "report" / "metrics.json").read_text())
    assert "pair1" in metrics
    # Quality gate was actually called
    assert call_count["n"] > 0

    # n_valid_runs < n_expected_runs for modes where seed_1 was skipped
    for mode in ("single_a", "single_b", "double"):
        mode_m = metrics["pair1"][mode]
        assert mode_m["n_valid_runs"] < mode_m["n_expected_runs"]


def test_main_missing_input(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pairs.yaml").write_text(yaml.dump(_pairs_cfg()))
    (tmp_path / "configs" / "experiment.yaml").write_text(yaml.dump(_experiment_cfg()))
    (tmp_path / "configs" / "metrics.yaml").write_text(yaml.dump(_metrics_cfg()))

    runner = CliRunner()
    # No results -> should still exit 0 but produce empty outputs
    result = runner.invoke(
        main,
        [
            "--results-root",
            "results",
            "--pairs-config",
            "configs/pairs.yaml",
            "--experiment-config",
            "configs/experiment.yaml",
            "--metrics-config",
            "configs/metrics.yaml",
        ],
    )
    assert result.exit_code == 0


def test_objective_means_in_metrics(tmp_path, monkeypatch):
    """Verify combined_*_mean and combined_*_sem keys in metrics output."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pairs.yaml").write_text(yaml.dump(_pairs_cfg()))
    (tmp_path / "configs" / "experiment.yaml").write_text(yaml.dump(_experiment_cfg_with_objectives()))
    (tmp_path / "configs" / "metrics.yaml").write_text(yaml.dump(_metrics_cfg()))

    rng = np.random.default_rng(42)
    for mode in ("single_a", "single_b", "double"):
        for seed in [1, 2, 3, 4, 5]:
            run_dir = tmp_path / "results" / "pair1" / mode / f"seed_{seed}"
            _write_run(run_dir, rng)

    import dualdocking.fairness as fairness_mod

    monkeypatch.setattr(fairness_mod, "validate_fairness", lambda _: {"budget_consistent": True, "issues": []})
    monkeypatch.setattr(
        fairness_mod,
        "validate_reward_quality",
        lambda *_args, **_kwargs: {"run_valid": True, "issues": []},
    )

    result = _invoke()
    assert result.exit_code == 0, result.output

    metrics = json.loads((tmp_path / "report" / "metrics.json").read_text())
    double_metrics = metrics["pair1"]["double"]
    assert "combined_min_mean" in double_metrics
    assert "combined_min_sem" in double_metrics
    assert "combined_harmonic_mean" in double_metrics
    assert "combined_harmonic_sem" in double_metrics


def test_completeness_gate_fails_on_missing(tmp_path, monkeypatch):
    """Missing 1 seed + --no-allow-incomplete -> exit code 1."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    _setup_full(tmp_path, missing_seeds=[5])

    import dualdocking.fairness as fairness_mod

    monkeypatch.setattr(fairness_mod, "validate_fairness", lambda _: {"budget_consistent": True, "issues": []})
    monkeypatch.setattr(
        fairness_mod,
        "validate_reward_quality",
        lambda *_args, **_kwargs: {"run_valid": True, "issues": []},
    )

    result = _invoke(extra_args=["--no-allow-incomplete"])
    assert result.exit_code == 1


def test_completeness_gate_allows_with_flag(tmp_path, monkeypatch):
    """Missing 1 seed + --allow-incomplete (default) -> exit code 0."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    _setup_full(tmp_path, missing_seeds=[5])

    import dualdocking.fairness as fairness_mod

    monkeypatch.setattr(fairness_mod, "validate_fairness", lambda _: {"budget_consistent": True, "issues": []})
    monkeypatch.setattr(
        fairness_mod,
        "validate_reward_quality",
        lambda *_args, **_kwargs: {"run_valid": True, "issues": []},
    )

    result = _invoke()  # default is --allow-incomplete
    assert result.exit_code == 0, result.output


def test_n_valid_runs_in_output(tmp_path, monkeypatch):
    """n_valid_runs, n_expected_runs, missing_rate in metrics; _completeness in stats."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    _setup_full(tmp_path, missing_seeds=[5])

    import dualdocking.fairness as fairness_mod

    monkeypatch.setattr(fairness_mod, "validate_fairness", lambda _: {"budget_consistent": True, "issues": []})
    monkeypatch.setattr(
        fairness_mod,
        "validate_reward_quality",
        lambda *_args, **_kwargs: {"run_valid": True, "issues": []},
    )

    result = _invoke()
    assert result.exit_code == 0, result.output

    metrics = json.loads((tmp_path / "report" / "metrics.json").read_text())
    stats = json.loads((tmp_path / "report" / "stats.json").read_text())

    for mode in ("single_a", "single_b", "double"):
        m = metrics["pair1"][mode]
        assert m["n_valid_runs"] == 4
        assert m["n_expected_runs"] == 5
        assert m["missing_rate"] == 0.2

    assert "_completeness" in stats
    comp = stats["_completeness"]
    assert comp["n_expected"] == 15  # 1 pair x 3 modes x 5 seeds
    assert comp["n_valid"] == 12  # 3 missing (one per mode)
    assert comp["missing_rate"] == 0.2
    assert len(comp["missing_runs"]) == 3


def test_fdr_scope_primary_only(tmp_path, monkeypatch):
    """FDR correction only applies to primary family (H1a/H1b core pairs, SAP 4.2)."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    # Two pairs: one core, one auxiliary
    pairs_cfg = {
        "pairs": {
            "core_pair": {
                "set": "core",
                "target_a": {"id": "A", "organism": "Homo sapiens"},
                "target_b": {"id": "B", "organism": "Homo sapiens"},
            },
            "aux_pair": {
                "set": "auxiliary",
                "target_a": {"id": "C", "organism": "Homo sapiens"},
                "target_b": {"id": "D", "organism": "Homo sapiens"},
            },
        }
    }
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pairs.yaml").write_text(yaml.dump(pairs_cfg))
    (tmp_path / "configs" / "experiment.yaml").write_text(yaml.dump(_experiment_cfg()))
    (tmp_path / "configs" / "metrics.yaml").write_text(yaml.dump({
        "thresholds": {
            "core_pair": {"target_a": 0.0, "target_b": 0.0},
            "aux_pair": {"target_a": 0.0, "target_b": 0.0},
        }
    }))

    rng = np.random.default_rng(42)
    for pair_id in ("core_pair", "aux_pair"):
        for mode in ("single_a", "single_b", "double"):
            for seed in [1, 2, 3, 4, 5]:
                run_dir = tmp_path / "results" / pair_id / mode / f"seed_{seed}"
                _write_run(run_dir, rng)

    import dualdocking.fairness as fairness_mod

    monkeypatch.setattr(fairness_mod, "validate_fairness", lambda _: {"budget_consistent": True, "issues": []})
    monkeypatch.setattr(
        fairness_mod,
        "validate_reward_quality",
        lambda *_args, **_kwargs: {"run_valid": True, "issues": []},
    )

    result = _invoke()
    assert result.exit_code == 0, result.output

    stats = json.loads((tmp_path / "report" / "stats.json").read_text())

    # Core pair H1a/H1b: must have p_value_adjusted (FDR-corrected float)
    for test_id in ("H1a", "H1b"):
        entry = stats["core_pair"][test_id]
        assert entry["p_value_adjusted"] is not None, f"core {test_id} should be FDR-corrected"
        assert isinstance(entry["p_value_adjusted"], float)

    # Core pair H2a/H2b: must NOT be FDR-corrected (secondary, not in primary family)
    for test_id in ("H2a", "H2b"):
        entry = stats["core_pair"][test_id]
        assert entry["p_value_adjusted"] is None, f"core {test_id} should not be in primary FDR family"
        assert entry["significant"] is False

    # Auxiliary pair: all tests must NOT be FDR-corrected
    for test_id in ("H1a", "H1b", "H2a", "H2b"):
        entry = stats["aux_pair"][test_id]
        assert entry["p_value_adjusted"] is None, f"aux {test_id} should not be FDR-corrected"
        assert entry["significant"] is False
