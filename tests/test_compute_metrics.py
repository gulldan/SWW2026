"""Tests for scripts/compute_metrics.py (new pipeline)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scripts.compute_metrics as mod
import yaml
from click.testing import CliRunner
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


def test_main_computes(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pairs.yaml").write_text(yaml.dump(_pairs_cfg()))
    (tmp_path / "configs" / "experiment.yaml").write_text(yaml.dump(_experiment_cfg()))
    (tmp_path / "configs" / "metrics.yaml").write_text(yaml.dump(_metrics_cfg()))

    rng = np.random.default_rng(42)
    for mode in ("single_a", "single_b", "double"):
        for seed in [1, 2, 3, 4, 5]:
            run_dir = tmp_path / "results" / "pair1" / mode / f"seed_{seed}"
            _write_run(run_dir, rng)

    # Bypass fairness + quality gate checks for this unit test
    import dualdocking.fairness as fairness_mod

    monkeypatch.setattr(fairness_mod, "validate_fairness", lambda _: {"budget_consistent": True, "issues": []})
    monkeypatch.setattr(
        fairness_mod,
        "validate_reward_quality",
        lambda *_args, **_kwargs: {"run_valid": True, "issues": []},
    )

    runner = CliRunner()
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
    assert result.exit_code == 0, result.output

    metrics = json.loads((tmp_path / "report" / "metrics.json").read_text())
    stats = json.loads((tmp_path / "report" / "stats.json").read_text())
    assert "pair1" in metrics
    assert "pair1" in stats

    # hr_b must be present
    for mode in ("single_a", "single_b", "double"):
        assert "hr_b_mean" in metrics["pair1"][mode]
        assert "hr_b_sem" in metrics["pair1"][mode]

    # completeness fields in stats
    assert "_completeness" in stats
    assert stats["_completeness"]["n_valid"] == stats["_completeness"]["n_expected"]


def test_quality_gate_skips_invalid_run(tmp_path, monkeypatch):
    """One seed fails quality gate → skipped, metrics computed from remaining."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pairs.yaml").write_text(yaml.dump(_pairs_cfg()))
    (tmp_path / "configs" / "experiment.yaml").write_text(yaml.dump(_experiment_cfg()))
    (tmp_path / "configs" / "metrics.yaml").write_text(yaml.dump(_metrics_cfg()))

    rng = np.random.default_rng(42)
    for mode in ("single_a", "single_b", "double"):
        for seed in [1, 2, 3, 4, 5]:
            run_dir = tmp_path / "results" / "pair1" / mode / f"seed_{seed}"
            _write_run(run_dir, rng)

    import dualdocking.fairness as fairness_mod

    monkeypatch.setattr(fairness_mod, "validate_fairness", lambda _: {"budget_consistent": True, "issues": []})

    # Seed 1 fails quality gate in all modes
    call_count = {"n": 0}
    original_validate = fairness_mod.validate_reward_quality

    def _mock_qg(results_dir, **kwargs):
        call_count["n"] += 1
        if "seed_1" in str(results_dir):
            return {"run_valid": False, "issues": ["test: invalid fraction too high"]}
        return {"run_valid": True, "issues": []}

    monkeypatch.setattr(fairness_mod, "validate_reward_quality", _mock_qg)

    runner = CliRunner()
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

    runner = CliRunner()
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
    assert result.exit_code == 0, result.output

    metrics = json.loads((tmp_path / "report" / "metrics.json").read_text())
    double_metrics = metrics["pair1"]["double"]
    assert "combined_min_mean" in double_metrics
    assert "combined_min_sem" in double_metrics
    assert "combined_harmonic_mean" in double_metrics
    assert "combined_harmonic_sem" in double_metrics


# ---------------------------------------------------------------------------
# New tests for hr_b, completeness gate, n_valid_runs
# ---------------------------------------------------------------------------


def _setup_full(tmp_path, monkeypatch, seeds=None, missing_seeds=None):
    """Helper: set up configs + runs, optionally omit some seeds."""
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]
    if missing_seeds is None:
        missing_seeds = []
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    (tmp_path / "configs").mkdir()
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

    import dualdocking.fairness as fairness_mod

    monkeypatch.setattr(fairness_mod, "validate_fairness", lambda _: {"budget_consistent": True, "issues": []})
    monkeypatch.setattr(
        fairness_mod,
        "validate_reward_quality",
        lambda *_args, **_kwargs: {"run_valid": True, "issues": []},
    )


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


def test_hr_b_computed(tmp_path, monkeypatch):
    """hr_b_mean / hr_b_sem present and non-None for each mode."""
    _setup_full(tmp_path, monkeypatch)
    result = _invoke()
    assert result.exit_code == 0, result.output

    metrics = json.loads((tmp_path / "report" / "metrics.json").read_text())
    for mode in ("single_a", "single_b", "double"):
        m = metrics["pair1"][mode]
        assert m["hr_b_mean"] is not None
        assert m["hr_b_sem"] is not None
        assert isinstance(m["hr_b_mean"], float)


def test_completeness_gate_fails_on_missing(tmp_path, monkeypatch):
    """Missing 1 seed + --no-allow-incomplete → exit code 1."""
    _setup_full(tmp_path, monkeypatch, missing_seeds=[5])
    result = _invoke(extra_args=["--no-allow-incomplete"])
    assert result.exit_code == 1


def test_completeness_gate_allows_with_flag(tmp_path, monkeypatch):
    """Missing 1 seed + --allow-incomplete (default) → exit code 0."""
    _setup_full(tmp_path, monkeypatch, missing_seeds=[5])
    result = _invoke()  # default is --allow-incomplete
    assert result.exit_code == 0, result.output


def test_n_valid_runs_in_output(tmp_path, monkeypatch):
    """n_valid_runs, n_expected_runs, missing_rate in metrics; _completeness in stats."""
    _setup_full(tmp_path, monkeypatch, missing_seeds=[5])
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
    assert comp["n_expected"] == 15  # 1 pair × 3 modes × 5 seeds
    assert comp["n_valid"] == 12  # 3 missing (one per mode)
    assert comp["missing_rate"] == 0.2
    assert len(comp["missing_runs"]) == 3
