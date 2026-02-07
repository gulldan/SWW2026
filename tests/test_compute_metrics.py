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
    return {"pairs": {"pair1": {"target_a": {"id": "A"}, "target_b": {"id": "B"}}}}


def _experiment_cfg():
    return {"modes": ["single_a", "single_b", "double"], "seeds": [1, 2, 3, 4, 5]}


def _metrics_cfg():
    return {"thresholds": {"pair1": {"target_a": 0.0, "target_b": 0.0}}}


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
