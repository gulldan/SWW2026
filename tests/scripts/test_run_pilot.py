"""Tests for scripts/run_pilot.py (KVA-273 / SCI-09)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts.pipeline.run_pilot import _verify_pilot_artifacts

PROJECT_ROOT_MARKER = "PROJECT_ROOT"


def _create_run_artifacts(
    root: Path,
    pair: str,
    seeds: list[int],
    *,
    skip_scored: str | None = None,
) -> None:
    """Create minimal pilot artifacts for verification testing."""
    rng = np.random.default_rng(42)
    for mode in ("single_a", "single_b", "double"):
        for seed in seeds:
            run_dir = root / "output" / "results" / pair / mode / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # generated.csv
            n = 20
            smiles = [f"C{'C' * i}O" for i in range(n)]
            pd.DataFrame({"smiles": smiles}).to_csv(
                run_dir / "generated.csv", index=False
            )

            # scored_joined.csv
            if skip_scored and skip_scored == f"{mode}/seed_{seed}":
                continue
            pd.DataFrame({
                "smiles_canonical": smiles,
                "affinity_a": rng.normal(-1.0, 0.5, n),
                "affinity_b": rng.normal(-0.5, 0.5, n),
            }).to_csv(run_dir / "scored_joined.csv", index=False)

    # Stage 2 artifacts
    report = root / "report"
    report.mkdir(parents=True, exist_ok=True)
    (report / "metrics.json").write_text(json.dumps({"CDK2_ERK2": {"single_a": {"hr_both_mean": 0.1}}}))
    (report / "stats.json").write_text(json.dumps({"CDK2_ERK2": {"H1a": {"p_value_raw": 0.05}}}))


def test_verify_all_present(tmp_path, monkeypatch):
    """All artifacts present → no issues."""
    import scripts.pipeline.run_pilot as mod

    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    _create_run_artifacts(tmp_path, "CDK2_ERK2", [42, 137])

    issues = _verify_pilot_artifacts("CDK2_ERK2", [42, 137])
    assert issues == []


def test_verify_missing_scored(tmp_path, monkeypatch):
    """Missing scored_joined.csv → reported."""
    import scripts.pipeline.run_pilot as mod

    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    _create_run_artifacts(tmp_path, "CDK2_ERK2", [42, 137], skip_scored="double/seed_137")

    issues = _verify_pilot_artifacts("CDK2_ERK2", [42, 137])
    assert len(issues) == 1
    assert "scored_joined.csv" in issues[0]


def test_verify_missing_metrics(tmp_path, monkeypatch):
    """Missing metrics.json → reported."""
    import scripts.pipeline.run_pilot as mod

    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    _create_run_artifacts(tmp_path, "CDK2_ERK2", [42, 137])
    (tmp_path / "report" / "metrics.json").unlink()

    issues = _verify_pilot_artifacts("CDK2_ERK2", [42, 137])
    assert any("metrics.json" in i for i in issues)


def test_verify_empty_stats(tmp_path, monkeypatch):
    """Empty stats.json → reported."""
    import scripts.pipeline.run_pilot as mod

    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    _create_run_artifacts(tmp_path, "CDK2_ERK2", [42, 137])
    (tmp_path / "report" / "stats.json").write_text("{}")

    issues = _verify_pilot_artifacts("CDK2_ERK2", [42, 137])
    assert any("stats.json" in i for i in issues)
