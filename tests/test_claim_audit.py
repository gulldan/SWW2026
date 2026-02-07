"""Tests for scripts/claim_audit.py (KVA-274 / SCI-10)."""

import json

import scripts.claim_audit as mod
from click.testing import CliRunner
from scripts.claim_audit import main


def _make_stats(
    *,
    n_significant: int = 5,
    flip_reported: str | None = None,
) -> dict:
    """Build a stats.json with controllable significance."""
    stats: dict = {}
    pairs = ["CDK2_ERK2", "AChE_BChE", "BACE1_GSK3B", "AChE_BACE1", "GSK3B_CDK5"]
    for i, pair in enumerate(pairs):
        sig = i < n_significant
        for test_id in ("H1a", "H1b"):
            if pair not in stats:
                stats[pair] = {}
            reported = sig
            if flip_reported == f"{pair}/{test_id}":
                reported = not sig
            stats[pair][test_id] = {
                "comparison": "double_vs_single_a" if test_id == "H1a" else "double_vs_single_b",
                "metric": "hr_both",
                "observed_diff": 0.05 if sig else -0.01,
                "p_value_raw": 0.01 if sig else 0.5,
                "p_value_adjusted": 0.02 if sig else 0.6,
                "cliffs_delta": 0.5 if sig else 0.1,
                "cohens_d": 0.8 if sig else 0.1,
                "ci_lower": 0.01 if sig else -0.05,
                "ci_upper": 0.10 if sig else 0.05,
                "significant": reported,
            }
    stats["_completeness"] = {"n_expected": 75, "n_valid": 75, "missing_rate": 0.0}
    return stats


def _make_metrics() -> dict:
    return {"CDK2_ERK2": {"single_a": {"hr_both_mean": 0.1}}}


def _make_sensitivity(*, all_pass: bool = True) -> dict:
    return {
        "thresholds_tested": [-1.0, 0.0, 1.0],
        "robustness_summary": {
            "-1.0": {"n_core_significant": 3 if all_pass else 1, "passes_success_criterion": all_pass},
            "0.0": {"n_core_significant": 4, "passes_success_criterion": True},
            "1.0": {"n_core_significant": 3 if all_pass else 2, "passes_success_criterion": all_pass},
        },
    }


def _setup(tmp_path, monkeypatch, *, n_significant=5, sensitivity_pass=True, flip_reported=None):
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    report = tmp_path / "report"
    report.mkdir(parents=True, exist_ok=True)
    (report / "stats.json").write_text(json.dumps(_make_stats(
        n_significant=n_significant, flip_reported=flip_reported
    )))
    (report / "metrics.json").write_text(json.dumps(_make_metrics()))
    (report / "data").mkdir(parents=True, exist_ok=True)
    (report / "data" / "threshold_sensitivity.json").write_text(
        json.dumps(_make_sensitivity(all_pass=sensitivity_pass))
    )


def test_full_success(tmp_path, monkeypatch):
    """5/5 significant + threshold robust → 6.1_full_success."""
    _setup(tmp_path, monkeypatch, n_significant=5, sensitivity_pass=True)
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0, result.output
    audit = json.loads((tmp_path / "report" / "data" / "claim_audit.json").read_text())
    assert audit["pass"] is True
    assert audit["claim_scenario"] == "6.1_full_success"


def test_partial_success(tmp_path, monkeypatch):
    """3/5 significant but not threshold robust → 6.2_partial_success."""
    _setup(tmp_path, monkeypatch, n_significant=3, sensitivity_pass=False)
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0, result.output
    audit = json.loads((tmp_path / "report" / "data" / "claim_audit.json").read_text())
    assert audit["claim_scenario"] == "6.2_partial_success"


def test_no_evidence(tmp_path, monkeypatch):
    """0/5 significant → 6.5_no_evidence."""
    _setup(tmp_path, monkeypatch, n_significant=0)
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0, result.output
    audit = json.loads((tmp_path / "report" / "data" / "claim_audit.json").read_text())
    assert audit["claim_scenario"] == "6.5_no_evidence"


def test_significance_mismatch_detected(tmp_path, monkeypatch):
    """Flipped significance flag → audit detects mismatch."""
    _setup(tmp_path, monkeypatch, n_significant=5, flip_reported="CDK2_ERK2/H1a")
    runner = CliRunner()
    result = runner.invoke(main, ["--no-strict"])
    assert result.exit_code == 0
    audit = json.loads((tmp_path / "report" / "data" / "claim_audit.json").read_text())
    assert audit["pass"] is False
    assert any("Significance mismatch" in i for i in audit["issues"])


def test_missing_stats_strict(tmp_path, monkeypatch):
    """Missing stats.json in strict mode → exit 1."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    (tmp_path / "report").mkdir(parents=True, exist_ok=True)
    (tmp_path / "report" / "data").mkdir(parents=True, exist_ok=True)
    runner = CliRunner()
    result = runner.invoke(main, ["--strict"])
    assert result.exit_code == 1


def test_weak_evidence(tmp_path, monkeypatch):
    """2/5 significant but 3+ positive direction → 6.3_weak_evidence."""
    _setup(tmp_path, monkeypatch, n_significant=2)
    # Patch stats to have 3+ positive direction
    stats_path = tmp_path / "report" / "stats.json"
    stats = json.loads(stats_path.read_text())
    # Make pairs 0-2 positive direction (pair 0,1 are significant, pair 2 is not but positive)
    stats["BACE1_GSK3B"]["H1a"]["observed_diff"] = 0.02
    stats_path.write_text(json.dumps(stats))

    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0
    audit = json.loads((tmp_path / "report" / "data" / "claim_audit.json").read_text())
    assert audit["claim_scenario"] == "6.3_weak_evidence"
