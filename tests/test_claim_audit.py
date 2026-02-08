"""Tests for scripts/claim_audit.py (KVA-274 / SCI-10)."""

import json

import yaml
from click.testing import CliRunner

import scripts.claim_audit as mod
from scripts.claim_audit import main

_CORE_PAIRS = ["CDK2_ERK2", "AChE_BChE", "BACE1_GSK3B", "AChE_BACE1", "GSK3B_CDK5"]


def _make_pairs_yaml() -> dict:
    """Minimal pairs.yaml with core pairs for testing."""
    pairs: dict = {}
    for pid in _CORE_PAIRS:
        a, b = pid.split("_", 1)
        pairs[pid] = {
            "set": "core",
            "target_a": {"id": a, "organism": "Homo sapiens"},
            "target_b": {"id": b, "organism": "Homo sapiens"},
        }
    return {"pairs": pairs}


def _make_stats(
    *,
    n_significant: int = 5,
    flip_reported: str | None = None,
    degradation: dict | None = None,
    missing_rate: float = 0.0,
) -> dict:
    """Build a stats.json with controllable significance."""
    stats: dict = {}
    pairs = _CORE_PAIRS
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

    # Degradation block (SAP 5.2)
    if degradation is not None:
        stats["_degradation"] = degradation
    else:
        # Default: no degradation on any pair
        deg: dict = {}
        for pair in pairs:
            deg[pair] = {
                "hr_a_double_vs_single_a": {"cliffs_delta": 0.1, "degraded": False},
                "hr_b_double_vs_single_b": {"cliffs_delta": 0.2, "degraded": False},
            }
        stats["_degradation"] = deg

    n_expected = 75
    n_valid = round(n_expected * (1.0 - missing_rate))
    stats["_completeness"] = {
        "n_expected": n_expected,
        "n_valid": n_valid,
        "missing_rate": missing_rate,
    }
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


def _setup(tmp_path, monkeypatch, *, n_significant=5, sensitivity_pass=True,
           flip_reported=None, degradation=None, missing_rate=0.0):
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    # Write pairs.yaml so _load_core_pairs can read it
    configs = tmp_path / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "pairs.yaml").write_text(yaml.dump(_make_pairs_yaml()))
    report = tmp_path / "report"
    report.mkdir(parents=True, exist_ok=True)
    (report / "stats.json").write_text(json.dumps(_make_stats(
        n_significant=n_significant, flip_reported=flip_reported,
        degradation=degradation, missing_rate=missing_rate,
    )))
    (report / "metrics.json").write_text(json.dumps(_make_metrics()))
    (report / "data").mkdir(parents=True, exist_ok=True)
    (report / "data" / "threshold_sensitivity.json").write_text(
        json.dumps(_make_sensitivity(all_pass=sensitivity_pass))
    )


def test_full_success(tmp_path, monkeypatch):
    """5/5 significant + threshold robust + no degradation -> 6.1_full_success."""
    _setup(tmp_path, monkeypatch, n_significant=5, sensitivity_pass=True)
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0, result.output
    audit = json.loads((tmp_path / "report" / "data" / "claim_audit.json").read_text())
    assert audit["pass"] is True
    assert audit["claim_scenario"] == "6.1_full_success"
    assert audit["degradation_warning"] is None
    assert audit["evidence_map"] is not None
    assert "report/stats.json" in audit["evidence_map"]["evidence"][0]["artifact"]


def test_partial_success(tmp_path, monkeypatch):
    """3/5 significant but not threshold robust -> 6.2_partial_success."""
    _setup(tmp_path, monkeypatch, n_significant=3, sensitivity_pass=False)
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0, result.output
    audit = json.loads((tmp_path / "report" / "data" / "claim_audit.json").read_text())
    assert audit["claim_scenario"] == "6.2_partial_success"
    assert audit["evidence_map"] is not None


def test_no_evidence(tmp_path, monkeypatch):
    """0/5 significant -> 6.5_no_evidence."""
    _setup(tmp_path, monkeypatch, n_significant=0)
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0, result.output
    audit = json.loads((tmp_path / "report" / "data" / "claim_audit.json").read_text())
    assert audit["claim_scenario"] == "6.5_no_evidence"


def test_significance_mismatch_detected(tmp_path, monkeypatch):
    """Flipped significance flag -> audit detects mismatch."""
    _setup(tmp_path, monkeypatch, n_significant=5, flip_reported="CDK2_ERK2/H1a")
    runner = CliRunner()
    result = runner.invoke(main, ["--no-strict"])
    assert result.exit_code == 0
    audit = json.loads((tmp_path / "report" / "data" / "claim_audit.json").read_text())
    assert audit["pass"] is False
    assert any("Significance mismatch" in i for i in audit["issues"])


def test_missing_stats_strict(tmp_path, monkeypatch):
    """Missing stats.json in strict mode -> exit 1."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "pairs.yaml").write_text(yaml.dump(_make_pairs_yaml()))
    (tmp_path / "report").mkdir(parents=True, exist_ok=True)
    (tmp_path / "report" / "data").mkdir(parents=True, exist_ok=True)
    runner = CliRunner()
    result = runner.invoke(main, ["--strict"])
    assert result.exit_code == 1


def test_weak_evidence(tmp_path, monkeypatch):
    """2/5 significant but 3+ positive direction -> 6.3_weak_evidence."""
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


def test_degradation_detected(tmp_path, monkeypatch):
    """Per-target degradation -> 6.4 warning appended, audit fails."""
    degradation = {
        "CDK2_ERK2": {
            "hr_a_double_vs_single_a": {"cliffs_delta": -0.6, "degraded": True},
            "hr_b_double_vs_single_b": {"cliffs_delta": 0.1, "degraded": False},
        },
        "AChE_BChE": {
            "hr_a_double_vs_single_a": {"cliffs_delta": 0.2, "degraded": False},
            "hr_b_double_vs_single_b": {"cliffs_delta": 0.3, "degraded": False},
        },
        "BACE1_GSK3B": {
            "hr_a_double_vs_single_a": {"cliffs_delta": 0.1, "degraded": False},
            "hr_b_double_vs_single_b": {"cliffs_delta": 0.1, "degraded": False},
        },
        "AChE_BACE1": {
            "hr_a_double_vs_single_a": {"cliffs_delta": 0.1, "degraded": False},
            "hr_b_double_vs_single_b": {"cliffs_delta": 0.1, "degraded": False},
        },
        "GSK3B_CDK5": {
            "hr_a_double_vs_single_a": {"cliffs_delta": 0.1, "degraded": False},
            "hr_b_double_vs_single_b": {"cliffs_delta": 0.1, "degraded": False},
        },
    }
    _setup(tmp_path, monkeypatch, n_significant=5, degradation=degradation)
    runner = CliRunner()
    result = runner.invoke(main, ["--no-strict"])
    assert result.exit_code == 0
    audit = json.loads((tmp_path / "report" / "data" / "claim_audit.json").read_text())
    assert audit["pass"] is False
    assert audit["degradation_warning"] is not None
    assert audit["degradation_warning"]["scenario"] == "6.4_degradation_detected"
    assert any("CDK2_ERK2" in p for p in audit["degradation_warning"]["pairs"])
    assert any("Degradation" in i for i in audit["issues"])
    # CDK2_ERK2 excluded from success count (SAP 5.2): 5 - 1 = 4, still >= 3
    assert audit["summary"]["n_core_significant"] == 4
    assert audit["summary"]["success_criterion_met"] is True
    assert "CDK2_ERK2" in audit["summary"]["degraded_excluded"]
    # Primary scenario still determined (6.4 is addendum)
    assert audit["claim_scenario"] == "6.1_full_success"


def test_insufficient_data(tmp_path, monkeypatch):
    """Missing rate > 20% -> 6.6_insufficient_data overrides everything."""
    _setup(tmp_path, monkeypatch, n_significant=5, missing_rate=0.30)
    runner = CliRunner()
    result = runner.invoke(main, ["--no-strict"])
    assert result.exit_code == 0
    audit = json.loads((tmp_path / "report" / "data" / "claim_audit.json").read_text())
    assert audit["claim_scenario"] == "6.6_insufficient_data"
    assert audit["evidence_map"] is not None
    assert any("Missing data rate" in i for i in audit["issues"])
    check = next(c for c in audit["checks"] if c["name"] == "completeness_gate")
    assert check["pass"] is False


def test_no_degradation_block_still_passes(tmp_path, monkeypatch):
    """When _degradation is missing from stats.json, guard passes (backward compat)."""
    _setup(tmp_path, monkeypatch, n_significant=5)
    stats_path = tmp_path / "report" / "stats.json"
    stats = json.loads(stats_path.read_text())
    del stats["_degradation"]
    stats_path.write_text(json.dumps(stats))

    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0
    audit = json.loads((tmp_path / "report" / "data" / "claim_audit.json").read_text())
    check = next(c for c in audit["checks"] if c["name"] == "no_degradation_guard")
    assert check["pass"] is True


def test_degradation_excludes_from_success_count(tmp_path, monkeypatch):
    """Degradation exclusion flips outcome: 3/5 sig - 1 degraded = 2/5 -> no longer success."""
    degradation = {
        "CDK2_ERK2": {
            "hr_a_double_vs_single_a": {"cliffs_delta": -0.6, "degraded": True},
            "hr_b_double_vs_single_b": {"cliffs_delta": 0.1, "degraded": False},
        },
        "AChE_BChE": {
            "hr_a_double_vs_single_a": {"cliffs_delta": 0.2, "degraded": False},
            "hr_b_double_vs_single_b": {"cliffs_delta": 0.3, "degraded": False},
        },
        "BACE1_GSK3B": {
            "hr_a_double_vs_single_a": {"cliffs_delta": 0.1, "degraded": False},
            "hr_b_double_vs_single_b": {"cliffs_delta": 0.1, "degraded": False},
        },
        "AChE_BACE1": {
            "hr_a_double_vs_single_a": {"cliffs_delta": 0.1, "degraded": False},
            "hr_b_double_vs_single_b": {"cliffs_delta": 0.1, "degraded": False},
        },
        "GSK3B_CDK5": {
            "hr_a_double_vs_single_a": {"cliffs_delta": 0.1, "degraded": False},
            "hr_b_double_vs_single_b": {"cliffs_delta": 0.1, "degraded": False},
        },
    }
    # 3 significant, CDK2_ERK2 is one of them AND degraded -> excluded -> 2/5
    _setup(tmp_path, monkeypatch, n_significant=3, degradation=degradation)
    runner = CliRunner()
    result = runner.invoke(main, ["--no-strict"])
    assert result.exit_code == 0
    audit = json.loads((tmp_path / "report" / "data" / "claim_audit.json").read_text())
    # CDK2_ERK2 excluded: 3 - 1 = 2 significant, below threshold of 3
    assert audit["summary"]["n_core_significant"] == 2
    assert audit["summary"]["success_criterion_met"] is False
    assert "CDK2_ERK2" in audit["summary"]["degraded_excluded"]
    # Scenario should reflect the reduced count (not 6.1/6.2)
    assert audit["claim_scenario"] in ("6.3_weak_evidence", "6.5_no_evidence")


def test_evidence_map_populated(tmp_path, monkeypatch):
    """Evidence map links claim scenario to artifacts and SAP sections."""
    _setup(tmp_path, monkeypatch, n_significant=0)
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 0
    audit = json.loads((tmp_path / "report" / "data" / "claim_audit.json").read_text())
    assert audit["claim_scenario"] == "6.5_no_evidence"
    emap = audit["evidence_map"]
    assert emap is not None
    assert "claim_template" in emap
    assert "evidence" in emap
    assert "sap_sections" in emap
