"""Unit tests for dualdocking.validation.audit module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from dualdocking.validation.audit import (
    CLIFFS_DELTA_THRESHOLD,
    DEGRADATION_DELTA_THRESHOLD,
    EVIDENCE_MAP,
    MISSING_DATA_THRESHOLD,
    P_THRESHOLD,
    SUCCESS_MIN_PAIRS,
    AuditConfig,
    check_significance,
    load_core_pairs,
    run_claim_audit,
)

_CORE_PAIRS = ["CDK2_ERK2", "AChE_BChE", "BACE1_GSK3B", "AChE_BACE1", "GSK3B_CDK5"]


# ---------------------------------------------------------------------------
# SAP constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_p_threshold(self) -> None:
        assert P_THRESHOLD == 0.05

    def test_cliffs_delta_threshold(self) -> None:
        assert CLIFFS_DELTA_THRESHOLD == 0.33

    def test_success_min_pairs(self) -> None:
        assert SUCCESS_MIN_PAIRS == 3

    def test_missing_data_threshold(self) -> None:
        assert MISSING_DATA_THRESHOLD == 0.20

    def test_degradation_delta_threshold(self) -> None:
        assert DEGRADATION_DELTA_THRESHOLD == -0.47


# ---------------------------------------------------------------------------
# check_significance
# ---------------------------------------------------------------------------


class TestCheckSignificance:
    def _entry(self, *, p_adj: float = 0.02, delta: float = 0.5,
               ci_lo: float = 0.01, ci_hi: float = 0.10,
               reported: bool = True) -> dict[str, Any]:
        return {
            "p_value_adjusted": p_adj,
            "cliffs_delta": delta,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "significant": reported,
        }

    def test_all_criteria_met_no_issues(self) -> None:
        derived, issues = check_significance(self._entry())
        assert derived is True
        assert issues == []

    def test_p_too_high(self) -> None:
        derived, issues = check_significance(self._entry(p_adj=0.10, reported=False))
        assert derived is False
        assert issues == []

    def test_delta_too_small(self) -> None:
        derived, issues = check_significance(self._entry(delta=0.2, reported=False))
        assert derived is False
        assert issues == []

    def test_ci_includes_zero(self) -> None:
        derived, issues = check_significance(self._entry(ci_lo=-0.05, ci_hi=0.10, reported=False))
        assert derived is False
        assert issues == []

    def test_mismatch_detected(self) -> None:
        # All criteria met but reported as False -> mismatch
        derived, issues = check_significance(self._entry(reported=False))
        assert derived is True
        assert len(issues) == 1
        assert "Significance mismatch" in issues[0]

    def test_fallback_to_p_value_raw(self) -> None:
        entry = {
            "p_value_adjusted": None,
            "p_value_raw": 0.01,
            "cliffs_delta": 0.5,
            "ci_lower": 0.01,
            "ci_upper": 0.10,
            "significant": True,
        }
        derived, issues = check_significance(entry)
        assert derived is True
        assert issues == []

    def test_no_p_value_at_all(self) -> None:
        entry = {
            "cliffs_delta": 0.5,
            "ci_lower": 0.01,
            "ci_upper": 0.10,
            "significant": False,
        }
        # p_adj=None and p_value_raw missing -> fallback to 1.0 -> not significant
        derived, issues = check_significance(entry)
        assert derived is False
        assert issues == []

    def test_negative_delta_uses_abs(self) -> None:
        derived, issues = check_significance(self._entry(delta=-0.5))
        assert derived is True
        assert issues == []

    def test_ci_entirely_negative_is_ok(self) -> None:
        # CI excludes zero when both bounds are negative
        derived, issues = check_significance(self._entry(ci_lo=-0.20, ci_hi=-0.05))
        assert derived is True
        assert issues == []

    def test_boundary_p_exactly_005(self) -> None:
        # p=0.05 is NOT < 0.05, so should not be significant
        derived, issues = check_significance(self._entry(p_adj=0.05, reported=False))
        assert derived is False
        assert issues == []

    def test_boundary_delta_exactly_033(self) -> None:
        # |delta|=0.33 IS >= 0.33, so meets criterion
        derived, issues = check_significance(self._entry(delta=0.33))
        assert derived is True
        assert issues == []


# ---------------------------------------------------------------------------
# load_core_pairs
# ---------------------------------------------------------------------------


class TestLoadCorePairs:
    def _write_pairs(self, tmp_path: Path, pairs_data: dict[str, Any]) -> Path:
        cfg = tmp_path / "pairs.yaml"
        cfg.write_text(yaml.dump({"pairs": pairs_data}))
        return cfg

    def _target(self, tid: str) -> dict[str, str]:
        return {"id": tid, "organism": "Homo sapiens"}

    def test_basic_core_pairs(self, tmp_path: Path) -> None:
        pairs = {}
        for pid in _CORE_PAIRS:
            a, b = pid.split("_", 1)
            pairs[pid] = {"set": "core", "target_a": self._target(a), "target_b": self._target(b)}
        cfg = self._write_pairs(tmp_path, pairs)
        result = load_core_pairs(cfg)
        assert sorted(result) == sorted(_CORE_PAIRS)

    def test_blocked_excluded(self, tmp_path: Path) -> None:
        pairs = {
            "P1": {"set": "core", "target_a": self._target("A"), "target_b": self._target("B")},
            "P2": {"set": "blocked", "target_a": self._target("C"), "target_b": self._target("D")},
        }
        cfg = self._write_pairs(tmp_path, pairs)
        result = load_core_pairs(cfg)
        assert result == ["P1"]

    def test_auxiliary_excluded(self, tmp_path: Path) -> None:
        pairs = {
            "P1": {"set": "core", "target_a": self._target("A"), "target_b": self._target("B")},
            "P2": {"set": "auxiliary", "target_a": self._target("C"), "target_b": self._target("D")},
        }
        cfg = self._write_pairs(tmp_path, pairs)
        result = load_core_pairs(cfg)
        assert result == ["P1"]

    def test_default_set_is_core(self, tmp_path: Path) -> None:
        # If 'set' key is missing, default to 'core'
        pairs = {
            "P1": {"target_a": self._target("A"), "target_b": self._target("B")},
        }
        cfg = self._write_pairs(tmp_path, pairs)
        result = load_core_pairs(cfg)
        assert result == ["P1"]


# ---------------------------------------------------------------------------
# EVIDENCE_MAP
# ---------------------------------------------------------------------------


class TestEvidenceMap:
    def test_all_scenarios_present(self) -> None:
        expected = {
            "6.1_full_success", "6.2_partial_success", "6.3_weak_evidence",
            "6.4_degradation_detected", "6.5_no_evidence", "6.6_insufficient_data",
        }
        assert set(EVIDENCE_MAP.keys()) == expected

    def test_each_scenario_has_required_keys(self) -> None:
        for scenario, data in EVIDENCE_MAP.items():
            assert "claim_template" in data, f"{scenario} missing claim_template"
            assert "evidence" in data, f"{scenario} missing evidence"
            assert "sap_sections" in data, f"{scenario} missing sap_sections"
            assert "manuscript_sections" in data, f"{scenario} missing manuscript_sections"


# ---------------------------------------------------------------------------
# AuditConfig
# ---------------------------------------------------------------------------


class TestAuditConfig:
    def test_frozen(self, tmp_path: Path) -> None:
        cfg = AuditConfig(project_root=tmp_path)
        with pytest.raises(AttributeError):
            cfg.strict = False  # type: ignore[misc]

    def test_defaults(self, tmp_path: Path) -> None:
        cfg = AuditConfig(project_root=tmp_path)
        assert cfg.stats_path == Path("report/stats.json")
        assert cfg.metrics_path == Path("report/metrics.json")
        assert cfg.sensitivity_path == Path("report/data/threshold_sensitivity.json")
        assert cfg.pairs_config == Path("configs/pairs.yaml")
        assert cfg.strict is True


# ---------------------------------------------------------------------------
# run_claim_audit
# ---------------------------------------------------------------------------


def _make_pairs_yaml() -> dict[str, Any]:
    pairs: dict[str, Any] = {}
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
    degradation: dict[str, Any] | None = None,
    missing_rate: float = 0.0,
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for i, pair in enumerate(_CORE_PAIRS):
        sig = i < n_significant
        stats[pair] = {}
        for test_id in ("H1a", "H1b"):
            stats[pair][test_id] = {
                "observed_diff": 0.05 if sig else -0.01,
                "p_value_adjusted": 0.02 if sig else 0.6,
                "cliffs_delta": 0.5 if sig else 0.1,
                "ci_lower": 0.01 if sig else -0.05,
                "ci_upper": 0.10 if sig else 0.05,
                "significant": sig,
            }
    if degradation is not None:
        stats["_degradation"] = degradation
    else:
        deg: dict[str, Any] = {}
        for pair in _CORE_PAIRS:
            deg[pair] = {
                "hr_a": {"cliffs_delta": 0.1, "degraded": False},
                "hr_b": {"cliffs_delta": 0.2, "degraded": False},
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


def _setup_project(
    tmp_path: Path,
    *,
    n_significant: int = 5,
    sensitivity_pass: bool = True,
    degradation: dict[str, Any] | None = None,
    missing_rate: float = 0.0,
    include_metrics: bool = True,
    include_sensitivity: bool = True,
) -> AuditConfig:
    configs = tmp_path / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "pairs.yaml").write_text(yaml.dump(_make_pairs_yaml()))

    report = tmp_path / "report"
    report.mkdir(parents=True, exist_ok=True)
    (report / "data").mkdir(parents=True, exist_ok=True)

    (report / "stats.json").write_text(
        json.dumps(_make_stats(n_significant=n_significant, degradation=degradation, missing_rate=missing_rate))
    )

    if include_metrics:
        (report / "metrics.json").write_text(json.dumps({"CDK2_ERK2": {}}))

    if include_sensitivity:
        sens = {
            "robustness_summary": {
                "-1.0": {"n_core_significant": 3 if sensitivity_pass else 1, "passes_success_criterion": sensitivity_pass},
                "0.0": {"n_core_significant": 4, "passes_success_criterion": True},
                "1.0": {"n_core_significant": 3 if sensitivity_pass else 2, "passes_success_criterion": sensitivity_pass},
            },
        }
        (report / "data" / "threshold_sensitivity.json").write_text(json.dumps(sens))

    return AuditConfig(project_root=tmp_path)


class TestRunClaimAudit:
    def test_full_success(self, tmp_path: Path) -> None:
        cfg = _setup_project(tmp_path, n_significant=5, sensitivity_pass=True)
        audit = run_claim_audit(cfg)
        assert audit["pass"] is True
        assert audit["claim_scenario"] == "6.1_full_success"
        assert audit["degradation_warning"] is None
        assert audit["evidence_map"] is not None

    def test_partial_success(self, tmp_path: Path) -> None:
        cfg = _setup_project(tmp_path, n_significant=3, sensitivity_pass=False)
        audit = run_claim_audit(cfg)
        assert audit["claim_scenario"] == "6.2_partial_success"

    def test_weak_evidence(self, tmp_path: Path) -> None:
        cfg = _setup_project(tmp_path, n_significant=2)
        # Make 3 pairs positive direction
        stats_path = tmp_path / "report" / "stats.json"
        stats = json.loads(stats_path.read_text())
        stats["BACE1_GSK3B"]["H1a"]["observed_diff"] = 0.02
        stats_path.write_text(json.dumps(stats))

        audit = run_claim_audit(cfg)
        assert audit["claim_scenario"] == "6.3_weak_evidence"

    def test_no_evidence(self, tmp_path: Path) -> None:
        cfg = _setup_project(tmp_path, n_significant=0)
        audit = run_claim_audit(cfg)
        assert audit["claim_scenario"] == "6.5_no_evidence"

    def test_insufficient_data(self, tmp_path: Path) -> None:
        cfg = _setup_project(tmp_path, n_significant=5, missing_rate=0.30)
        audit = run_claim_audit(cfg)
        assert audit["claim_scenario"] == "6.6_insufficient_data"
        assert any("Missing data rate" in i for i in audit["issues"])

    def test_missing_stats_file(self, tmp_path: Path) -> None:
        cfg = _setup_project(tmp_path, n_significant=5)
        (tmp_path / "report" / "stats.json").unlink()
        audit = run_claim_audit(cfg)
        assert audit["pass"] is False
        assert any("stats.json not found" in i for i in audit["issues"])

    def test_missing_metrics_file(self, tmp_path: Path) -> None:
        cfg = _setup_project(tmp_path, n_significant=5, include_metrics=False)
        audit = run_claim_audit(cfg)
        assert audit["pass"] is False
        assert any("metrics.json not found" in i for i in audit["issues"])

    def test_degradation_appends_warning(self, tmp_path: Path) -> None:
        degradation = {pair: {"hr_a": {"cliffs_delta": 0.1, "degraded": False}, "hr_b": {"cliffs_delta": 0.2, "degraded": False}} for pair in _CORE_PAIRS}
        degradation["CDK2_ERK2"]["hr_a"] = {"cliffs_delta": -0.6, "degraded": True}
        cfg = _setup_project(tmp_path, n_significant=5, degradation=degradation)
        audit = run_claim_audit(cfg)
        assert audit["degradation_warning"] is not None
        assert audit["degradation_warning"]["scenario"] == "6.4_degradation_detected"

    def test_degradation_excludes_from_count(self, tmp_path: Path) -> None:
        degradation = {pair: {"hr_a": {"cliffs_delta": 0.1, "degraded": False}, "hr_b": {"cliffs_delta": 0.2, "degraded": False}} for pair in _CORE_PAIRS}
        degradation["CDK2_ERK2"]["hr_a"] = {"cliffs_delta": -0.6, "degraded": True}
        cfg = _setup_project(tmp_path, n_significant=3, degradation=degradation)
        audit = run_claim_audit(cfg)
        # CDK2_ERK2 was 1 of 3 significant -> after exclusion -> 2
        assert audit["summary"]["n_core_significant"] == 2
        assert audit["summary"]["success_criterion_met"] is False
        assert "CDK2_ERK2" in audit["summary"]["degraded_excluded"]

    def test_no_degradation_block_passes(self, tmp_path: Path) -> None:
        cfg = _setup_project(tmp_path, n_significant=5)
        stats_path = tmp_path / "report" / "stats.json"
        stats = json.loads(stats_path.read_text())
        del stats["_degradation"]
        stats_path.write_text(json.dumps(stats))
        audit = run_claim_audit(cfg)
        check = next(c for c in audit["checks"] if c["name"] == "no_degradation_guard")
        assert check["pass"] is True

    def test_sensitivity_not_present(self, tmp_path: Path) -> None:
        cfg = _setup_project(tmp_path, n_significant=5, include_sensitivity=False)
        audit = run_claim_audit(cfg)
        # Without sensitivity file, threshold_robust defaults to False -> partial success
        assert audit["claim_scenario"] == "6.2_partial_success"

    def test_absolute_paths_in_config(self, tmp_path: Path) -> None:
        cfg = _setup_project(tmp_path, n_significant=5)
        abs_cfg = AuditConfig(
            project_root=tmp_path,
            stats_path=tmp_path / "report" / "stats.json",
            metrics_path=tmp_path / "report" / "metrics.json",
            sensitivity_path=tmp_path / "report" / "data" / "threshold_sensitivity.json",
        )
        audit = run_claim_audit(abs_cfg)
        assert audit["pass"] is True

    def test_output_structure(self, tmp_path: Path) -> None:
        cfg = _setup_project(tmp_path, n_significant=5)
        audit = run_claim_audit(cfg)
        # Verify required top-level keys
        assert "checks" in audit
        assert "pass" in audit
        assert "claim_scenario" in audit
        assert "degradation_warning" in audit
        assert "evidence_map" in audit
        assert "summary" in audit
        assert "issues" in audit
        # Summary should have expected subkeys
        assert "n_core_significant" in audit["summary"]
        assert "pair_significant" in audit["summary"]
        assert "success_criterion_met" in audit["summary"]
        assert "missing_rate" in audit["summary"]

    def test_json_serializable(self, tmp_path: Path) -> None:
        cfg = _setup_project(tmp_path, n_significant=5)
        audit = run_claim_audit(cfg)
        # Must not raise
        serialized = json.dumps(audit, indent=2)
        roundtrip = json.loads(serialized)
        assert roundtrip["claim_scenario"] == audit["claim_scenario"]
