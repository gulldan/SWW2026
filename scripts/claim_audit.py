"""Claim traceability audit (SCI-10 / KVA-274).

Verifies that all primary claims in the manuscript are traceable to
statistical outputs. Checks:

1. stats.json exists
2. Every core pair has H1a + H1b results in stats.json
3. Significance decisions match SAP criteria (Section 4.4)
4. Success criterion evaluation (>= 3/5 core significant)
5. No-degradation guard (SAP Section 5.2)
6. Completeness / missing-data guard (SAP Section 6.6)
7. Claim language selection matches outcome
8. Evidence-to-claims traceability mapping

Outputs:
  - report/data/claim_audit.json (structured audit trail)
  - Exit code 0 = pass, 1 = fail
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CORE_PAIRS = ["CDK2_ERK2", "AChE_BChE", "BACE1_GSK3B", "AChE_BACE1", "GSK3B_CDK5"]
SIGNIFICANCE_CRITERIA = {
    "p_threshold": 0.05,
    "cliffs_delta_threshold": 0.33,
    "ci_excludes_zero": True,
}
NO_DEGRADATION_THRESHOLD = -0.47
SUCCESS_MIN_PAIRS = 3
MISSING_DATA_THRESHOLD = 0.20

# SAP Section 6: evidence mapping from claim scenarios to supporting artifacts.
EVIDENCE_MAP: dict[str, dict] = {
    "6.1_full_success": {
        "claim_template": (
            "Double-target RL generation with worst-case scalarization produces a "
            "significantly higher dual hit-rate than single-target baselines on "
            "{n_sig}/5 core target pairs (BH-adjusted p < 0.05, |Cliff's delta| >= 0.33), "
            "with no per-target degradation. The effect is robust across hit-rate "
            "thresholds (100 nM, 1 uM, 10 uM)."
        ),
        "evidence": [
            {"artifact": "report/stats.json", "fields": ["H1a", "H1b"], "purpose": "bootstrap p-values + effect sizes"},
            {"artifact": "report/metrics.json", "fields": ["hr_both_mean"], "purpose": "per-mode hit-rates"},
            {"artifact": "report/data/threshold_sensitivity.json", "purpose": "robustness across thresholds"},
            {"artifact": "report/stats.json:_degradation", "purpose": "no per-target degradation"},
        ],
        "sap_sections": ["4.4", "5.1", "5.2", "5.3"],
        "manuscript_sections": ["Quantitative Results", "Figures"],
    },
    "6.2_partial_success": {
        "claim_template": (
            "Double-target RL generation improves dual hit-rate on {n_sig}/5 core pairs "
            "at the primary threshold (T = 0.0, 1 uM), but the effect is sensitive "
            "to threshold choice."
        ),
        "evidence": [
            {"artifact": "report/stats.json", "fields": ["H1a", "H1b"], "purpose": "bootstrap p-values + effect sizes"},
            {"artifact": "report/metrics.json", "fields": ["hr_both_mean"], "purpose": "per-mode hit-rates"},
            {"artifact": "report/data/threshold_sensitivity.json", "purpose": "sensitivity failure documentation"},
        ],
        "sap_sections": ["4.4", "5.1", "5.3"],
        "manuscript_sections": ["Quantitative Results", "Limitations"],
    },
    "6.3_weak_evidence": {
        "claim_template": (
            "A consistent directional advantage for double-target RL is observed on "
            "{n_pos}/5 core pairs, but the effect size is negligible-to-small "
            "and does not meet the pre-registered practical significance threshold."
        ),
        "evidence": [
            {"artifact": "report/stats.json", "fields": ["H1a", "H1b"], "purpose": "observed differences and effect sizes"},
            {"artifact": "report/metrics.json", "fields": ["hr_both_mean"], "purpose": "per-mode hit-rates"},
        ],
        "sap_sections": ["4.4", "5.1"],
        "manuscript_sections": ["Quantitative Results", "Conclusion and Next Steps"],
    },
    "6.4_degradation_detected": {
        "claim_template": (
            "Double-target RL incurs a per-target hit-rate penalty on pair(s) {pairs} "
            "(Cliff's delta <= -0.47, large negative effect)."
        ),
        "evidence": [
            {"artifact": "report/stats.json:_degradation", "purpose": "per-target Cliff's delta"},
            {"artifact": "report/metrics.json", "fields": ["hr_a_mean", "hr_b_mean"], "purpose": "per-target hit-rates"},
        ],
        "sap_sections": ["5.2", "6.4"],
        "manuscript_sections": ["Limitations"],
    },
    "6.5_no_evidence": {
        "claim_template": (
            "No evidence that double-target RL improves dual hit-rate: only {n_pos}/5 core pairs show positive direction."
        ),
        "evidence": [
            {"artifact": "report/stats.json", "fields": ["H1a", "H1b"], "purpose": "observed differences"},
        ],
        "sap_sections": ["5.1"],
        "manuscript_sections": ["Conclusion and Next Steps"],
    },
    "6.6_insufficient_data": {
        "claim_template": (
            "Analysis inconclusive due to missing data ({missing_rate}% of expected "
            "runs unavailable). Results are reported descriptively only."
        ),
        "evidence": [
            {"artifact": "report/stats.json:_completeness", "purpose": "missing-run inventory"},
        ],
        "sap_sections": ["6.6"],
        "manuscript_sections": ["Limitations"],
    },
}


def _check_significance(entry: dict) -> tuple[bool, list[str]]:
    """Re-derive significance from raw values per SAP Section 4.4."""
    issues: list[str] = []
    p_adj = entry.get("p_value_adjusted", entry.get("p_value_raw", 1.0))
    cliffs_d = abs(float(entry.get("cliffs_delta", 0)))
    ci_lo = float(entry.get("ci_lower", 0))
    ci_hi = float(entry.get("ci_upper", 0))
    ci_ok = ci_lo > 0 or ci_hi < 0

    derived_sig = p_adj < 0.05 and cliffs_d >= 0.33 and ci_ok
    reported_sig = entry.get("significant", False)

    if derived_sig != reported_sig:
        issues.append(
            f"Significance mismatch: derived={derived_sig}, reported={reported_sig} "
            f"(p_adj={p_adj:.4f}, |delta|={cliffs_d:.3f}, CI=[{ci_lo:.4f}, {ci_hi:.4f}])"
        )

    return derived_sig, issues


@click.command()
@click.option("--stats-json", default="report/stats.json")
@click.option("--metrics-json", default="report/metrics.json")
@click.option("--sensitivity-json", default="report/data/threshold_sensitivity.json")
@click.option("--output-json", default="report/data/claim_audit.json")
@click.option("--strict/--no-strict", default=True, help="Exit 1 on any audit failure.")
def main(
    stats_json: str,
    metrics_json: str,
    sensitivity_json: str,
    output_json: str,
    strict: bool,
) -> None:
    """Run claim traceability audit."""
    stats_path = PROJECT_ROOT / stats_json
    metrics_path = PROJECT_ROOT / metrics_json
    sensitivity_path = PROJECT_ROOT / sensitivity_json
    output_path = PROJECT_ROOT / output_json

    audit: dict = {
        "checks": [],
        "pass": True,
        "claim_scenario": None,
        "degradation_warning": None,
        "evidence_map": None,
        "summary": {},
    }
    all_issues: list[str] = []

    # --- Check 1: stats.json exists and has data ---
    if not stats_path.exists():
        all_issues.append("stats.json not found")
        audit["checks"].append({"name": "stats_exists", "pass": False, "issue": "file missing"})
    else:
        stats = json.loads(stats_path.read_text())
        audit["checks"].append({"name": "stats_exists", "pass": True})

        # --- Check 2: All core pairs have H1a + H1b ---
        missing_tests: list[str] = [
            f"{pair_id}/{test_id}"
            for pair_id in CORE_PAIRS
            for test_id in ("H1a", "H1b")
            if test_id not in stats.get(pair_id, {})
        ]
        if missing_tests:
            all_issues.extend([f"Missing test: {t}" for t in missing_tests])
            audit["checks"].append({"name": "core_tests_complete", "pass": False, "missing": missing_tests})
        else:
            audit["checks"].append({"name": "core_tests_complete", "pass": True})

        # --- Check 3: Significance consistency (SAP Section 4.4) ---
        sig_issues: list[str] = []
        n_significant = 0
        pair_significant: dict[str, bool] = {}
        for pair_id in CORE_PAIRS:
            pair_stats = stats.get(pair_id, {})
            pair_sig = False
            for test_id in ("H1a", "H1b"):
                entry = pair_stats.get(test_id, {})
                if not entry:
                    continue
                derived, entry_issues = _check_significance(entry)
                sig_issues.extend([f"{pair_id}/{test_id}: {i}" for i in entry_issues])
                if derived:
                    pair_sig = True
            pair_significant[pair_id] = pair_sig
            if pair_sig:
                n_significant += 1

        if sig_issues:
            all_issues.extend(sig_issues)
            audit["checks"].append({"name": "significance_consistency", "pass": False, "issues": sig_issues})
        else:
            audit["checks"].append({"name": "significance_consistency", "pass": True})

        # --- Check 4: No-degradation guard (SAP Section 5.2) ---
        degradation = stats.get("_degradation", {})
        degraded_pairs: list[str] = []
        for pair_id in CORE_PAIRS:
            pair_deg = degradation.get(pair_id, {})
            for check_name, check_data in pair_deg.items():
                if check_data.get("degraded", False):
                    delta = check_data.get("cliffs_delta", float("nan"))
                    degraded_pairs.append(f"{pair_id}/{check_name} (delta={delta:.3f})")

        if degraded_pairs:
            all_issues.extend([f"Degradation: {d}" for d in degraded_pairs])
            audit["checks"].append({"name": "no_degradation_guard", "pass": False, "degraded": degraded_pairs})
        else:
            audit["checks"].append({"name": "no_degradation_guard", "pass": True})

        audit["summary"]["degraded_pairs"] = degraded_pairs

        # Exclude degraded pairs from success count (SAP 5.2)
        degraded_pair_ids = {dp.split("/")[0] for dp in degraded_pairs}
        for pair_id in degraded_pair_ids:
            if pair_significant.get(pair_id, False):
                pair_significant[pair_id] = False
                n_significant -= 1

        audit["summary"]["n_core_significant"] = n_significant
        audit["summary"]["pair_significant"] = pair_significant
        audit["summary"]["success_criterion_met"] = n_significant >= SUCCESS_MIN_PAIRS
        audit["summary"]["degraded_excluded"] = sorted(degraded_pair_ids)

        # --- Check 5: Completeness / missing data (SAP Section 6.6) ---
        completeness = stats.get("_completeness", {})
        missing_rate = completeness.get("missing_rate", 0.0)
        audit["summary"]["missing_rate"] = missing_rate

        if missing_rate > MISSING_DATA_THRESHOLD:
            all_issues.append(f"Missing data rate {missing_rate:.1%} exceeds {MISSING_DATA_THRESHOLD:.0%} threshold")
            audit["checks"].append({"name": "completeness_gate", "pass": False, "missing_rate": missing_rate})
        else:
            audit["checks"].append({"name": "completeness_gate", "pass": True, "missing_rate": missing_rate})

        # --- Check 6: Determine claim scenario (SAP Section 6) ---
        # 6.6 overrides everything if data is insufficient
        if missing_rate > MISSING_DATA_THRESHOLD:
            audit["claim_scenario"] = "6.6_insufficient_data"
        elif n_significant >= SUCCESS_MIN_PAIRS:
            # Check threshold robustness
            threshold_robust = False
            if sensitivity_path.exists():
                sens = json.loads(sensitivity_path.read_text())
                robustness = sens.get("robustness_summary", {})
                threshold_robust = all(v.get("passes_success_criterion", False) for v in robustness.values())
            audit["summary"]["threshold_robust"] = threshold_robust

            if threshold_robust:
                audit["claim_scenario"] = "6.1_full_success"
            else:
                audit["claim_scenario"] = "6.2_partial_success"
        else:
            n_positive = sum(
                1
                for pair_id in CORE_PAIRS
                if stats.get(pair_id, {}).get("H1a", {}).get("observed_diff", 0) > 0
                or stats.get(pair_id, {}).get("H1b", {}).get("observed_diff", 0) > 0
            )
            if n_positive >= SUCCESS_MIN_PAIRS:
                audit["claim_scenario"] = "6.3_weak_evidence"
            else:
                audit["claim_scenario"] = "6.5_no_evidence"

        audit["summary"]["n_positive_direction"] = sum(
            1
            for pair_id in CORE_PAIRS
            if stats.get(pair_id, {}).get("H1a", {}).get("observed_diff", 0) > 0
            or stats.get(pair_id, {}).get("H1b", {}).get("observed_diff", 0) > 0
        )

        # 6.4 is an addendum — appended to any scenario when degradation is found
        if degraded_pairs:
            audit["degradation_warning"] = {
                "scenario": "6.4_degradation_detected",
                "pairs": degraded_pairs,
                "note": "Appended to primary claim per SAP Section 6.4",
            }

    # --- Check 7: metrics.json exists ---
    if not metrics_path.exists():
        all_issues.append("metrics.json not found")
        audit["checks"].append({"name": "metrics_exists", "pass": False})
    else:
        audit["checks"].append({"name": "metrics_exists", "pass": True})

    # --- Evidence traceability mapping ---
    scenario = audit.get("claim_scenario")
    if scenario and scenario in EVIDENCE_MAP:
        audit["evidence_map"] = EVIDENCE_MAP[scenario]

    # --- Final verdict ---
    audit["pass"] = len(all_issues) == 0
    audit["issues"] = all_issues

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2))
    logger.info("Audit -> {}", output_path)

    if all_issues:
        for issue in all_issues:
            logger.warning("  ISSUE: {}", issue)
        logger.warning("Claim audit: {} issue(s) found", len(all_issues))
        if strict:
            sys.exit(1)
    else:
        logger.info("Claim audit PASSED. Scenario: {}", audit["claim_scenario"])


if __name__ == "__main__":
    main()
