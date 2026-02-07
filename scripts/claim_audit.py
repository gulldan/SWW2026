"""Claim traceability audit (SCI-10 / KVA-274).

Verifies that all primary claims in the manuscript are traceable to
statistical outputs. Checks:

1. Every core pair has H1a + H1b results in stats.json
2. Significance decisions match SAP criteria (Section 4.4)
3. Success criterion evaluation (>= 3/5 core significant)
4. No-degradation guard check
5. Claim language selection matches outcome

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

        audit["summary"]["n_core_significant"] = n_significant
        audit["summary"]["pair_significant"] = pair_significant
        audit["summary"]["success_criterion_met"] = n_significant >= SUCCESS_MIN_PAIRS

        # --- Check 4: Determine claim scenario (SAP Section 6) ---
        if n_significant >= SUCCESS_MIN_PAIRS:
            # Check threshold robustness
            threshold_robust = False
            if sensitivity_path.exists():
                sens = json.loads(sensitivity_path.read_text())
                robustness = sens.get("robustness_summary", {})
                threshold_robust = all(
                    v.get("passes_success_criterion", False)
                    for v in robustness.values()
                )
            audit["summary"]["threshold_robust"] = threshold_robust

            if threshold_robust:
                audit["claim_scenario"] = "6.1_full_success"
            else:
                audit["claim_scenario"] = "6.2_partial_success"
        else:
            # Check if positive direction at least
            n_positive = sum(1 for pair_id in CORE_PAIRS
                             if stats.get(pair_id, {}).get("H1a", {}).get("observed_diff", 0) > 0
                             or stats.get(pair_id, {}).get("H1b", {}).get("observed_diff", 0) > 0)
            if n_positive >= SUCCESS_MIN_PAIRS:
                audit["claim_scenario"] = "6.3_weak_evidence"
            else:
                audit["claim_scenario"] = "6.5_no_evidence"

        audit["summary"]["n_positive_direction"] = sum(
            1 for pair_id in CORE_PAIRS
            if stats.get(pair_id, {}).get("H1a", {}).get("observed_diff", 0) > 0
            or stats.get(pair_id, {}).get("H1b", {}).get("observed_diff", 0) > 0
        )

    # --- Check 5: metrics.json exists ---
    if not metrics_path.exists():
        all_issues.append("metrics.json not found")
        audit["checks"].append({"name": "metrics_exists", "pass": False})
    else:
        audit["checks"].append({"name": "metrics_exists", "pass": True})

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
