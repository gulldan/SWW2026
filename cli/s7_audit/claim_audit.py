"""Claim traceability audit (SCI-10 / KVA-274).

Thin CLI wrapper around :func:`dualdocking.validation.audit.run_claim_audit`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@click.command()
@click.option("--stats-json", default="report/stats.json")
@click.option("--metrics-json", default="report/metrics.json")
@click.option("--sensitivity-json", default="report/data/threshold_sensitivity.json")
@click.option("--output-json", default="report/data/claim_audit.json")
@click.option("--pairs-config", default="configs/pairs.yaml", help="Path to pairs.yaml (relative to project root).")
@click.option("--strict/--no-strict", default=True, help="Exit 1 on any audit failure.")
def main(
    stats_json: str,
    metrics_json: str,
    sensitivity_json: str,
    output_json: str,
    pairs_config: str,
    strict: bool,
) -> None:
    """Run claim traceability audit against experiment outputs."""
    from dualdocking.validation.audit import AuditConfig, run_claim_audit

    cfg = AuditConfig(
        project_root=PROJECT_ROOT,
        stats_path=Path(stats_json),
        metrics_path=Path(metrics_json),
        sensitivity_path=Path(sensitivity_json),
        pairs_config=Path(pairs_config),
        strict=strict,
    )
    audit = run_claim_audit(cfg)

    output_path = PROJECT_ROOT / output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2))
    logger.info("Audit -> {}", output_path)

    if audit["issues"]:
        for issue in audit["issues"]:
            logger.warning("  ISSUE: {}", issue)
        logger.warning("Claim audit: {} issue(s) found", len(audit["issues"]))
        if strict:
            sys.exit(1)
    else:
        logger.info("Claim audit PASSED. Scenario: {}", audit["claim_scenario"])


if __name__ == "__main__":
    main()
