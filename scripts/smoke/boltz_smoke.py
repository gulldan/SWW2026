"""Smoke test for Boltz-2 scoring.

Runs a minimal scoring example to verify Boltz-2 is functional.
Result: ``results/smoke/boltz_score.json``

Usage::

    uv run python scripts/smoke/boltz_smoke.py --target CDK2
    uv run python scripts/smoke/boltz_smoke.py --target CDK2 --dry-run
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Minimal set: 3 molecules for quick validation
SMOKE_SMILES = [
    "c1ccccc1",  # benzene
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
]

# Valid score range (log10(IC50, μM))
# Example: 1 nM = 0.001 μM → log10 = -3
SCORE_MIN = -6.0
SCORE_MAX = 2.0


@click.command()
@click.option("--target", default="CDK2", help="Target ID for scoring")
@click.option("--dry-run", is_flag=True, default=False, help="Only generate YAML without running Boltz")
def main(target: str, dry_run: bool) -> None:
    """Run Boltz-2 smoke test with a small set of known molecules.

    Scores benzene, aspirin, and ibuprofen against the specified target
    and validates that returned scores fall within the expected range.

    Args:
        target: Target identifier (e.g. ``"CDK2"``).
        dry_run: If ``True``, only generate input YAML files without scoring.
    """
    if dry_run:
        _run_dry(target)
        return

    from dualdocking.scoring.boltz2 import score_batch

    logger.info("Running Boltz-2 smoke test: {} molecules for target {}", len(SMOKE_SMILES), target)

    scores = score_batch(target, SMOKE_SMILES, use_cache=False)

    results = {
        "target": target,
        "n_molecules": len(SMOKE_SMILES),
        "molecules": [{"smiles": smi, "score": float(s)} for smi, s in zip(SMOKE_SMILES, scores, strict=True)],
        "mean_score": float(scores[~np.isnan(scores)].mean()) if not np.all(np.isnan(scores)) else None,
        "status": "ok",
    }

    out_dir = PROJECT_ROOT / "output" / "results" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "boltz_score.json"
    out_path.write_text(json.dumps(results, indent=2))

    logger.info("Smoke test completed. Results -> {}", out_path)
    n_valid = 0
    for mol in results["molecules"]:
        score = mol["score"]
        logger.info("  {} -> {:.3f}", mol["smiles"], score)
        if SCORE_MIN <= score <= SCORE_MAX:
            n_valid += 1
        else:
            logger.warning("  Score {:.3f} outside expected range [{}, {}]", score, SCORE_MIN, SCORE_MAX)

    logger.info("Valid scores: {}/{}", n_valid, len(SMOKE_SMILES))


def _run_dry(target: str) -> None:
    """Dry-run: generate input YAML files without invoking Boltz.

    Args:
        target: Target identifier for sequence and MSA lookup.
    """
    from dualdocking.scoring.boltz2 import _get_msa_path, _load_sequence, _write_boltz_yaml

    logger.info("Dry-run: generating YAML for target {}", target)

    sequence = _load_sequence(target)
    logger.info("Sequence loaded: {} characters", len(sequence))

    msa_path = _get_msa_path(target)
    if msa_path:
        logger.info("MSA found: {}", msa_path)
    else:
        logger.warning("MSA not found for {}. Boltz will require --use_msa_server (online).", target)

    out_dir = PROJECT_ROOT / "output" / "results" / "smoke" / "dry_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, smi in enumerate(SMOKE_SMILES):
        yaml_path = out_dir / f"input_{i}.yaml"
        _write_boltz_yaml(sequence, smi, yaml_path, msa_path=msa_path)
        logger.info("YAML generated: {}", yaml_path)

    logger.info("Dry-run completed. YAML files in {}", out_dir)


if __name__ == "__main__":
    main()
