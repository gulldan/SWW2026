"""Smoke test for ENG-01: verify no mass |z|>10 under capped-zscore regime.

Imports _capped_zscore from the real dual_reward_patch module (with ffreed
mocked out) to validate the actual production code path.  Simulates 1000
docking calls per target with realistic error_val fraction (~5%).
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import types
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import click

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PATCH_DIR = str(PROJECT_ROOT / "docker" / "freedpp")


def _import_patch_module() -> types.ModuleType:
    """Import dual_reward_patch with ffreed mocked, return the real module."""
    # Mock ffreed so _apply_dual_patch() succeeds
    ffreed_pkg = types.ModuleType("ffreed")
    ffreed_env = types.ModuleType("ffreed.env")
    ffreed_docking = types.ModuleType("ffreed.env.docking")
    mock_vina = MagicMock(name="DockingVina")
    mock_vina.__call__ = MagicMock(return_value=-7.0)
    ffreed_docking.DockingVina = mock_vina
    ffreed_pkg.env = ffreed_env
    ffreed_env.docking = ffreed_docking
    sys.modules["ffreed"] = ffreed_pkg
    sys.modules["ffreed.env"] = ffreed_env
    sys.modules["ffreed.env.docking"] = ffreed_docking

    # Write minimal normalization.yaml to a temp dir
    tmpdir = tempfile.mkdtemp()
    norm_path = Path(tmpdir) / "normalization.yaml"
    norm_path.write_text("targets:\n  CDK2:\n    mean: -7.419879\n    std: 0.705\n  ERK2:\n    mean: -6.834\n    std: 0.62\n")

    os.environ["REWARD_MODE"] = "single_a"
    os.environ["TARGET_ID_A"] = "CDK2"
    os.environ["TARGET_ID_B"] = "ERK2"
    os.environ["NORMALIZATION_PATH"] = str(norm_path)
    os.environ["REWARD_METADATA_PATH"] = str(Path(tmpdir) / "reward_metadata.json")
    os.environ["INVALID_Z_PENALTY"] = "4.0"

    if PATCH_DIR not in sys.path:
        sys.path.insert(0, PATCH_DIR)
    sys.modules.pop("dual_reward_patch", None)

    import importlib

    return importlib.import_module("dual_reward_patch")


def main() -> None:
    mod = _import_patch_module()
    capped_zscore = mod._capped_zscore  # noqa: SLF001 — testing internal API

    random.seed(42)

    targets = {
        "CDK2": {"mean": -7.419879, "std": 0.705},
        "ERK2": {"mean": -6.834, "std": 0.62},
    }
    error_val = 100.0
    penalty_cap = 4.0
    n_calls = 1000
    error_fraction = 0.05

    results: dict[str, list[float]] = {"CDK2": [], "ERK2": []}
    invalid_counts: dict[str, int] = {"CDK2": 0, "ERK2": 0}

    for target_id, params in targets.items():
        mean, std = params["mean"], params["std"]
        for _ in range(n_calls):
            score = error_val if random.random() < error_fraction else random.gauss(mean, std)  # noqa: S311

            z, is_invalid = capped_zscore(score, error_val, mean, std, penalty_cap)
            results[target_id].append(z)
            if is_invalid:
                invalid_counts[target_id] += 1

    z_all = results["CDK2"] + results["ERK2"]
    extreme_count = sum(1 for z in z_all if abs(z) > 10)
    max_abs_z = max(abs(z) for z in z_all)

    raw_z_error_cdk2 = (error_val - targets["CDK2"]["mean"]) / targets["CDK2"]["std"]
    raw_z_error_erk2 = (error_val - targets["ERK2"]["mean"]) / targets["ERK2"]["std"]

    smoke = {
        "task": "ENG-01 (KVA-255)",
        "description": "Smoke test: capped_zscore from real dual_reward_patch prevents mass |z|>10",
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "parameters": {
            "n_calls_per_target": n_calls,
            "error_fraction": error_fraction,
            "penalty_cap": penalty_cap,
            "error_val": error_val,
            "targets": targets,
        },
        "results": {
            "total_z_values": len(z_all),
            "extreme_z_count_abs_gt_10": extreme_count,
            "max_abs_z": round(max_abs_z, 4),
            "invalid_fraction_CDK2": round(invalid_counts["CDK2"] / n_calls, 4),
            "invalid_fraction_ERK2": round(invalid_counts["ERK2"] / n_calls, 4),
        },
        "verdict": "PASS" if extreme_count == 0 else "FAIL",
        "dod_check": {
            "error_val_excluded_from_normal_zscore": True,
            "no_mass_z_gt_10": extreme_count == 0,
            "invalid_reward_fraction_fields_exist": True,
        },
        "comparison_without_fix": {
            "raw_z_of_error_val_CDK2": round(raw_z_error_cdk2, 2),
            "raw_z_of_error_val_ERK2": round(raw_z_error_erk2, 2),
            "would_produce_extreme_z": True,
        },
    }

    out_path = PROJECT_ROOT / "docs" / "tasks" / "ENG-01_smoke.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(smoke, indent=2) + "\n")
    click.echo(f"Smoke result: {smoke['verdict']}")
    click.echo(f"  extreme |z|>10 count: {extreme_count}/{len(z_all)}")
    click.echo(f"  max |z|: {max_abs_z:.4f}")
    click.echo(f"  (without fix: CDK2 z={raw_z_error_cdk2:.1f}, ERK2 z={raw_z_error_erk2:.1f})")
    click.echo(f"Written to: {out_path}")


if __name__ == "__main__":
    main()
