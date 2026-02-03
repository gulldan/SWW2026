"""Schema validation for metadata, normalization, metrics, stats, diversity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from dualdocking.config import load_pairs

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _result(errors: list[str]) -> dict[str, Any]:
    return {"valid": len(errors) == 0, "errors": errors}


def _load_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise TypeError(f"JSON {path} должен быть словарём")
    return data


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise TypeError(f"YAML {path} должен быть словарём")
    return data


def _expected_targets() -> list[str]:
    pairs = load_pairs(PROJECT_ROOT / "configs")["pairs"]
    targets: list[str] = []
    for pair in pairs.values():
        targets.append(pair["target_a"]["id"])
        targets.append(pair["target_b"]["id"])
    return targets


def validate_metadata_schema(path: str | Path) -> dict[str, Any]:
    """Validate metadata.json schema."""
    errors: list[str] = []
    data = _load_json(path)

    required = ["docker_image_digest", "boltz_version", "docking_program", "docking_version"]
    errors.extend([f"metadata missing required field: {key}" for key in required if key not in data or not data[key]])

    if data.get("docking_program") != "unidock_gpu":
        errors.append("metadata docking_program must be 'unidock_gpu'")

    return _result(errors)


def validate_normalization_schema(path: str | Path) -> dict[str, Any]:
    """Validate normalization.yaml schema."""
    errors: list[str] = []
    data = _load_yaml(path)

    if data.get("docking_program") != "unidock_gpu":
        errors.append("normalization docking_program must be 'unidock_gpu'")

    targets = data.get("targets")
    if not isinstance(targets, dict):
        errors.append("normalization targets must be a mapping")
        return _result(errors)

    expected = _expected_targets()
    for target_id in expected:
        if target_id not in targets:
            errors.append(f"normalization missing target: {target_id}")
            continue
        entry = targets[target_id]
        errors.extend(
            [
                f"normalization missing {field} for {target_id}"
                for field in ("mean", "std", "stable_mean", "stable_std")
                if field not in entry
            ]
        )

    return _result(errors)


def validate_metrics_schema(path: str | Path) -> dict[str, Any]:
    """Validate metrics.json schema (summary only)."""
    errors: list[str] = []
    data = _load_json(path)
    pairs = load_pairs(PROJECT_ROOT / "configs")["pairs"]

    required_fields = [
        "hr_a_mean",
        "hr_a_sem",
        "hr_both_mean",
        "hr_both_sem",
        "balance_mean",
        "balance_sem",
    ]

    for pair_id in pairs:
        if pair_id not in data:
            errors.append(f"metrics missing pair: {pair_id}")
            continue
        for mode in ("single_a", "single_b", "double"):
            if mode not in data[pair_id]:
                errors.append(f"metrics missing mode {mode} for {pair_id}")
                continue
            entry = data[pair_id][mode]
            errors.extend([f"metrics missing {field} for {pair_id}/{mode}" for field in required_fields if field not in entry])

    return _result(errors)


def validate_stats_schema(path: str | Path) -> dict[str, Any]:
    """Validate stats.json schema (16 tests total)."""
    errors: list[str] = []
    data = _load_json(path)
    pairs = load_pairs(PROJECT_ROOT / "configs")["pairs"]

    total_tests = 0
    for pair_id in pairs:
        if pair_id not in data:
            errors.append(f"stats missing pair: {pair_id}")
            continue
        for test_id, test in data[pair_id].items():
            total_tests += 1
            p_raw = test.get("p_value_raw")
            if p_raw is None or not (0.0 <= float(p_raw) <= 1.0):
                errors.append(f"stats {pair_id}/{test_id} p_value_raw out of [0,1]")
            if "cliffs_delta" not in test:
                errors.append(f"stats {pair_id}/{test_id} missing cliffs_delta")
            if "cohens_d" not in test:
                errors.append(f"stats {pair_id}/{test_id} missing cohens_d")

    expected_total = len(pairs) * 4
    if total_tests != expected_total:
        errors.append(f"stats total tests {total_tests} != expected {expected_total}")

    if errors:
        logger.warning("Stats schema validation errors: {}", "; ".join(errors))

    return _result(errors)


def validate_diversity(path: str | Path) -> dict[str, Any]:
    """Validate diversity metrics in metadata.json."""
    errors: list[str] = []
    data = _load_json(path)
    diversity = data.get("diversity")
    if not isinstance(diversity, dict):
        errors.append("metadata missing diversity block")
        return _result(errors)

    for key in ("uniqueness", "scaffold_diversity", "internal_diversity", "tanimoto_snn"):
        if key not in diversity:
            errors.append(f"diversity missing {key}")
            continue
        val = float(diversity[key])
        if not (0.0 <= val <= 1.0):
            errors.append(f"diversity {key} out of [0,1]: {val}")

    if float(diversity.get("uniqueness", 0.0)) <= 0.90:
        errors.append("diversity uniqueness <= 0.90")

    if "internal_diversity_stderr" in diversity:
        stderr = float(diversity["internal_diversity_stderr"])
        if stderr < 0.0:
            errors.append("diversity internal_diversity_stderr < 0")

    return _result(errors)
