"""Tests for validate_reward_quality (ENG-03 / KVA-257)."""

from __future__ import annotations

import json
from pathlib import Path

from dualdocking.analysis.fairness import validate_reward_quality


def _write_metadata(run_dir: Path, **overrides: object) -> Path:
    """Create freedpp_workspace/reward_metadata.json with sensible defaults."""
    ws = run_dir / "freedpp_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    metadata = {
        "total_calls": 100,
        "invalid_count_a": 2,
        "invalid_count_b": 1,
        "invalid_reward_fraction_a": 0.02,
        "invalid_reward_fraction_b": 0.01,
        "timeout_count_a": 1,
        "timeout_count_b": 0,
        "timeout_fraction_a": 0.01,
        "timeout_fraction_b": 0.00,
        "invalid_z_penalty": 4.0,
        "primary_target": "CDK2",
        "reward_mode": "single_a",
        "run_valid": True,
    }
    metadata.update(overrides)
    path = ws / "reward_metadata.json"
    path.write_text(json.dumps(metadata, indent=2))
    return path


class TestValidateRewardQuality:
    def test_valid_run(self, tmp_path: Path) -> None:
        _write_metadata(tmp_path)
        result = validate_reward_quality(tmp_path)
        assert result["run_valid"] is True
        assert result["issues"] == []
        assert result["total_calls"] == 100

    def test_high_invalid_fraction_a(self, tmp_path: Path) -> None:
        _write_metadata(tmp_path, invalid_reward_fraction_a=0.15)
        result = validate_reward_quality(tmp_path)
        assert result["run_valid"] is False
        assert any("invalid_reward_fraction_a" in i for i in result["issues"])

    def test_high_invalid_fraction_b(self, tmp_path: Path) -> None:
        _write_metadata(tmp_path, invalid_reward_fraction_b=0.15)
        result = validate_reward_quality(tmp_path)
        assert result["run_valid"] is False
        assert any("invalid_reward_fraction_b" in i for i in result["issues"])

    def test_high_timeout_fraction(self, tmp_path: Path) -> None:
        _write_metadata(tmp_path, timeout_fraction_a=0.08)
        result = validate_reward_quality(tmp_path)
        assert result["run_valid"] is False
        assert any("timeout_fraction_a" in i for i in result["issues"])

    def test_missing_metadata_file(self, tmp_path: Path) -> None:
        result = validate_reward_quality(tmp_path)
        assert result["run_valid"] is False
        assert any("not found" in i for i in result["issues"])

    def test_malformed_json(self, tmp_path: Path) -> None:
        ws = tmp_path / "freedpp_workspace"
        ws.mkdir(parents=True)
        (ws / "reward_metadata.json").write_text("{bad json")
        result = validate_reward_quality(tmp_path)
        assert result["run_valid"] is False
        assert any("malformed" in i for i in result["issues"])

    def test_missing_keys(self, tmp_path: Path) -> None:
        ws = tmp_path / "freedpp_workspace"
        ws.mkdir(parents=True)
        (ws / "reward_metadata.json").write_text(json.dumps({"total_calls": 50}))
        result = validate_reward_quality(tmp_path)
        # Missing keys → issues about missing keys, defaults to 0.0 (passes threshold)
        assert any("missing key" in i for i in result["issues"])
        assert result["total_calls"] == 50

    def test_custom_thresholds(self, tmp_path: Path) -> None:
        _write_metadata(tmp_path, invalid_reward_fraction_a=0.15)
        # Default threshold 0.10 would fail; custom 0.20 passes
        result = validate_reward_quality(tmp_path, max_invalid_fraction=0.20)
        assert result["run_valid"] is True

    def test_double_checks_both(self, tmp_path: Path) -> None:
        _write_metadata(
            tmp_path,
            invalid_reward_fraction_a=0.15,
            invalid_reward_fraction_b=0.12,
        )
        result = validate_reward_quality(tmp_path)
        assert result["run_valid"] is False
        assert any("invalid_reward_fraction_a" in i for i in result["issues"])
        assert any("invalid_reward_fraction_b" in i for i in result["issues"])
        assert len([i for i in result["issues"] if "invalid_reward_fraction" in i]) == 2
