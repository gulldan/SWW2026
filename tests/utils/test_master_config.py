"""Tests for ``dualdocking.utils.master_config`` helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from dualdocking.utils.master_config import read_master_config_field


def test_read_master_config_field_returns_value(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "master_config.yaml").write_text("software:\n  boltz_version: '2.2.1'\n")

    value = read_master_config_field(tmp_path, "software.boltz_version")

    assert value == "2.2.1"


def test_read_master_config_field_missing_file_with_default(tmp_path: Path) -> None:
    value = read_master_config_field(tmp_path, "software.boltz_version", default="unknown")
    assert value == "unknown"


def test_read_master_config_field_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="master_config.yaml not found"):
        read_master_config_field(tmp_path, "software.boltz_version")


def test_read_master_config_field_missing_key_with_default(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "master_config.yaml").write_text("software:\n  other: 'x'\n")

    value = read_master_config_field(tmp_path, "software.boltz_version", default="unknown")

    assert value == "unknown"


def test_read_master_config_field_missing_key_raises(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "master_config.yaml").write_text("software:\n  other: 'x'\n")

    with pytest.raises(KeyError, match="software.boltz_version"):
        read_master_config_field(tmp_path, "software.boltz_version")
