"""Tests for dualdocking.pipeline.stages.fetch orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from dualdocking.pipeline.stages.fetch import (
    step_fetch_freed_data,
    step_uv_sync,
    verify_offline_readiness,
)


def test_step_uv_sync_dry_run() -> None:
    """step_uv_sync should return True on dry_run."""
    assert step_uv_sync(dry_run=True, project_root=Path("/repo")) is True


def test_step_fetch_freed_data_dry_run() -> None:
    """step_fetch_freed_data should return True on dry_run."""
    assert step_fetch_freed_data(Path("/repo"), dry_run=True) is True


def test_step_fetch_freed_data_calls_library(tmp_path: Path) -> None:
    """step_fetch_freed_data should call fetch_freed_data from library."""
    with patch("dualdocking.pipeline.stages.fetch.fetch_freed_data") as mock_fetch:
        result = step_fetch_freed_data(tmp_path, dry_run=False, force=True)
        mock_fetch.assert_called_once_with(tmp_path / "docker" / "freedpp" / "data", force=True)
    assert result is True


def test_verify_offline_readiness_missing_config(tmp_path: Path) -> None:
    """verify_offline_readiness returns missing when pairs.yaml absent."""
    missing = verify_offline_readiness(tmp_path)
    assert "configs/pairs.yaml" in missing


def test_verify_offline_readiness_all_present(tmp_path: Path) -> None:
    """verify_offline_readiness returns empty when all artifacts exist."""
    # Create pairs.yaml
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "pairs.yaml").write_text(
        "pairs:\n"
        "  P1:\n"
        "    target_a:\n"
        "      id: T1\n"
        "      pdb_id: 1ABC\n"
        "    target_b:\n"
        "      id: T2\n"
        "      pdb_id: 2DEF\n"
    )

    # Create all artifacts
    for role, pdb, tid in [("A", "1abc", "T1"), ("B", "2def", "T2")]:
        pdb_dir = tmp_path / "output" / "data" / "targets" / "P1" / role
        pdb_dir.mkdir(parents=True)
        (pdb_dir / f"{pdb}.pdb").write_text("ATOM")
        (tmp_path / "output" / "data" / "targets" / f"{tid}.fasta").write_text(">seq")
        msa_dir = tmp_path / "output" / "cache" / "boltz_msa"
        msa_dir.mkdir(parents=True, exist_ok=True)
        (msa_dir / f"{tid}.csv").write_text("msa")

    freed = tmp_path / "docker" / "freedpp" / "data"
    freed.mkdir(parents=True)
    (freed / "zinc_crem.json").write_text("[]")
    (freed / "alert_collections.csv").write_text("h")
    cal = tmp_path / "output" / "data" / "calibration"
    cal.mkdir(parents=True)
    (cal / "test.smi").write_text("CCC")

    with patch("dualdocking.pipeline.stages.fetch.docker_image_exists", return_value=True):
        missing = verify_offline_readiness(tmp_path)

    assert missing == []
