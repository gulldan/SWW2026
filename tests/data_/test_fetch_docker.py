"""Tests for dualdocking.data.fetch.docker module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from dualdocking.data.fetch.docker import (
    _build_zinc_crem,
    _validate_alert_csv,
    build_freedpp_image,
    docker_image_exists,
    fetch_freed_data,
)


def test_docker_image_exists_true() -> None:
    """docker_image_exists returns True when docker inspect succeeds."""
    mock_result = MagicMock()
    mock_result.returncode = 0

    with patch("dualdocking.data.fetch.docker.subprocess.run", return_value=mock_result):
        assert docker_image_exists("test:latest") is True


def test_docker_image_exists_false() -> None:
    """docker_image_exists returns False when docker inspect fails."""
    mock_result = MagicMock()
    mock_result.returncode = 1

    with patch("dualdocking.data.fetch.docker.subprocess.run", return_value=mock_result):
        assert docker_image_exists("test:latest") is False


def test_validate_alert_csv_valid(tmp_path: Path) -> None:
    """_validate_alert_csv should pass for valid CSV with header."""
    csv_file = tmp_path / "alerts.csv"
    csv_file.write_text("rule_set_name,smarts\nPAINS,[#6]\n")
    _validate_alert_csv(csv_file)


def test_validate_alert_csv_empty(tmp_path: Path) -> None:
    """_validate_alert_csv should raise for empty CSV."""
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("")

    import pytest

    with pytest.raises(RuntimeError, match="Empty"):
        _validate_alert_csv(csv_file)


def test_build_zinc_crem_writes_json(tmp_path: Path) -> None:
    """_build_zinc_crem should write JSON fragment list."""
    out_path = tmp_path / "zinc_crem.json"

    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.iter_content = MagicMock(return_value=[b"frag1\nfrag2\nfrag3\n"])
    fake_resp.__enter__ = MagicMock(return_value=fake_resp)
    fake_resp.__exit__ = MagicMock(return_value=False)

    with patch("dualdocking.data.http_utils.requests.get", return_value=fake_resp):
        _build_zinc_crem(out_path, timeout=30)

    assert out_path.exists()
    fragments = json.loads(out_path.read_text())
    assert len(fragments) == 3
    assert "frag1" in fragments


def test_fetch_freed_data_skips_existing(tmp_path: Path) -> None:
    """fetch_freed_data should skip if both files already exist."""
    (tmp_path / "zinc_crem.json").write_text("[]")
    (tmp_path / "alert_collections.csv").write_text("header\n")

    fetch_freed_data(tmp_path, force=False)

    # Files should be unchanged
    assert json.loads((tmp_path / "zinc_crem.json").read_text()) == []


def test_build_freedpp_image_skips_existing() -> None:
    """build_freedpp_image should skip if image exists and force=False."""
    with patch("dualdocking.data.fetch.docker.docker_image_exists", return_value=True):
        result = build_freedpp_image(Path("/repo"), image="test:latest", force=False)
    assert result is True


def test_build_freedpp_image_dry_run() -> None:
    """build_freedpp_image should return True on dry_run."""
    with patch("dualdocking.data.fetch.docker.docker_image_exists", return_value=False):
        result = build_freedpp_image(Path("/repo"), image="test:latest", dry_run=True)
    assert result is True
