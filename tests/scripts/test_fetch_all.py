"""Tests for scripts/s0_fetch/fetch_all.py and dualdocking/data/http_utils.py."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

import scripts.s0_fetch.fetch_all as mod
from dualdocking.data.http_utils import download_file, get_with_retry, sha256_file


def test_step_docker_build_uses_repo_root_context(monkeypatch) -> None:
    """Docker build should use project root as context for Dockerfile COPY paths."""
    monkeypatch.setattr(mod, "_docker_image_exists", lambda _image: False)
    monkeypatch.setattr(mod, "DOCKER_BIN", "docker")
    monkeypatch.setattr(mod, "PROJECT_ROOT", Path("/repo"))

    captured: dict[str, list[str]] = {}

    def _fake_run(cmd, *, dry_run, timeout=None, label=""):
        captured["cmd"] = cmd
        return True

    monkeypatch.setattr(mod, "_run", _fake_run)

    ok = mod.step_docker_build(dry_run=False, image="dualdocking/freedpp:latest", force=False)

    assert ok is True
    assert captured["cmd"][-1] == "/repo"


def test_step_fetch_freed_data_uses_inprocess_runner(monkeypatch) -> None:
    """Internal fetch script should run via in-process-capable path."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", Path("/repo"))
    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return True

    monkeypatch.setattr(mod, "_run", _fake_run)

    ok = mod.step_fetch_freed_data(dry_run=False, force=True)

    assert ok is True
    assert "--force" in captured["cmd"]
    assert captured["kwargs"]["allow_inprocess"] is True


def test_step_prepare_targets_uses_inprocess_runner(monkeypatch) -> None:
    """Target preparation should use in-process-capable execution path."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", Path("/repo"))
    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return True

    monkeypatch.setattr(mod, "_run", _fake_run)

    ok = mod.step_prepare_targets(dry_run=False)

    assert ok is True
    assert "prepare_targets.py" in str(captured["cmd"][1])
    assert captured["kwargs"]["allow_inprocess"] is True


def test_step_fetch_chembl_actives_uses_inprocess_runner(monkeypatch) -> None:
    """ChEMBL fetch should run via in-process-capable path."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", Path("/repo"))
    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return True

    monkeypatch.setattr(mod, "_run", _fake_run)

    ok = mod.step_fetch_chembl_actives(dry_run=False)

    assert ok is True
    assert "fetch_chembl_actives.py" in str(captured["cmd"][1])
    assert captured["kwargs"]["allow_inprocess"] is True


def test_step_fetch_gnina_uses_inprocess_runner(monkeypatch) -> None:
    """GNINA fetch should run via in-process-capable path."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", Path("/repo"))
    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return True

    monkeypatch.setattr(mod, "_run", _fake_run)

    ok = mod.step_fetch_gnina(dry_run=False)

    assert ok is True
    assert "fetch_gnina.py" in str(captured["cmd"][1])
    assert captured["kwargs"]["allow_inprocess"] is True


# ── verify_offline_readiness: flat layout support ────────────────────────


def test_verify_finds_pdb_in_output_prefix(monkeypatch, tmp_path) -> None:
    """verify_offline_readiness should find PDB under output/ prefix."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(mod, "_docker_image_exists", lambda _: True)

    # Create minimal pairs.yaml
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

    # Create all artifacts under output/ prefix
    for role, pdb, tid in [("A", "1abc", "T1"), ("B", "2def", "T2")]:
        pdb_dir = tmp_path / "output" / "data" / "targets" / "P1" / role
        pdb_dir.mkdir(parents=True)
        (pdb_dir / f"{pdb}.pdb").write_text("ATOM")
        (tmp_path / "output" / "data" / "targets" / f"{tid}.fasta").write_text(">seq")
        msa_dir = tmp_path / "output" / "cache" / "boltz_msa"
        msa_dir.mkdir(parents=True, exist_ok=True)
        (msa_dir / f"{tid}.csv").write_text("msa")

    # FREED data + calibration
    freed = tmp_path / "docker" / "freedpp" / "data"
    freed.mkdir(parents=True)
    (freed / "zinc_crem.json").write_text("[]")
    (freed / "alert_collections.csv").write_text("h")
    cal = tmp_path / "output" / "data" / "calibration"
    cal.mkdir(parents=True)
    (cal / "test.smi").write_text("CCC")

    missing = mod.verify_offline_readiness()
    assert missing == []


def test_verify_finds_pdb_in_flat_layout(monkeypatch, tmp_path) -> None:
    """verify_offline_readiness should find PDB in flat layout (no output/ prefix)."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(mod, "_docker_image_exists", lambda _: True)

    # Create minimal pairs.yaml
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

    # Create all artifacts under FLAT layout (no output/ prefix)
    for role, pdb, tid in [("A", "1abc", "T1"), ("B", "2def", "T2")]:
        pdb_dir = tmp_path / "data" / "targets" / "P1" / role
        pdb_dir.mkdir(parents=True)
        (pdb_dir / f"{pdb}.pdb").write_text("ATOM")
        (tmp_path / "data" / "targets" / f"{tid}.fasta").write_text(">seq")
        msa_dir = tmp_path / "cache" / "boltz_msa"
        msa_dir.mkdir(parents=True, exist_ok=True)
        (msa_dir / f"{tid}.csv").write_text("msa")

    # FREED data + calibration (flat)
    freed = tmp_path / "docker" / "freedpp" / "data"
    freed.mkdir(parents=True)
    (freed / "zinc_crem.json").write_text("[]")
    (freed / "alert_collections.csv").write_text("h")
    cal = tmp_path / "data" / "calibration"
    cal.mkdir(parents=True)
    (cal / "test.smi").write_text("CCC")

    missing = mod.verify_offline_readiness()
    assert missing == []


# ── http_utils tests ─────────────────────────────────────────────────────


def test_get_with_retry_succeeds_on_first_attempt() -> None:
    """get_with_retry returns immediately on successful response."""
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.raise_for_status = MagicMock()

    with patch("dualdocking.data.http_utils.requests.get", return_value=fake_resp) as mock_get:
        resp = get_with_retry("https://example.com/test", attempts=3, backoff=0)

    assert resp is fake_resp
    assert mock_get.call_count == 1


def test_get_with_retry_retries_on_failure() -> None:
    """get_with_retry retries the configured number of times."""
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()

    with patch(
        "dualdocking.data.http_utils.requests.get",
        side_effect=[requests.ConnectionError("fail"), fake_resp],
    ) as mock_get, patch("dualdocking.data.http_utils.time.sleep"):
        resp = get_with_retry("https://example.com/test", attempts=3, backoff=0)

    assert resp is fake_resp
    assert mock_get.call_count == 2


def test_get_with_retry_raises_after_exhaustion() -> None:
    """get_with_retry raises after all attempts fail."""
    with patch(
        "dualdocking.data.http_utils.requests.get",
        side_effect=requests.ConnectionError("fail"),
    ), patch("dualdocking.data.http_utils.time.sleep"):
        try:
            get_with_retry("https://example.com/test", attempts=2, backoff=0)
            assert False, "Should have raised"  # noqa: B011
        except requests.ConnectionError:
            pass


def test_sha256_file(tmp_path) -> None:
    """sha256_file computes correct hash."""
    f = tmp_path / "test.txt"
    f.write_text("hello")
    sha = sha256_file(f)
    assert len(sha) == 64
    assert sha == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_download_file_with_retry(tmp_path) -> None:
    """download_file streams to disk and returns SHA-256."""
    dest = tmp_path / "out" / "file.bin"
    content = b"test data for download"

    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.iter_content = MagicMock(return_value=[content])
    fake_resp.__enter__ = MagicMock(return_value=fake_resp)
    fake_resp.__exit__ = MagicMock(return_value=False)

    with patch("dualdocking.data.http_utils.requests.get", return_value=fake_resp):
        sha = download_file("https://example.com/file", dest, attempts=1)

    assert dest.exists()
    assert dest.read_bytes() == content
    assert len(sha) == 64


# ── prepare_targets manifest tests ───────────────────────────────────────


def test_record_manifest_writes_sha256(monkeypatch, tmp_path) -> None:
    """_record_manifest writes SHA-256, URL, size, timestamp."""
    import scripts.s1_prepare.prepare_targets as pt_mod

    monkeypatch.setattr(pt_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pt_mod, "MANIFEST_PATH", tmp_path / "manifest.json")

    # Create a test file
    test_file = tmp_path / "data" / "targets" / "test.pdb"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("ATOM 1 N ALA A 1 0.0 0.0 0.0")

    pt_mod._record_manifest(test_file, url="https://rcsb.org/test.pdb")

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    key = "data/targets/test.pdb"
    assert key in manifest
    assert len(manifest[key]["sha256"]) == 64
    assert manifest[key]["url"] == "https://rcsb.org/test.pdb"
    assert manifest[key]["size_bytes"] > 0
