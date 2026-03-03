"""Tests for dualdocking.data.fetch.pdb module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from dualdocking.data.fetch.pdb import (
    download_fasta,
    download_pdb,
    read_sequence,
    record_manifest,
)


def test_download_pdb_writes_file(tmp_path: Path) -> None:
    """download_pdb should write PDB content to the output path."""
    output = tmp_path / "targets" / "1abc.pdb"
    fake_resp = MagicMock()
    fake_resp.text = "ATOM 1 N ALA A 1 0.0 0.0 0.0\nEND"

    with patch("dualdocking.data.fetch.pdb.get_with_retry", return_value=fake_resp):
        result = download_pdb("1ABC", output)

    assert result == output
    assert output.exists()
    assert "ATOM" in output.read_text()


def test_download_pdb_uppercase_url(tmp_path: Path) -> None:
    """download_pdb should uppercase the PDB ID in the RCSB URL."""
    output = tmp_path / "test.pdb"
    fake_resp = MagicMock()
    fake_resp.text = "ATOM"

    with patch("dualdocking.data.fetch.pdb.get_with_retry", return_value=fake_resp) as mock_get:
        download_pdb("1abc", output)

    url_arg = mock_get.call_args[0][0]
    assert "1ABC" in url_arg


def test_download_fasta_writes_file(tmp_path: Path) -> None:
    """download_fasta should write FASTA content to the output path."""
    output = tmp_path / "CDK2.fasta"
    fake_resp = MagicMock()
    fake_resp.text = ">sp|P24941|CDK2_HUMAN\nMENFQKV\n"

    with patch("dualdocking.data.fetch.pdb.get_with_retry", return_value=fake_resp):
        result = download_fasta("P24941", output)

    assert result == output
    assert output.exists()
    assert "MENFQKV" in output.read_text()


def test_download_fasta_skips_existing(tmp_path: Path) -> None:
    """download_fasta should skip download if file exists."""
    output = tmp_path / "CDK2.fasta"
    output.write_text(">existing\nSEQ\n")

    result = download_fasta("P24941", output)
    assert result == output
    assert "existing" in output.read_text()


def test_read_sequence_strips_headers(tmp_path: Path) -> None:
    """read_sequence should return concatenated sequence without headers."""
    fasta = tmp_path / "test.fasta"
    fasta.write_text(">header line\nABC\nDEF\nGHI\n")

    seq = read_sequence(fasta)
    assert seq == "ABCDEFGHI"


def test_read_sequence_multiple_headers(tmp_path: Path) -> None:
    """read_sequence should handle multiple header lines."""
    fasta = tmp_path / "test.fasta"
    fasta.write_text(">header1\nABC\n>header2\nDEF\n")

    seq = read_sequence(fasta)
    assert seq == "ABCDEF"


def test_record_manifest_creates_manifest(tmp_path: Path) -> None:
    """record_manifest should create a manifest JSON with SHA-256."""
    test_file = tmp_path / "data" / "test.pdb"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("ATOM 1 N ALA")

    manifest_path = tmp_path / "manifest.json"

    record_manifest(
        test_file,
        url="https://rcsb.org/test.pdb",
        manifest_path=manifest_path,
        project_root=tmp_path,
    )

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert "data/test.pdb" in manifest
    entry = manifest["data/test.pdb"]
    assert len(entry["sha256"]) == 64
    assert entry["url"] == "https://rcsb.org/test.pdb"
    assert entry["size_bytes"] > 0


def test_record_manifest_appends_to_existing(tmp_path: Path) -> None:
    """record_manifest should append to an existing manifest."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"existing/file.txt": {"sha256": "abc"}}))

    test_file = tmp_path / "new.pdb"
    test_file.write_text("ATOM")

    record_manifest(
        test_file,
        url="https://example.com",
        manifest_path=manifest_path,
        project_root=tmp_path,
    )

    manifest = json.loads(manifest_path.read_text())
    assert "existing/file.txt" in manifest
    assert "new.pdb" in manifest
