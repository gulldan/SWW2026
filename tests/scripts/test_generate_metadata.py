"""Tests for metadata helpers in scripts/s3_generate/generate.py."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import scripts.s3_generate.generate as mod


def test_git_commit_reads_head_ref(tmp_path: Path, monkeypatch) -> None:
    """Read short commit hash from .git/HEAD reference."""
    git_dir = tmp_path / ".git"
    ref_dir = git_dir / "refs" / "heads"
    ref_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (ref_dir / "main").write_text("0123456789abcdef0123456789abcdef01234567\n")
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    assert mod._git_commit() == "0123456"


def test_docker_digest_reads_master_config(tmp_path: Path, monkeypatch) -> None:
    """Use digest from master config instead of runtime docker inspect."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "master_config.yaml").write_text("software:\n  docker_image_digest: sha256:abc123\n")
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    digest = mod._docker_image_digest("dualdocking/freedpp:latest")

    assert digest == "sha256:abc123"


def test_hardware_info_uses_torch_cuda_device(monkeypatch) -> None:
    """Select GPU by FREEDPP_CUDA_DEVICE when torch reports multiple GPUs."""

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def get_device_name(idx: int) -> str:
            return ["GPU0", "GPU1"][idx]

    fake_torch = SimpleNamespace(cuda=_FakeCuda(), version=SimpleNamespace(cuda="12.4"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("FREEDPP_CUDA_DEVICE", "1")

    hw = mod._hardware_info()

    assert hw["gpu"] == "GPU1"
    assert hw["cuda_version"] == "12.4"
