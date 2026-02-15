"""Tests for dualdocking.validation.md_docker module.

Verifies Docker image management and container run command construction.
All subprocess calls are mocked -- no Docker daemon required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dualdocking.validation.md_docker import (
    build_md_image,
    docker_image_exists,
    run_md_in_docker,
)


# ── docker_image_exists ─────────────────────────────────────────────


class TestDockerImageExists:
    def test_exists(self) -> None:
        with patch("dualdocking.validation.md_docker.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert docker_image_exists("dualdocking/md:latest") is True
            cmd = mock_run.call_args[0][0]
            assert "image" in cmd
            assert "inspect" in cmd
            assert "dualdocking/md:latest" in cmd

    def test_not_exists(self) -> None:
        with patch("dualdocking.validation.md_docker.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert docker_image_exists("dualdocking/md:latest") is False


# ── build_md_image ──────────────────────────────────────────────────


class TestBuildMdImage:
    def test_skips_when_exists(self) -> None:
        with patch("dualdocking.validation.md_docker.docker_image_exists", return_value=True) as mock_exists, \
             patch("dualdocking.validation.md_docker.subprocess.run") as mock_run:
            result = build_md_image(Path("/fake/root"))
            assert result is True
            mock_exists.assert_called_once()
            mock_run.assert_not_called()

    def test_builds_when_missing(self, tmp_path: Path) -> None:
        dockerfile = tmp_path / "docker" / "md" / "Dockerfile"
        dockerfile.parent.mkdir(parents=True)
        dockerfile.write_text("FROM scratch\n")

        with patch("dualdocking.validation.md_docker.docker_image_exists", return_value=False), \
             patch("dualdocking.validation.md_docker.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = build_md_image(tmp_path, cuda_arch="89")
            assert result is True

            cmd = mock_run.call_args[0][0]
            assert "build" in cmd
            assert "-f" in cmd
            assert str(dockerfile) in cmd
            assert "--build-arg" in cmd
            idx = cmd.index("--build-arg")
            assert cmd[idx + 1] == "CUDA_ARCHITECTURES=89"

    def test_dry_run_skips_execution(self, tmp_path: Path) -> None:
        dockerfile = tmp_path / "docker" / "md" / "Dockerfile"
        dockerfile.parent.mkdir(parents=True)
        dockerfile.write_text("FROM scratch\n")

        with patch("dualdocking.validation.md_docker.docker_image_exists", return_value=False), \
             patch("dualdocking.validation.md_docker.subprocess.run") as mock_run:
            result = build_md_image(tmp_path, dry_run=True)
            assert result is True
            mock_run.assert_not_called()

    def test_missing_dockerfile_raises(self, tmp_path: Path) -> None:
        with patch("dualdocking.validation.md_docker.docker_image_exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Dockerfile not found"):
                build_md_image(tmp_path)

    def test_build_failure_returns_false(self, tmp_path: Path) -> None:
        dockerfile = tmp_path / "docker" / "md" / "Dockerfile"
        dockerfile.parent.mkdir(parents=True)
        dockerfile.write_text("FROM scratch\n")

        with patch("dualdocking.validation.md_docker.docker_image_exists", return_value=False), \
             patch("dualdocking.validation.md_docker.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
            result = build_md_image(tmp_path)
            assert result is False

    def test_force_rebuilds(self, tmp_path: Path) -> None:
        dockerfile = tmp_path / "docker" / "md" / "Dockerfile"
        dockerfile.parent.mkdir(parents=True)
        dockerfile.write_text("FROM scratch\n")

        with patch("dualdocking.validation.md_docker.docker_image_exists") as mock_exists, \
             patch("dualdocking.validation.md_docker.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            build_md_image(tmp_path, force=True)
            mock_exists.assert_not_called()
            mock_run.assert_called_once()

    def test_timeout_returns_false(self, tmp_path: Path) -> None:
        import subprocess

        dockerfile = tmp_path / "docker" / "md" / "Dockerfile"
        dockerfile.parent.mkdir(parents=True)
        dockerfile.write_text("FROM scratch\n")

        with patch("dualdocking.validation.md_docker.docker_image_exists", return_value=False), \
             patch("dualdocking.validation.md_docker.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker build", timeout=10)
            result = build_md_image(tmp_path, timeout=10)
            assert result is False


# ── run_md_in_docker ────────────────────────────────────────────────


class TestRunMdInDocker:
    def test_basic_run(self, tmp_path: Path) -> None:
        with patch("dualdocking.validation.md_docker.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            result = run_md_in_docker(
                pair="CDK2_ERK2",
                project_root=tmp_path,
            )
            assert result.returncode == 0
            cmd = mock_run.call_args[0][0]

            # GPU flags present
            assert "--gpus" in cmd
            assert "all" in cmd

            # Volume mounts
            cmd_str = " ".join(cmd)
            assert "/workspace/output" in cmd_str
            assert "/workspace/configs" in cmd_str
            assert "/workspace/dualdocking" in cmd_str
            assert "/workspace/cli" in cmd_str
            assert "/workspace/pyproject.toml" in cmd_str

            # CLI args forwarded
            assert "--pair" in cmd
            assert "CDK2_ERK2" in cmd
            assert "--top-n" in cmd
            assert "3" in cmd

    def test_no_gpu_flag(self, tmp_path: Path) -> None:
        with patch("dualdocking.validation.md_docker.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            run_md_in_docker(pair="P", no_gpu=True, project_root=tmp_path)
            cmd = mock_run.call_args[0][0]
            assert "--gpus" not in cmd
            assert "--no-gpu" in cmd

    def test_all_options_forwarded(self, tmp_path: Path) -> None:
        with patch("dualdocking.validation.md_docker.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            run_md_in_docker(
                pair="CDK2_ERK2",
                mode="single_a",
                top_n=5,
                mmpbsa=True,
                simulation_time=10.0,
                project_root=tmp_path,
            )
            cmd = mock_run.call_args[0][0]
            assert "--mode" in cmd
            assert "single_a" in cmd
            assert "--top-n" in cmd
            assert "5" in cmd
            assert "--mmpbsa" in cmd
            assert "--simulation-time" in cmd
            assert "10.0" in cmd

    def test_dry_run(self, tmp_path: Path) -> None:
        with patch("dualdocking.validation.md_docker.subprocess.run") as mock_run:
            result = run_md_in_docker(
                pair="P",
                dry_run=True,
                project_root=tmp_path,
            )
            mock_run.assert_not_called()
            assert result.returncode == 0
            assert result.stdout == "[dry-run]"

    def test_custom_image(self, tmp_path: Path) -> None:
        with patch("dualdocking.validation.md_docker.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            run_md_in_docker(
                pair="P",
                image="custom/md:v2",
                project_root=tmp_path,
            )
            cmd = mock_run.call_args[0][0]
            assert "custom/md:v2" in cmd

    def test_volume_mount_readonly(self, tmp_path: Path) -> None:
        with patch("dualdocking.validation.md_docker.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            run_md_in_docker(pair="P", project_root=tmp_path)
            cmd = mock_run.call_args[0][0]
            cmd_str = " ".join(cmd)
            # output is rw (no :ro suffix)
            assert f"{tmp_path / 'output'}:/workspace/output" in cmd_str
            # configs, dualdocking, cli, pyproject.toml are read-only
            assert f"{tmp_path / 'configs'}:/workspace/configs:ro" in cmd_str
            assert f"{tmp_path / 'dualdocking'}:/workspace/dualdocking:ro" in cmd_str
            assert f"{tmp_path / 'cli'}:/workspace/cli:ro" in cmd_str
            assert f"{tmp_path / 'pyproject.toml'}:/workspace/pyproject.toml:ro" in cmd_str

    def test_failed_container(self, tmp_path: Path) -> None:
        with patch("dualdocking.validation.md_docker.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="crash")
            result = run_md_in_docker(pair="P", project_root=tmp_path)
            assert result.returncode == 1
