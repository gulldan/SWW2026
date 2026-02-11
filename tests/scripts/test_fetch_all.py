"""Tests for scripts/s0_fetch/fetch_all.py."""

from __future__ import annotations

from pathlib import Path

import scripts.s0_fetch.fetch_all as mod


def test_step_docker_build_uses_repo_root_context(monkeypatch) -> None:
    """Docker build should use project root as context for Dockerfile COPY paths."""
    monkeypatch.setattr(mod, "_docker_image_exists", lambda _image: False)
    monkeypatch.setattr(mod, "DOCKER_BIN", "docker")
    monkeypatch.setattr(mod, "PROJECT_ROOT", Path("/repo"))

    captured: dict[str, list[str]] = {}

    def _fake_run(cmd, *, dry_run, timeout=None, label=""):  # noqa: ARG001
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

    def _fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
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

    def _fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return True

    monkeypatch.setattr(mod, "_run", _fake_run)

    ok = mod.step_prepare_targets(dry_run=False)

    assert ok is True
    assert "prepare_targets.py" in str(captured["cmd"][1])
    assert captured["kwargs"]["allow_inprocess"] is True
