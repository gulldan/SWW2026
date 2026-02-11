"""Tests for scripts/s1_prepare/prepare_all.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import click
import pytest

import scripts.s1_prepare.prepare_all as mod


def test_run_uses_inprocess_project_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Internal script commands should be routed through run_project_command."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", Path("/repo"))
    captured: dict[str, object] = {}

    def _fake_run_project_command(cmd, **kwargs):  # noqa: ANN001, ANN003
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod, "run_project_command", _fake_run_project_command)

    mod._run(["python", "scripts/s0_fetch/fetch_freed_fragments.py"], dry_run=False)

    assert captured["cmd"] == ["python", "scripts/s0_fetch/fetch_freed_fragments.py"]
    assert captured["kwargs"]["project_root"] == Path("/repo")
    assert captured["kwargs"]["capture_output"] is True
    assert captured["kwargs"]["allow_inprocess"] is True


def test_run_converts_inprocess_runtime_error_to_click_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """In-process dispatch failure should surface as ClickException."""
    monkeypatch.setattr(mod, "PROJECT_ROOT", Path("/repo"))

    def _boom(*_args, **_kwargs):  # noqa: ANN001, ANN002, ANN003
        raise RuntimeError("dispatch failed")

    monkeypatch.setattr(mod, "run_project_command", _boom)

    with pytest.raises(click.ClickException, match="dispatch failed"):
        mod._run(["python", "scripts/s0_fetch/fetch_freed_fragments.py"], dry_run=False)
