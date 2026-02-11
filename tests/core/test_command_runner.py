"""Tests for shared command runner."""

from __future__ import annotations

from dualdocking.command_runner import PROJECT_ROOT, run_project_command


def test_run_project_command_inprocess_internal_script() -> None:
    """Internal Click script should run successfully via in-process dispatch."""
    result = run_project_command(
        ["python", "scripts/pipeline/run_stages.py", "--list-stages"],
        project_root=PROJECT_ROOT,
        capture_output=True,
        allow_inprocess=True,
    )
    assert result.returncode == 0


def test_run_project_command_subprocess_fallback() -> None:
    """Non-dispatchable command should run via subprocess fallback."""
    result = run_project_command(
        ["python", "-c", "print('ok')"],
        project_root=PROJECT_ROOT,
        capture_output=True,
        allow_inprocess=True,
    )
    assert result.returncode == 0
    assert "ok" in (result.stdout or "")
