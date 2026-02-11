"""Tests for in-process script dispatcher."""

from __future__ import annotations

from pathlib import Path

from dualdocking.script_dispatch import PROJECT_ROOT, parse_script_invocation, run_script_inprocess


def test_parse_uv_run_python() -> None:
    """Dispatcher should parse `uv run python scripts/...` commands."""
    parsed = parse_script_invocation(
        ["uv", "run", "python", "scripts/pipeline/run_stages.py", "--list-stages"],
        project_root=PROJECT_ROOT,
    )
    assert parsed is not None
    module, args, script_rel = parsed
    assert module == "scripts.pipeline.run_stages"
    assert args == ["--list-stages"]
    assert script_rel == "scripts/pipeline/run_stages.py"


def test_parse_python_absolute_path() -> None:
    """Dispatcher should parse absolute script path with python executable."""
    script = PROJECT_ROOT / "scripts" / "pipeline" / "run_stages.py"
    parsed = parse_script_invocation(
        ["python", str(script), "--list-stages"],
        project_root=PROJECT_ROOT,
    )
    assert parsed is not None
    module, args, script_rel = parsed
    assert module == "scripts.pipeline.run_stages"
    assert args == ["--list-stages"]
    assert script_rel == "scripts/pipeline/run_stages.py"


def test_run_script_inprocess_list_stages() -> None:
    """run_script_inprocess should execute lightweight stage-list command."""
    ok = run_script_inprocess(
        ["python", "scripts/pipeline/run_stages.py", "--list-stages"],
        project_root=PROJECT_ROOT,
        cwd=PROJECT_ROOT,
    )
    assert ok is True


def test_run_script_inprocess_non_project_command_returns_false() -> None:
    """Non-script commands should not be handled by dispatcher."""
    ok = run_script_inprocess(["echo", "hello"], project_root=Path("/tmp"))
    assert ok is False
