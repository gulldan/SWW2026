"""Shared command runner for pipeline orchestration.

Provides a single execution path for internal project scripts with:
- in-process dispatch for known Click scripts;
- subprocess fallback for external commands.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dualdocking.script_dispatch import run_script_inprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_project_command(
    cmd: list[str],
    *,
    project_root: Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    capture_output: bool = False,
    stdout_pipe: bool = False,
    stderr_to_parent: bool = False,
    allow_inprocess: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command with in-process optimization for internal scripts.

    Args:
        cmd: Command tokens to execute.
        project_root: Repository root used as command working directory.
        env: Optional environment mapping for command execution.
        timeout: Optional timeout in seconds.
        capture_output: Capture both stdout and stderr into returned result.
        stdout_pipe: Capture stdout only (stderr can be inherited from parent).
        stderr_to_parent: Send stderr to current process stderr.
        allow_inprocess: If True, dispatch internal project scripts in-process.

    Returns:
        Completed process result. For in-process execution, returncode is 0.

    Raises:
        ValueError: On incompatible output-capture options.
        RuntimeError: If in-process execution fails.
    """
    if capture_output and (stdout_pipe or stderr_to_parent):
        msg = "capture_output is incompatible with stdout_pipe/stderr_to_parent"
        raise ValueError(msg)

    if allow_inprocess and run_script_inprocess(cmd, project_root=project_root, env=env, cwd=project_root):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    if capture_output:
        return subprocess.run(  # noqa: S603
            cmd,
            check=False,
            cwd=project_root,
            env=env,
            timeout=timeout,
            text=True,
            capture_output=True,
        )

    if stdout_pipe:
        if stderr_to_parent:
            return subprocess.run(  # noqa: S603
                cmd,
                check=False,
                cwd=project_root,
                env=env,
                timeout=timeout,
                text=True,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
            )
        return subprocess.run(  # noqa: S603
            cmd,
            check=False,
            cwd=project_root,
            env=env,
            timeout=timeout,
            text=True,
            capture_output=True,
        )

    if stderr_to_parent:
        return subprocess.run(  # noqa: S603
            cmd,
            check=False,
            cwd=project_root,
            env=env,
            timeout=timeout,
            text=True,
            stderr=sys.stderr,
        )

    return subprocess.run(  # noqa: S603
        cmd,
        check=False,
        cwd=project_root,
        env=env,
        timeout=timeout,
        text=True,
    )
