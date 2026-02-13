"""Run post-processing stage: analytics and report generation.

Executes the Stage 2 pipeline steps in order: aggregation, metrics
computation, result plotting, molecule rendering, summarization, and
optionally LaTeX report generation. Each step is logged to a timestamped
log file under ``report/``.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import click

from dualdocking.command_runner import run_project_command

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _ts() -> str:
    """Return the current UTC time as an ISO-8601 string with ``Z`` suffix."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _log_to_file(log_path: Path, message: str) -> None:
    """Append a message line to the post-pipeline log file.

    Args:
        log_path: Path to the log file (parent dirs created if needed).
        message: Line to append.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(message + "\n")


def _commands(latex_mode: str) -> list[list[str]]:
    """Build the ordered list of post-pipeline commands.

    Args:
        latex_mode: LaTeX generation mode (``"off"``, ``"draft"``, or ``"full"``).

    Returns:
        List of command token lists for each pipeline step.
    """
    cmds = [
        ["uv", "run", "python", "scripts/s5_analyze/aggregate.py"],
        ["uv", "run", "python", "scripts/s5_analyze/compute_metrics.py"],
        ["uv", "run", "python", "scripts/s6_report/plot_results.py"],
        ["uv", "run", "python", "scripts/s6_report/render_molecules.py"],
        ["uv", "run", "python", "scripts/s5_analyze/summarize.py"],
    ]
    if latex_mode != "off":
        cmds.append(["uv", "run", "python", "scripts/s6_report/generate_latex_report.py", "--mode", latex_mode])
    return cmds


def _run_cmd(cmd: list[str], log_path: Path) -> None:
    """Execute a subprocess command, logging output and enforcing success.

    Args:
        cmd: Command tokens to execute.
        log_path: Path to the post-pipeline log file.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero return code.
    """
    _log_to_file(log_path, f"[{_ts()}] RUN: {' '.join(cmd)}")
    result = run_project_command(
        cmd,
        project_root=PROJECT_ROOT,
        capture_output=True,
        allow_inprocess=True,
    )
    if result.returncode != 0:
        combined = (result.stdout or "") + (result.stderr or "")
        _log_to_file(log_path, f"[{_ts()}] ERROR (rc={result.returncode}):\n{combined[-4000:]}")
        raise RuntimeError(f"Command failed (rc={result.returncode}): {' '.join(cmd)}")
    if result.stdout:
        _log_to_file(log_path, f"[{_ts()}] OUT: {result.stdout[-2000:]}")


@click.command()
@click.option("--dry-run", is_flag=True, default=False)
@click.option(
    "--latex-mode",
    type=click.Choice(["off", "draft", "full"]),
    default="draft",
    show_default=True,
    help="LaTeX report generation mode."
)
def main(dry_run: bool, latex_mode: str) -> None:
    """Run the post-processing pipeline (aggregate, metrics, plots, report).

    Args:
        dry_run: Print commands without executing.
        latex_mode: LaTeX report mode (``"off"``, ``"draft"``, or ``"full"``).
    """
    run_id = os.environ.get("EXPERIMENT_RUN_ID", "adhoc")
    log_ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = PROJECT_ROOT / "report" / f"post_{run_id}_{log_ts}.log"
    _log_to_file(log_path, f"[{_ts()}] START run_id={run_id} latex_mode={latex_mode}")

    commands = _commands(latex_mode)
    if dry_run:
        for cmd in commands:
            click.echo(" ".join(cmd))
        return

    for cmd in commands:
        _run_cmd(cmd, log_path)

    _log_to_file(log_path, f"[{_ts()}] DONE")


if __name__ == "__main__":
    main()
