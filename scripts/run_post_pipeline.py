"""Run post-processing stage: analytics + report generation."""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ts() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _log_to_file(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(message + "\n")


def _commands(latex_mode: str) -> list[list[str]]:
    cmds = [
        ["uv", "run", "python", "scripts/aggregate.py"],
        ["uv", "run", "python", "scripts/compute_metrics.py"],
        ["uv", "run", "python", "scripts/plot_results.py"],
        ["uv", "run", "python", "scripts/render_molecules.py"],
        ["uv", "run", "python", "scripts/summarize.py"],
    ]
    if latex_mode != "off":
        cmds.append(["uv", "run", "python", "scripts/generate_latex_report.py", "--mode", latex_mode])
    return cmds


def _run_cmd(cmd: list[str], log_path: Path) -> None:
    _log_to_file(log_path, f"[{_ts()}] RUN: {' '.join(cmd)}")
    result = subprocess.run(  # noqa: S603
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
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
    help="Режим LaTeX-генерации статьи.",
)
def main(dry_run: bool, latex_mode: str) -> None:
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
