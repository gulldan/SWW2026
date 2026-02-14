"""In-process dispatcher for Click-based project CLI entry points.

This module lets pipeline orchestrators execute internal CLI modules without
spawning a new Python process. It keeps the CLI contract intact while also
enabling direct programmatic execution from Python code.
"""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# script path (relative to repo root) -> module path containing click `main`
_SCRIPT_TO_MODULE: dict[str, str] = {
    "cli/pipeline/run_full_experiment.py": "cli.pipeline.run_full_experiment",
    "cli/pipeline/run_post_pipeline.py": "cli.pipeline.run_post_pipeline",
    "cli/pipeline/run_pilot.py": "cli.pipeline.run_pilot",
    "cli/pipeline/run_stages.py": "cli.pipeline.run_stages",
    "cli/s0_fetch/fetch_all.py": "cli.s0_fetch.fetch_all",
    "cli/s0_fetch/fetch_freed_fragments.py": "cli.s0_fetch.fetch_freed_fragments",
    "cli/s0_fetch/fetch_zinc_baseline.py": "cli.s0_fetch.fetch_zinc_baseline",
    "cli/s0_fetch/fetch_calibration_smiles.py": "cli.s0_fetch.fetch_calibration_smiles",
    "cli/s0_fetch/fetch_chembl_actives.py": "cli.s0_fetch.fetch_chembl_actives",
    "cli/s1_prepare/prepare_all.py": "cli.s1_prepare.prepare_all",
    "cli/s1_prepare/prepare_targets.py": "cli.s1_prepare.prepare_targets",
    "cli/s1_prepare/convert_baseline_parquet.py": "cli.s1_prepare.convert_baseline_parquet",
    "cli/s1_prepare/sample_baseline.py": "cli.s1_prepare.sample_baseline",
    "cli/s2_calibrate/dock_calibration.py": "cli.s2_calibrate.dock_calibration",
    "cli/s2_calibrate/compute_normalization.py": "cli.s2_calibrate.compute_normalization",
    "cli/s3_generate/generate.py": "cli.s3_generate.generate",
    "cli/s4_score/score.py": "cli.s4_score.score",
    "cli/s5_analyze/aggregate.py": "cli.s5_analyze.aggregate",
    "cli/s5_analyze/compute_metrics.py": "cli.s5_analyze.compute_metrics",
    "cli/s5_analyze/threshold_sensitivity.py": "cli.s5_analyze.threshold_sensitivity",
    "cli/s5_analyze/proxy_correlation.py": "cli.s5_analyze.proxy_correlation",
    "cli/s5_analyze/summarize.py": "cli.s5_analyze.summarize",
    "cli/s6_report/plot_results.py": "cli.s6_report.plot_results",
    "cli/s6_report/render_molecules.py": "cli.s6_report.render_molecules",
    "cli/s6_report/generate_latex_report.py": "cli.s6_report.generate_latex_report",
    "cli/s7_audit/claim_audit.py": "cli.s7_audit.claim_audit",
}


def _normalize_script_path(script_token: str, project_root: Path) -> str | None:
    """Normalize an invocation token to repo-relative script path."""
    script_path = Path(script_token)
    if not script_path.is_absolute():
        script_path = project_root / script_path
    try:
        return script_path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return None


def parse_script_invocation(cmd: list[str], project_root: Path = PROJECT_ROOT) -> tuple[str, list[str], str] | None:
    """Parse a command if it points to an internal Python script.

    Supported forms:
        - ``uv run python cli/...``
        - ``python cli/...`` (absolute python path also supported)

    Returns:
        Tuple ``(module_path, args, script_rel_path)`` or ``None`` if not
        dispatchable in-process.
    """
    if len(cmd) >= 4 and cmd[0] == "uv" and cmd[1] == "run" and Path(cmd[2]).name.startswith("python"):
        script_token = cmd[3]
        args = cmd[4:]
    elif len(cmd) >= 2 and Path(cmd[0]).name.startswith("python"):
        script_token = cmd[1]
        args = cmd[2:]
    else:
        return None

    script_rel = _normalize_script_path(script_token, project_root)
    if script_rel is None:
        return None
    module = _SCRIPT_TO_MODULE.get(script_rel)
    if module is None:
        return None
    return module, args, script_rel


@contextmanager
def _patched_env(env: dict[str, str] | None) -> Iterator[None]:
    """Temporarily patch process environment variables."""
    if not env:
        yield
        return

    marker = object()
    old_values: dict[str, str | object] = {}
    for key, value in env.items():
        old_values[key] = os.environ.get(key, marker)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old in old_values.items():
            if old is marker:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(old)


@contextmanager
def _patched_cwd(cwd: Path | None) -> Iterator[None]:
    """Temporarily switch current working directory."""
    if cwd is None:
        yield
        return
    prev = Path.cwd()
    os.chdir(cwd)
    try:
        yield
    finally:
        os.chdir(prev)


def run_script_inprocess(
    cmd: list[str],
    *,
    project_root: Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> bool:
    """Execute an internal script command in-process when possible.

    Args:
        cmd: Original command tokens.
        project_root: Repository root for script path normalization.
        env: Optional environment variables to patch for this run.
        cwd: Optional working directory to apply for this run.

    Returns:
        ``True`` if command was handled in-process, ``False`` otherwise.

    Raises:
        RuntimeError: If the underlying click command exits non-zero.
    """
    parsed = parse_script_invocation(cmd, project_root=project_root)
    if parsed is None:
        return False

    module_name, args, script_rel = parsed
    module = importlib.import_module(module_name)
    command = getattr(module, "main", None)
    if command is None or not hasattr(command, "main"):
        msg = f"Module {module_name} does not expose Click command 'main'"
        raise RuntimeError(msg)

    logger.debug("In-process dispatch: {} {}", script_rel, " ".join(args))
    with _patched_cwd(cwd), _patched_env(env):
        try:
            command.main(args=args, standalone_mode=False)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            if code not in {0, None}:
                raise RuntimeError(f"In-process command failed (rc={code}): {' '.join(cmd)}") from exc
    return True
