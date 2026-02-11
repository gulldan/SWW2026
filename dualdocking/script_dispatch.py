"""In-process dispatcher for Click-based project scripts.

This module lets pipeline orchestrators execute internal scripts without
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
    "scripts/pipeline/run_full_experiment.py": "scripts.pipeline.run_full_experiment",
    "scripts/pipeline/run_post_pipeline.py": "scripts.pipeline.run_post_pipeline",
    "scripts/pipeline/run_pilot.py": "scripts.pipeline.run_pilot",
    "scripts/pipeline/run_stages.py": "scripts.pipeline.run_stages",
    "scripts/s0_fetch/fetch_all.py": "scripts.s0_fetch.fetch_all",
    "scripts/s0_fetch/fetch_freed_fragments.py": "scripts.s0_fetch.fetch_freed_fragments",
    "scripts/s0_fetch/fetch_zinc_baseline.py": "scripts.s0_fetch.fetch_zinc_baseline",
    "scripts/s0_fetch/fetch_calibration_smiles.py": "scripts.s0_fetch.fetch_calibration_smiles",
    "scripts/s0_fetch/fetch_chembl_actives.py": "scripts.s0_fetch.fetch_chembl_actives",
    "scripts/s1_prepare/prepare_all.py": "scripts.s1_prepare.prepare_all",
    "scripts/s1_prepare/prepare_targets.py": "scripts.s1_prepare.prepare_targets",
    "scripts/s1_prepare/convert_baseline_parquet.py": "scripts.s1_prepare.convert_baseline_parquet",
    "scripts/s1_prepare/sample_baseline.py": "scripts.s1_prepare.sample_baseline",
    "scripts/s2_calibrate/dock_calibration.py": "scripts.s2_calibrate.dock_calibration",
    "scripts/s2_calibrate/compute_normalization.py": "scripts.s2_calibrate.compute_normalization",
    "scripts/s3_generate/generate.py": "scripts.s3_generate.generate",
    "scripts/s4_score/score.py": "scripts.s4_score.score",
    "scripts/s5_analyze/aggregate.py": "scripts.s5_analyze.aggregate",
    "scripts/s5_analyze/compute_metrics.py": "scripts.s5_analyze.compute_metrics",
    "scripts/s5_analyze/threshold_sensitivity.py": "scripts.s5_analyze.threshold_sensitivity",
    "scripts/s5_analyze/proxy_correlation.py": "scripts.s5_analyze.proxy_correlation",
    "scripts/s5_analyze/summarize.py": "scripts.s5_analyze.summarize",
    "scripts/s6_report/plot_results.py": "scripts.s6_report.plot_results",
    "scripts/s6_report/render_molecules.py": "scripts.s6_report.render_molecules",
    "scripts/s6_report/generate_latex_report.py": "scripts.s6_report.generate_latex_report",
    "scripts/s7_audit/claim_audit.py": "scripts.s7_audit.claim_audit",
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
        - ``uv run python scripts/...``
        - ``python scripts/...`` (absolute python path also supported)

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
