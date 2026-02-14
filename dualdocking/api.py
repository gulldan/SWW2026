"""Programmatic API for running DualDocking workflows without CLI shells.

These helpers call the same Click commands used by ``cli/`` but execute
them in-process. This enables notebook/service usage while preserving the
existing command behavior.
"""

from __future__ import annotations

from pathlib import Path

from dualdocking.script_dispatch import run_script_inprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(script_rel: str, args: list[str]) -> None:
    """Run one internal script command in-process."""
    cmd = ["python", script_rel, *args]
    handled = run_script_inprocess(cmd, project_root=PROJECT_ROOT, cwd=PROJECT_ROOT)
    if not handled:
        msg = f"Unsupported internal script for API execution: {script_rel}"
        raise RuntimeError(msg)


def run_generate(
    *,
    pair: str,
    mode: str,
    seed: int = 42,
    n_molecules: int | None = None,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
) -> None:
    """Run molecule generation for one ``(pair, mode, seed)`` combination."""
    args = ["--pair", pair, "--mode", mode, "--seed", str(seed)]
    if n_molecules is not None:
        args.extend(["--n-molecules", str(n_molecules)])
    if epochs is not None:
        args.extend(["--epochs", str(epochs)])
    if steps_per_epoch is not None:
        args.extend(["--steps-per-epoch", str(steps_per_epoch)])
    _run("cli/s3_generate/generate.py", args)


def run_score(
    *,
    pair: str,
    mode: str,
    seed: int = 42,
    force: bool = False,
    prune_cache: bool = False,
    molecules: str | None = None,
    parallel_scoring: str = "auto",
) -> None:
    """Run Boltz-2 scoring for one ``(pair, mode, seed)`` combination."""
    args = [
        "--pair",
        pair,
        "--mode",
        mode,
        "--seed",
        str(seed),
        "--parallel-scoring",
        parallel_scoring,
    ]
    if force:
        args.append("--force")
    if prune_cache:
        args.append("--prune-cache")
    if molecules is not None:
        args.extend(["--molecules", molecules])
    _run("cli/s4_score/score.py", args)


def run_full_experiment(
    *,
    dry_run: bool = False,
    parallel: int = 0,
    resume: bool = False,
    pairs: list[str] | None = None,
    seeds: list[int] | None = None,
    prepare_freed_data: bool = True,
    prepare_baseline: bool = False,
    baseline_url: str | None = None,
    with_post: bool = False,
    latex_mode: str = "off",
) -> None:
    """Run the full pipeline (equivalent to ``run_full_experiment.py``)."""
    args = ["--parallel", str(parallel), "--latex-mode", latex_mode]
    if dry_run:
        args.append("--dry-run")
    if resume:
        args.append("--resume")
    if not prepare_freed_data:
        args.append("--skip-prepare-freed-data")
    if prepare_baseline:
        args.append("--prepare-baseline")
    else:
        args.append("--skip-prepare-baseline")
    if baseline_url is not None:
        args.extend(["--baseline-url", baseline_url])
    if with_post:
        args.append("--with-post")
    else:
        args.append("--skip-post")
    if pairs:
        for pair in pairs:
            args.extend(["--pair", pair])
    if seeds:
        args.extend(["--seeds", ",".join(str(s) for s in seeds)])
    _run("cli/pipeline/run_full_experiment.py", args)


def run_post_pipeline(*, dry_run: bool = False, latex_mode: str = "draft") -> None:
    """Run post-processing/report stage only."""
    args = ["--latex-mode", latex_mode]
    if dry_run:
        args.append("--dry-run")
    _run("cli/pipeline/run_post_pipeline.py", args)


def run_stages(
    *,
    stages: str,
    pair: str | None = None,
    seeds: list[int] | None = None,
    molecules: str | None = None,
    latex_mode: str = "off",
    dry_run: bool = False,
    skip_preflight: bool = False,
) -> None:
    """Run selected stages via stage runner."""
    args = ["--stages", stages, "--latex-mode", latex_mode]
    if pair is not None:
        args.extend(["--pair", pair])
    if seeds:
        args.extend(["--seeds", ",".join(str(s) for s in seeds)])
    if molecules is not None:
        args.extend(["--molecules", molecules])
    if dry_run:
        args.append("--dry-run")
    if skip_preflight:
        args.append("--skip-preflight")
    _run("cli/pipeline/run_stages.py", args)
