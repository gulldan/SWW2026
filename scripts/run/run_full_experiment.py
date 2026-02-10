"""Run full experiment across pairs, modes, and seeds."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import click
import yaml
from loguru import logger
from tqdm import tqdm

from dualdocking.config import load_experiment, load_pairs
from dualdocking.reporting.run_manifest import (
    finalize_manifest,
    init_manifest,
    manifest_path,
    skip_task,
    update_task,
)


def _detect_gpu_count() -> int:
    """Detect number of CUDA GPUs; returns 1 if detection fails."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        logger.debug("GPU detection failed, assuming 1 GPU")
    return 1


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _ensure_calibrated(pair_id: str, pair_cfg: dict, log_path: Path) -> None:
    """Ensure both targets in a pair have calibrated normalization statistics.

    If any target is uncalibrated (n_samples=0 or stable=false), runs the full
    calibration pipeline: prepare_targets → dock_calibration → calibrate_normalization.
    Fails hard if calibration still incomplete after running.
    """
    norm_path = PROJECT_ROOT / "configs" / "normalization.yaml"
    if not norm_path.exists():
        raise RuntimeError(f"normalization.yaml not found: {norm_path}")

    norm_data = yaml.safe_load(norm_path.read_text())
    norm_targets = norm_data.get("targets", {})

    uncalibrated: list[str] = []
    for role in ("target_a", "target_b"):
        tid = pair_cfg[role]["id"]
        entry = norm_targets.get(tid, {})
        if not entry or entry.get("n_samples", 0) == 0 or not entry.get("stable_mean") or not entry.get("stable_std"):
            uncalibrated.append(tid)

    if not uncalibrated:
        return

    _log_to_file(log_path, f"[{_ts()}] CALIBRATE {pair_id}: uncalibrated targets = {uncalibrated}")
    logger.info("Calibrating targets for pair {}: {}", pair_id, uncalibrated)

    # Step 1: prepare target data
    _run_cmd(
        ["uv", "run", "python", "scripts/prep/prepare_targets.py", "--pair", pair_id, "--prepare-all"],
        log_path,
    )

    # Step 2: dock calibration for uncalibrated targets only
    dock_cmd = ["uv", "run", "python", "scripts/validation/dock_calibration.py"]
    for tid in uncalibrated:
        dock_cmd.extend(["--target-id", tid])
    _run_cmd(dock_cmd, log_path)

    # Step 3: regenerate normalization.yaml
    _run_cmd(["uv", "run", "python", "scripts/validation/calibrate_normalization.py"], log_path)

    # Re-read and validate calibration succeeded
    norm_data = yaml.safe_load(norm_path.read_text())
    norm_targets = norm_data.get("targets", {})
    still_bad: list[str] = []
    for tid in uncalibrated:
        entry = norm_targets.get(tid, {})
        if not entry or entry.get("n_samples", 0) == 0 or not entry.get("stable_mean") or not entry.get("stable_std"):
            still_bad.append(tid)

    if still_bad:
        raise RuntimeError(
            f"Calibration failed for targets {still_bad} in pair {pair_id}. "
            "Check dock_calibration.py output and data/calibration/scores/."
        )

    _log_to_file(log_path, f"[{_ts()}] CALIBRATE {pair_id}: OK")
    logger.info("Calibration complete for pair {}", pair_id)


def _log_to_file(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(message + "\n")


def _ts() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _command_generate(pair: str, mode: str, seed: int) -> list[str]:
    return ["uv", "run", "python", "scripts/run/generate.py", "--pair", pair, "--mode", mode, "--seed", str(seed)]


def _command_score(pair: str, mode: str, seed: int) -> list[str]:
    return ["uv", "run", "python", "scripts/run/score.py", "--pair", pair, "--mode", mode, "--seed", str(seed)]


def _post_commands(latex_mode: str) -> list[list[str]]:
    commands = [
        ["uv", "run", "python", "scripts/analysis/aggregate.py"],
        ["uv", "run", "python", "scripts/analysis/compute_metrics.py"],
        ["uv", "run", "python", "scripts/viz/plot_results.py"],
        ["uv", "run", "python", "scripts/viz/render_molecules.py"],
        ["uv", "run", "python", "scripts/analysis/summarize.py"],
    ]
    if latex_mode != "off":
        commands.append(["uv", "run", "python", "scripts/viz/generate_latex_report.py", "--mode", latex_mode])
    return commands


def _prepare_commands(prepare_freed_data: bool, prepare_baseline: bool, baseline_url: str | None) -> list[list[str]]:
    if not prepare_freed_data and not prepare_baseline:
        return []

    cmd = ["uv", "run", "python", "scripts/prep/prepare_data_dependencies.py"]
    if not prepare_freed_data:
        cmd.append("--skip-fetch-freed-data")
    if prepare_baseline:
        cmd.append("--fetch-baseline")
        if baseline_url:
            cmd.extend(["--baseline-url", baseline_url])
    return [cmd]


# Module-level scorer config (set in main() from experiment.yaml)
_SCORER_BACKEND: str = "cli"
_NIM_URLS: list[str] = ["http://localhost:8000"]


def _run_cmd(cmd: list[str], log_path: Path, gpu_id: int | None = None) -> None:
    env = dict(os.environ)
    if gpu_id is not None:
        env["FREEDPP_CUDA_DEVICE"] = str(gpu_id)
        env["BOLTZ_CUDA_DEVICE"] = str(gpu_id)
    # Boltz-2 backend routing
    env["BOLTZ_BACKEND"] = _SCORER_BACKEND
    if _SCORER_BACKEND == "nim":
        # Pass ALL NIM URLs for round-robin across GPU instances
        env["BOLTZ_NIM_URLS"] = ",".join(_NIM_URLS)
        env["BOLTZ_NIM_URL"] = _NIM_URLS[0]  # fallback for single-URL code paths
    _log_to_file(log_path, f"[{_ts()}] RUN (gpu={gpu_id}): {' '.join(cmd)}")
    result = subprocess.run(  # noqa: S603
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        _log_to_file(log_path, f"[{_ts()}] ERROR (rc={result.returncode}):\n{(result.stdout or '')[-4000:]}")
        raise RuntimeError(f"Command failed (rc={result.returncode}): {' '.join(cmd)}")
    if result.stdout:
        _log_to_file(log_path, f"[{_ts()}] OUT: {result.stdout[-2000:]}")


def _run_one(
    pair: str, mode: str, seed: int, log_path: Path, resume: bool, mf_path: Path | None = None, gpu_id: int | None = None
) -> None:
    run_dir = PROJECT_ROOT / "results" / pair / mode / f"seed_{seed}"
    scored = run_dir / "scored_joined.csv"
    generated = run_dir / "generated.csv"
    tid = f"{pair}/{mode}/seed_{seed}"

    if scored.exists():
        logger.info("[SKIP] {} {} seed_{} (scored_joined.csv exists)", pair, mode, seed)
        if mf_path:
            skip_task(mf_path, task_id=tid)
        return

    gpu_tag = f" [GPU {gpu_id}]" if gpu_id is not None else ""

    # --- Generate stage ---
    if generated.exists() and resume:
        logger.info("[RESUME] {} {} seed_{} (generated.csv exists)", pair, mode, seed)
        if mf_path:
            update_task(mf_path, task_id=tid, stage="generate", status="skipped")
    else:
        logger.info("[GENERATE]{} {} {} seed_{} — started", gpu_tag, pair, mode, seed)
        if mf_path:
            update_task(mf_path, task_id=tid, stage="generate", status="running")
        t0 = time.monotonic()
        try:
            _run_cmd(_command_generate(pair, mode, seed), log_path, gpu_id=gpu_id)
        except RuntimeError as exc:
            logger.error("[GENERATE]{} {} {} seed_{} — FAILED ({:.0f}s)", gpu_tag, pair, mode, seed, time.monotonic() - t0)
            if mf_path:
                update_task(mf_path, task_id=tid, stage="generate", status="failed", error=str(exc))
            raise
        elapsed = time.monotonic() - t0
        logger.info("[GENERATE]{} {} {} seed_{} — done ({:.0f}s)", gpu_tag, pair, mode, seed, elapsed)
        if mf_path:
            update_task(mf_path, task_id=tid, stage="generate", status="completed", duration_seconds=elapsed)

    # --- Score stage ---
    logger.info("[SCORE]{} {} {} seed_{} — started", gpu_tag, pair, mode, seed)
    if mf_path:
        update_task(mf_path, task_id=tid, stage="score", status="running")
    t0 = time.monotonic()
    try:
        _run_cmd(_command_score(pair, mode, seed), log_path, gpu_id=gpu_id)
    except RuntimeError as exc:
        logger.error("[SCORE]{} {} {} seed_{} — FAILED ({:.0f}s)", gpu_tag, pair, mode, seed, time.monotonic() - t0)
        if mf_path:
            update_task(mf_path, task_id=tid, stage="score", status="failed", error=str(exc))
        raise
    elapsed = time.monotonic() - t0
    logger.info("[SCORE]{} {} {} seed_{} — done ({:.0f}s)", gpu_tag, pair, mode, seed, elapsed)
    if mf_path:
        update_task(mf_path, task_id=tid, stage="score", status="completed", duration_seconds=elapsed)


@click.command()
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--parallel", default=0, type=int, help="Parallel tasks. 0 = auto-detect (# GPUs).")
@click.option("--resume", is_flag=True, default=False)
@click.option(
    "--pair",
    multiple=True,
    default=None,
    help="Filter to specific pair(s). Can be repeated. Default: all pairs from config.",
)
@click.option(
    "--seeds",
    "seeds_override",
    default=None,
    help="Comma-separated seed list override. Default: all seeds from config.",
)
@click.option(
    "--prepare-freed-data/--skip-prepare-freed-data",
    default=True,
    show_default=True,
    help="Перед запуском проверить/скачать docker/freedpp/data/{zinc_crem.json,alert_collections.csv}.",
)
@click.option(
    "--prepare-baseline/--skip-prepare-baseline",
    default=False,
    show_default=True,
    help="Опционально скачать baseline archive data/baseline/zinc-druglike-cano.tar.xz.",
)
@click.option(
    "--baseline-url",
    default=None,
    help="URL для baseline archive (или используйте переменную окружения ZINC_BASELINE_URL).",
)
@click.option(
    "--with-post/--skip-post",
    default=False,
    show_default=True,
    help="Опционально выполнить post-пайплайн (aggregate/metrics/plots/render/summarize/latex) после generate+score.",
)
@click.option(
    "--latex-mode",
    type=click.Choice(["off", "draft", "full"]),
    default="off",
    show_default=True,
    help="Режим LaTeX, используется только с --with-post.",
)
def main(
    dry_run: bool,
    parallel: int,
    resume: bool,
    pair: tuple[str, ...],
    seeds_override: str | None,
    prepare_freed_data: bool,
    prepare_baseline: bool,
    baseline_url: str | None,
    with_post: bool,
    latex_mode: str,
) -> None:
    global _SCORER_BACKEND, _NIM_URLS

    if parallel <= 0:
        parallel = _detect_gpu_count()
        logger.info("Auto-detected {} GPU(s) → --parallel={}", parallel, parallel)

    exp_cfg = load_experiment(PROJECT_ROOT / "configs")
    all_pairs = load_pairs(PROJECT_ROOT / "configs")["pairs"]

    # Configure scorer backend from experiment.yaml
    scorer_cfg = exp_cfg.get("scorer", {})
    _SCORER_BACKEND = scorer_cfg.get("backend", "cli").lower()
    if _SCORER_BACKEND == "nim":
        nim_cfg = scorer_cfg.get("nim", {})
        _NIM_URLS = nim_cfg.get("urls", ["http://localhost:8000"])
        logger.info("Scorer backend: NIM ({} instance(s): {})", len(_NIM_URLS), ", ".join(_NIM_URLS))
    else:
        logger.info("Scorer backend: CLI (boltz predict subprocess)")
    modes = exp_cfg.get("modes", ["single_a", "single_b", "double"])
    all_seeds = exp_cfg.get("seeds", [42, 137, 256, 512, 1024])

    # Apply filters
    pairs = {k: v for k, v in all_pairs.items() if k in pair} if pair else all_pairs
    seeds = [int(s) for s in seeds_override.split(",")] if seeds_override else all_seeds

    if pair and not pairs:
        logger.error("No matching pairs found for filter: {}", pair)
        raise SystemExit(1)

    run_id = os.environ.get("EXPERIMENT_RUN_ID", "adhoc")
    log_ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = PROJECT_ROOT / "report" / f"experiment_{run_id}_{log_ts}.log"

    tasks = [(pair_id, mode, seed) for pair_id in pairs for mode in modes for seed in seeds]
    start_line = (
        f"[{_ts()}] START run_id={run_id} parallel={parallel} resume={resume} "
        f"prepare_freed_data={prepare_freed_data} prepare_baseline={prepare_baseline} "
        f"with_post={with_post} latex_mode={latex_mode}"
    )
    _log_to_file(log_path, start_line)
    _log_to_file(log_path, f"[{_ts()}] TASKS total={len(tasks)}")
    prep_cmds = _prepare_commands(prepare_freed_data, prepare_baseline, baseline_url)

    if dry_run:
        for cmd in prep_cmds:
            click.echo(" ".join(cmd))
        for pair_id, mode, seed in tasks:
            cmd = " ".join(_command_generate(pair_id, mode, seed)) + " && " + " ".join(_command_score(pair_id, mode, seed))
            click.echo(cmd)
        if with_post:
            for cmd in _post_commands(latex_mode):
                click.echo(" ".join(cmd))
        return

    # Initialize run manifest
    results_dir = PROJECT_ROOT / "results"
    mf_path = manifest_path(results_dir, run_id)
    init_manifest(
        mf_path,
        run_id=run_id,
        config_dir=PROJECT_ROOT / "configs",
        tasks=tasks,
        parallel=parallel,
        resume=resume,
    )

    if prep_cmds:
        logger.info("[PREP] Checking data dependencies...")
        for cmd in prep_cmds:
            _run_cmd(cmd, log_path)
        logger.info("[PREP] Data dependencies ready")

    # Ensure normalization calibration for all pairs before dispatching tasks
    unique_pairs = list(dict.fromkeys(p for p, _m, _s in tasks))
    logger.info("[CALIBRATION] Checking {} pair(s): {}", len(unique_pairs), ", ".join(unique_pairs))
    for pair_id in unique_pairs:
        _ensure_calibrated(pair_id, pairs[pair_id], log_path)
    logger.info("[CALIBRATION] All pairs calibrated")

    n_gpus = _detect_gpu_count()
    total = len(tasks)
    n_completed = 0
    n_failed = 0

    logger.info("Starting {} tasks (parallel={}, GPUs={})", total, parallel, n_gpus)

    run_failed = False
    if parallel <= 1:
        pbar = tqdm(tasks, desc="Tasks", unit="task", dynamic_ncols=True)
        for pair_id, mode, seed in pbar:
            pbar.set_postfix_str(f"{pair_id}/{mode}/seed_{seed}", refresh=True)
            try:
                _run_one(pair_id, mode, seed, log_path, resume, mf_path, gpu_id=0 if n_gpus == 1 else None)
                n_completed += 1
            except RuntimeError:
                n_failed += 1
                run_failed = True
        pbar.set_postfix_str(f"done={n_completed} failed={n_failed}")
        pbar.close()
    else:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {}
            for i, (pair_id, mode, seed) in enumerate(tasks):
                gid = i % n_gpus if n_gpus > 1 else None
                fut = executor.submit(_run_one, pair_id, mode, seed, log_path, resume, mf_path, gpu_id=gid)
                futures[fut] = f"{pair_id}/{mode}/seed_{seed}"
            with tqdm(total=total, desc="Tasks", unit="task", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    tid_label = futures[future]
                    try:
                        future.result()
                        n_completed += 1
                    except RuntimeError:
                        n_failed += 1
                        run_failed = True
                    pbar.update(1)
                    pbar.set_postfix_str(f"done={n_completed} failed={n_failed}")

    logger.info("All tasks finished: {} completed, {} failed out of {}", n_completed, n_failed, total)

    if not with_post:
        _log_to_file(log_path, f"[{_ts()}] POST-SKIP (stage1 only)")
        finalize_manifest(mf_path, status="failed" if run_failed else "completed")
        return

    _log_to_file(log_path, f"[{_ts()}] POST-START")
    post_cmds = _post_commands(latex_mode)
    post_names = ["aggregate", "metrics", "plots", "render_molecules", "summarize", "latex"]
    for i, cmd in enumerate(tqdm(post_cmds, desc="Post-pipeline", unit="step", dynamic_ncols=True)):
        step_name = post_names[i] if i < len(post_names) else cmd[-1]
        logger.info("[POST] {} — started", step_name)
        _run_cmd(cmd, log_path)
        logger.info("[POST] {} — done", step_name)
    _log_to_file(log_path, f"[{_ts()}] POST-DONE")
    finalize_manifest(mf_path, status="failed" if run_failed else "completed")


if __name__ == "__main__":
    main()
