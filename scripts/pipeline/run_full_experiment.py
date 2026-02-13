"""Run full experiment across pairs, modes, and seeds.

Orchestrates the complete DualDocking experiment pipeline: data preparation,
calibration, molecule generation, scoring, and optional post-processing
(aggregation, metrics, plots, rendering, summarization, LaTeX report).

Supports multi-GPU parallelism, resume from partial runs, and run manifests
for provenance tracking.
"""

from __future__ import annotations

import os
import queue
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import click
import yaml
from loguru import logger
from tqdm import tqdm

from dualdocking.command_runner import run_project_command
from dualdocking.config import load_experiment, load_pairs
from dualdocking.reporting.run_manifest import (
    finalize_manifest,
    init_manifest,
    manifest_path,
    skip_task,
    update_task,
)


def _detect_gpu_count() -> int:
    """Detect the number of available CUDA GPUs.

    Checks ``CUDA_VISIBLE_DEVICES`` first, then ``nvidia-smi`` (not affected
    by env restrictions), then ``torch.cuda.device_count()``.

    Returns:
        Number of CUDA devices, or 1 if detection fails.
    """
    # 1. Honour CUDA_VISIBLE_DEVICES if explicitly set
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        devices = [d.strip() for d in cvd.split(",") if d.strip()]
        if devices:
            logger.debug("GPU count via CUDA_VISIBLE_DEVICES: {}", len(devices))
            return len(devices)
        # CUDA_VISIBLE_DEVICES="" → no GPUs
        logger.debug("CUDA_VISIBLE_DEVICES is empty, assuming 1 GPU")
        return 1

    # 2. nvidia-smi (shows physical GPUs, unaffected by env vars)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],  # noqa: S603, S607
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
            if lines:
                logger.debug("GPU count via nvidia-smi: {}", len(lines))
                return len(lines)
    except Exception:
        pass

    # 3. Fallback: torch
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            if n >= 1:
                logger.debug("GPU count via torch: {}", n)
                return n
    except Exception:
        pass

    logger.debug("GPU detection failed, assuming 1 GPU")
    return 1


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _ensure_calibrated(pair_id: str, pair_cfg: dict[str, object], log_path: Path) -> None:
    """Ensure both targets in a pair have calibrated normalization statistics.

    If any target is uncalibrated (n_samples=0 or stable=false), runs the full
    calibration pipeline: prepare_targets -> dock_calibration -> compute_normalization.
    Fails hard if calibration is still incomplete after running.

    Args:
        pair_id: Identifier for the target pair (e.g. ``"CDK2_ERK2"``).
        pair_cfg: Pair configuration dict with ``target_a`` and ``target_b`` keys.
        log_path: Path to the experiment log file.

    Raises:
        RuntimeError: If ``normalization.yaml`` is missing or calibration fails.
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
        ["uv", "run", "python", "scripts/s1_prepare/prepare_targets.py", "--pair", pair_id, "--prepare-all"],
        log_path,
    )

    # Step 2: dock calibration for uncalibrated targets only
    dock_cmd = ["uv", "run", "python", "scripts/s2_calibrate/dock_calibration.py"]
    for tid in uncalibrated:
        dock_cmd.extend(["--target-id", tid])
    _run_cmd(dock_cmd, log_path)

    # Step 3: regenerate normalization.yaml
    _run_cmd(["uv", "run", "python", "scripts/s2_calibrate/compute_normalization.py"], log_path)

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
    """Append a timestamped message to the experiment log file.

    Args:
        log_path: Path to the log file (parent dirs created if needed).
        message: Line to append.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(message + "\n")


def _ts() -> str:
    """Return the current UTC time as an ISO-8601 string with ``Z`` suffix."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _command_generate(pair: str, mode: str, seed: int) -> list[str]:
    """Build the CLI command to run the molecule generation stage.

    Args:
        pair: Target pair identifier.
        mode: Experiment mode (``single_a``, ``single_b``, or ``double``).
        seed: Random seed for the run.

    Returns:
        Command tokens suitable for ``subprocess.run``.
    """
    return ["uv", "run", "python", "scripts/s3_generate/generate.py", "--pair", pair, "--mode", mode, "--seed", str(seed)]


def _command_score(pair: str, mode: str, seed: int) -> list[str]:
    """Build the CLI command to run the scoring stage.

    Args:
        pair: Target pair identifier.
        mode: Experiment mode (``single_a``, ``single_b``, or ``double``).
        seed: Random seed for the run.

    Returns:
        Command tokens suitable for ``subprocess.run``.
    """
    return ["uv", "run", "python", "scripts/s4_score/score.py", "--pair", pair, "--mode", mode, "--seed", str(seed)]


def _post_commands(latex_mode: str) -> list[list[str]]:
    """Build the ordered list of post-pipeline commands.

    Args:
        latex_mode: LaTeX generation mode (``"off"``, ``"draft"``, or ``"full"``).

    Returns:
        List of command token lists for aggregation, metrics, plots, rendering,
        summarization, and optionally LaTeX report generation.
    """
    commands = [
        ["uv", "run", "python", "scripts/s5_analyze/aggregate.py"],
        ["uv", "run", "python", "scripts/s5_analyze/compute_metrics.py"],
        ["uv", "run", "python", "scripts/s6_report/plot_results.py"],
        ["uv", "run", "python", "scripts/s6_report/render_molecules.py"],
        ["uv", "run", "python", "scripts/s5_analyze/summarize.py"],
    ]
    if latex_mode != "off":
        commands.append(["uv", "run", "python", "scripts/s6_report/generate_latex_report.py", "--mode", latex_mode])
    return commands


def _prepare_commands(prepare_freed_data: bool, prepare_baseline: bool, baseline_url: str | None) -> list[list[str]]:
    """Build data-preparation commands based on the requested options.

    Args:
        prepare_freed_data: Whether to fetch FREED++ data dependencies.
        prepare_baseline: Whether to fetch the ZINC baseline archive.
        baseline_url: Optional custom URL for the baseline archive.

    Returns:
        List containing zero or one command token lists.
    """
    if not prepare_freed_data and not prepare_baseline:
        return []

    cmd = ["uv", "run", "python", "scripts/s1_prepare/prepare_all.py"]
    if not prepare_freed_data:
        cmd.append("--skip-fetch-freed-data")
    if prepare_baseline:
        cmd.append("--fetch-baseline")
        if baseline_url:
            cmd.extend(["--baseline-url", baseline_url])
    return [cmd]


# Module-level scorer config (set in main() from experiment.yaml)
_SCORER_ENGINE: str = "boltz2"
_SCORER_BOLTZ_BACKEND: str = "cli"
_SCORER_NIM_URLS: list[str] = ["http://localhost:8000"]

_NIM_HEALTH_ENDPOINT = "/v1/health/ready"
_NIM_HEALTH_TIMEOUT = 5.0  # seconds


def _check_nim_health(urls: list[str]) -> list[str]:
    """Probe NIM URLs and return only healthy ones.

    Args:
        urls: List of NIM base URLs to check.

    Returns:
        Subset of *urls* that responded as healthy.
    """
    import urllib.request

    healthy: list[str] = []
    for url in urls:
        health_url = f"{url.rstrip('/')}{_NIM_HEALTH_ENDPOINT}"
        try:
            req = urllib.request.Request(health_url)  # noqa: S310
            with urllib.request.urlopen(req, timeout=_NIM_HEALTH_TIMEOUT) as resp:  # noqa: S310
                body = resp.read().lower()
            if b"false" not in body:
                healthy.append(url)
                logger.info("NIM health OK: {}", url)
            else:
                logger.warning("NIM not ready: {} (responded false)", url)
        except Exception as exc:
            logger.warning("NIM not reachable: {} ({})", url, exc)
    return healthy


def _canonical_scorer_engine(name: str) -> str:
    """Normalize scorer engine name to canonical built-in aliases."""
    normalized = "".join(ch for ch in name.lower() if ch.isalnum())
    if normalized in {"boltz", "boltz2"}:
        return "boltz2"
    return name.strip()


def _run_cmd(cmd: list[str], log_path: Path, gpu_id: int | None = None) -> None:
    """Execute a subprocess command, logging output and enforcing success.

    Sets ``FREEDPP_CUDA_DEVICE``, ``SCORER_CUDA_DEVICE``, and scorer-specific
    environment variables before execution.

    Args:
        cmd: Command tokens to execute.
        log_path: Path to the experiment log file.
        gpu_id: CUDA device index to pin execution to, or ``None`` for default.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero return code.
    """
    env = dict(os.environ)
    if gpu_id is not None:
        env["FREEDPP_CUDA_DEVICE"] = str(gpu_id)
        env["SCORER_CUDA_DEVICE"] = str(gpu_id)
    env["DUALDOCKING_SCORER_ENGINE"] = _SCORER_ENGINE

    if _SCORER_ENGINE == "boltz2":
        if gpu_id is not None:
            env["BOLTZ_CUDA_DEVICE"] = str(gpu_id)
        env["BOLTZ_BACKEND"] = _SCORER_BOLTZ_BACKEND
        if _SCORER_BOLTZ_BACKEND == "nim":
            # Pass ALL NIM URLs for round-robin across GPU instances.
            env["BOLTZ_NIM_URLS"] = ",".join(_SCORER_NIM_URLS)
            env["BOLTZ_NIM_URL"] = _SCORER_NIM_URLS[0]  # Fallback for single-URL code paths.
        else:
            env.pop("BOLTZ_NIM_URLS", None)
            env.pop("BOLTZ_NIM_URL", None)
    else:
        env.pop("BOLTZ_BACKEND", None)
        env.pop("BOLTZ_NIM_URLS", None)
        env.pop("BOLTZ_NIM_URL", None)
        env.pop("BOLTZ_CUDA_DEVICE", None)
    _log_to_file(log_path, f"[{_ts()}] RUN (gpu={gpu_id}): {' '.join(cmd)}")
    result = run_project_command(
        cmd,
        project_root=PROJECT_ROOT,
        env=env,
        stdout_pipe=True,
        stderr_to_parent=True,
        allow_inprocess=True,
    )
    if result.returncode != 0:
        _log_to_file(log_path, f"[{_ts()}] ERROR (rc={result.returncode}):\n{(result.stdout or '')[-4000:]}")
        raise RuntimeError(f"Command failed (rc={result.returncode}): {' '.join(cmd)}")
    if result.stdout:
        _log_to_file(log_path, f"[{_ts()}] OUT: {result.stdout[-2000:]}")


def _run_one(
    pair: str, mode: str, seed: int, log_path: Path, resume: bool, mf_path: Path | None = None, gpu_id: int | None = None
) -> None:
    """Run the generate + score pipeline for a single (pair, mode, seed) task.

    Skips entirely if ``scored_joined`` already exists (Parquet or CSV). In
    resume mode, also skips generation if ``generated`` exists (Parquet or CSV).

    Args:
        pair: Target pair identifier.
        mode: Experiment mode (``single_a``, ``single_b``, or ``double``).
        seed: Random seed for the run.
        log_path: Path to the experiment log file.
        resume: If ``True``, skip generation when ``generated`` already exists (Parquet or CSV).
        mf_path: Path to the run manifest file, or ``None`` to skip tracking.
        gpu_id: CUDA device index for this task, or ``None`` for default.

    Raises:
        RuntimeError: If either the generate or score subprocess fails.
    """
    run_dir = PROJECT_ROOT / "output" / "results" / pair / mode / f"seed_{seed}"
    scored = run_dir / "scored_joined"
    generated = run_dir / "generated"
    tid = f"{pair}/{mode}/seed_{seed}"

    from dualdocking.data.io_utils import stem_exists

    if stem_exists(scored):
        logger.info("[SKIP] {} {} seed_{} (scored_joined exists)", pair, mode, seed)
        if mf_path:
            skip_task(mf_path, task_id=tid)
        return

    gpu_tag = f" [GPU {gpu_id}]" if gpu_id is not None else ""

    # --- Generate stage ---
    if stem_exists(generated) and resume:
        logger.info("[RESUME] {} {} seed_{} (generated exists)", pair, mode, seed)
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
    help="Before running, check/download docker/freedpp/data/{zinc_crem.json,alert_collections.csv}."
)
@click.option(
    "--prepare-baseline/--skip-prepare-baseline",
    default=False,
    show_default=True,
    help="Optionally download baseline archive data/baseline/zinc-druglike-cano.tar.xz."
)
@click.option(
    "--baseline-url",
    default=None,
    help="URL for baseline archive (or use ZINC_BASELINE_URL environment variable)."
)
@click.option(
    "--with-post/--skip-post",
    default=False,
    show_default=True,
    help="Optionally run post-pipeline (aggregate/metrics/plots/render/summarize/latex) after generate+score."
)
@click.option(
    "--latex-mode",
    type=click.Choice(["off", "draft", "full"]),
    default="off",
    show_default=True,
    help="LaTeX mode, only used with --with-post."
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
    """Run the full DualDocking experiment pipeline.

    Loads experiment and pair configs, optionally prepares data, ensures
    calibration, then dispatches generate+score tasks across GPUs. Optionally
    runs post-processing (aggregate, metrics, plots, render, summarize, LaTeX).

    Args:
        dry_run: Print commands without executing.
        parallel: Number of parallel tasks (0 = auto-detect from GPU count).
        resume: Skip generation for runs where ``generated`` already exists (Parquet or CSV).
        pair: Filter to specific pair(s); empty tuple means all pairs.
        seeds_override: Comma-separated seed list, or ``None`` for config default.
        prepare_freed_data: Fetch FREED++ data dependencies before running.
        prepare_baseline: Fetch the ZINC baseline archive before running.
        baseline_url: Custom URL for the baseline archive.
        with_post: Run post-processing pipeline after generate+score.
        latex_mode: LaTeX report mode (``"off"``, ``"draft"``, or ``"full"``).
    """
    global _SCORER_ENGINE, _SCORER_BOLTZ_BACKEND, _SCORER_NIM_URLS

    if parallel <= 0:
        parallel = _detect_gpu_count()
        logger.info("Auto-detected {} GPU(s) → --parallel={}", parallel, parallel)

    exp_cfg = load_experiment(PROJECT_ROOT / "configs")
    all_pairs = load_pairs(PROJECT_ROOT / "configs")["pairs"]
    active_pairs = {pair_id: pair_cfg for pair_id, pair_cfg in all_pairs.items() if pair_cfg.get("set") != "blocked"}

    # Configure scorer backend from experiment.yaml.
    scorer_cfg = exp_cfg.get("scorer", {})
    engine_raw = scorer_cfg.get("engine", scorer_cfg.get("name", "boltz2"))
    _SCORER_ENGINE = _canonical_scorer_engine(str(engine_raw))
    if _SCORER_ENGINE == "boltz2":
        _SCORER_BOLTZ_BACKEND = str(scorer_cfg.get("backend", "cli")).lower()
        if _SCORER_BOLTZ_BACKEND == "nim":
            nim_cfg = scorer_cfg.get("nim", {}) if isinstance(scorer_cfg.get("nim"), dict) else {}
            base_port = int(nim_cfg.get("base_port", 8000))

            # Auto-generate NIM URLs: one per detected GPU, starting at base_port.
            n_gpus = _detect_gpu_count()
            _SCORER_NIM_URLS = [f"http://localhost:{base_port + i}" for i in range(n_gpus)]
            logger.info(
                "Scorer engine: boltz2 (backend: NIM, {} GPU(s) → {} instance(s): {})",
                n_gpus,
                len(_SCORER_NIM_URLS),
                ", ".join(_SCORER_NIM_URLS),
            )

            # Health-check: probe each URL and warn/remove dead ones before starting.
            _healthy_urls = _check_nim_health(_SCORER_NIM_URLS)
            if not _healthy_urls:
                raise RuntimeError(
                    f"No NIM Boltz-2 instances are healthy. "
                    f"Expected {len(_SCORER_NIM_URLS)} instance(s) at ports "
                    f"{base_port}-{base_port + n_gpus - 1}. "
                    f"Start NIM containers first (see docs/setup.md)."
                )
            if len(_healthy_urls) < len(_SCORER_NIM_URLS):
                dead = [u for u in _SCORER_NIM_URLS if u not in _healthy_urls]
                logger.warning(
                    "Only {}/{} NIM instances healthy. Dead: {}. Proceeding with healthy only.",
                    len(_healthy_urls),
                    len(_SCORER_NIM_URLS),
                    ", ".join(dead),
                )
            _SCORER_NIM_URLS = _healthy_urls
        else:
            logger.info("Scorer engine: boltz2 (backend: CLI)")
    else:
        logger.info("Scorer engine: {}", _SCORER_ENGINE)
    modes = exp_cfg.get("modes", ["single_a", "single_b", "double"])
    all_seeds = exp_cfg.get("seeds", [42, 137, 256, 512, 1024])

    # Apply filters
    pairs = {k: v for k, v in active_pairs.items() if k in pair} if pair else active_pairs
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
    results_dir = PROJECT_ROOT / "output" / "results"
    mf_path = manifest_path(results_dir, run_id)
    init_manifest(
        mf_path,
        run_id=run_id,
        config_dir=PROJECT_ROOT / "configs",
        tasks=tasks,
        parallel=parallel,
        resume=resume,
    )

    run_failed = False
    try:
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

        # Cap effective parallelism at GPU count — running more tasks than GPUs
        # causes contention (FREED++ and Boltz-2 each need a full GPU).
        effective_parallel = min(parallel, n_gpus)
        if effective_parallel < parallel:
            logger.warning(
                "Capping --parallel {} to {} (available GPUs). "
                "Cannot run more concurrent tasks than GPUs.",
                parallel,
                effective_parallel,
            )

        if effective_parallel <= 1:
            pbar = tqdm(tasks, desc="Tasks", unit="task", dynamic_ncols=True)
            for pair_id, mode, seed in pbar:
                pbar.set_postfix_str(f"{pair_id}/{mode}/seed_{seed}", refresh=True)
                try:
                    _run_one(pair_id, mode, seed, log_path, resume, mf_path, gpu_id=0)
                    n_completed += 1
                except RuntimeError:
                    n_failed += 1
                    run_failed = True
            pbar.set_postfix_str(f"done={n_completed} failed={n_failed}")
            pbar.close()
        else:
            # GPU pool: each worker acquires a GPU id before running, releases after.
            # Pool has exactly n_gpus items so at most 1 task per GPU at any time.
            gpu_pool: queue.Queue[int] = queue.Queue()
            for g in range(n_gpus):
                gpu_pool.put(g)

            def _run_with_gpu_pool(
                pair_id: str, mode: str, seed: int,
            ) -> None:
                gid = gpu_pool.get()
                try:
                    _run_one(pair_id, mode, seed, log_path, resume, mf_path, gpu_id=gid)
                finally:
                    gpu_pool.put(gid)

            with ThreadPoolExecutor(max_workers=effective_parallel) as executor:
                futures = {}
                for pair_id, mode, seed in tasks:
                    fut = executor.submit(_run_with_gpu_pool, pair_id, mode, seed)
                    futures[fut] = f"{pair_id}/{mode}/seed_{seed}"
                with tqdm(total=total, desc="Tasks", unit="task", dynamic_ncols=True) as pbar:
                    for future in as_completed(futures):
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
    except Exception:
        run_failed = True
        raise
    finally:
        finalize_manifest(mf_path, status="failed" if run_failed else "completed")


if __name__ == "__main__":
    main()
