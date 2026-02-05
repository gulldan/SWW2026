"""Run full experiment across pairs, modes, and seeds."""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import click
from loguru import logger

from dualdocking.config import load_experiment, load_pairs

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _log_to_file(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(message + "\n")


def _ts() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _command_generate(pair: str, mode: str, seed: int) -> list[str]:
    return ["uv", "run", "python", "scripts/generate.py", "--pair", pair, "--mode", mode, "--seed", str(seed)]


def _command_score(pair: str, mode: str, seed: int) -> list[str]:
    return ["uv", "run", "python", "scripts/score.py", "--pair", pair, "--mode", mode, "--seed", str(seed)]


def _post_commands(latex_mode: str) -> list[list[str]]:
    commands = [
        ["uv", "run", "python", "scripts/aggregate.py"],
        ["uv", "run", "python", "scripts/compute_metrics.py"],
        ["uv", "run", "python", "scripts/plot_results.py"],
        ["uv", "run", "python", "scripts/render_molecules.py"],
        ["uv", "run", "python", "scripts/summarize.py"],
    ]
    if latex_mode != "off":
        commands.append(["uv", "run", "python", "scripts/generate_latex_report.py", "--mode", latex_mode])
    return commands


def _prepare_commands(prepare_freed_data: bool, prepare_baseline: bool, baseline_url: str | None) -> list[list[str]]:
    if not prepare_freed_data and not prepare_baseline:
        return []

    cmd = ["uv", "run", "python", "scripts/prepare_data_dependencies.py"]
    if not prepare_freed_data:
        cmd.append("--skip-fetch-freed-data")
    if prepare_baseline:
        cmd.append("--fetch-baseline")
        if baseline_url:
            cmd.extend(["--baseline-url", baseline_url])
    return [cmd]


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


def _run_one(pair: str, mode: str, seed: int, log_path: Path, resume: bool) -> None:
    run_dir = PROJECT_ROOT / "results" / pair / mode / f"seed_{seed}"
    scored = run_dir / "scored_joined.csv"
    generated = run_dir / "generated.csv"

    if scored.exists():
        logger.info("[SKIP] {} {} seed_{} (scored_joined.csv exists)", pair, mode, seed)
        return

    if generated.exists() and resume:
        logger.info("[RESUME] {} {} seed_{} (generated.csv exists)", pair, mode, seed)
    else:
        _run_cmd(_command_generate(pair, mode, seed), log_path)

    _run_cmd(_command_score(pair, mode, seed), log_path)


@click.command()
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--parallel", default=1, type=int)
@click.option("--resume", is_flag=True, default=False)
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
    prepare_freed_data: bool,
    prepare_baseline: bool,
    baseline_url: str | None,
    with_post: bool,
    latex_mode: str,
) -> None:
    exp_cfg = load_experiment(PROJECT_ROOT / "configs")
    pairs = load_pairs(PROJECT_ROOT / "configs")["pairs"]
    modes = exp_cfg.get("modes", ["single_a", "single_b", "double"])
    seeds = exp_cfg.get("seeds", [42, 137, 256, 512, 1024])

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

    for cmd in prep_cmds:
        _run_cmd(cmd, log_path)

    if parallel <= 1:
        for pair_id, mode, seed in tasks:
            _run_one(pair_id, mode, seed, log_path, resume)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = [executor.submit(_run_one, pair_id, mode, seed, log_path, resume) for pair_id, mode, seed in tasks]
            for future in as_completed(futures):
                future.result()

    if not with_post:
        _log_to_file(log_path, f"[{_ts()}] POST-SKIP (stage1 only)")
        return

    _log_to_file(log_path, f"[{_ts()}] POST-START")
    for cmd in _post_commands(latex_mode):
        _run_cmd(cmd, log_path)
    _log_to_file(log_path, f"[{_ts()}] POST-DONE")


if __name__ == "__main__":
    main()
