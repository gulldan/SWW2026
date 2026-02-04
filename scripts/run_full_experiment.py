"""Run full experiment across pairs, modes, and seeds."""

from __future__ import annotations

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
        _log_to_file(log_path, f"[{_ts()}] ERROR: {result.stderr[-2000:]}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
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
def main(dry_run: bool, parallel: int, resume: bool) -> None:
    exp_cfg = load_experiment(PROJECT_ROOT / "configs")
    pairs = load_pairs(PROJECT_ROOT / "configs")["pairs"]
    modes = exp_cfg.get("modes", ["single_a", "single_b", "double"])
    seeds = exp_cfg.get("seeds", [42, 137, 256, 512, 1024])

    log_path = PROJECT_ROOT / "report" / "experiment.log"

    tasks = [(pair_id, mode, seed) for pair_id in pairs for mode in modes for seed in seeds]

    if dry_run:
        for pair_id, mode, seed in tasks:
            cmd = " ".join(_command_generate(pair_id, mode, seed)) + " && " + " ".join(_command_score(pair_id, mode, seed))
            click.echo(cmd)
        return

    if parallel <= 1:
        for pair_id, mode, seed in tasks:
            _run_one(pair_id, mode, seed, log_path, resume)
        return

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = [
            executor.submit(_run_one, pair_id, mode, seed, log_path, resume) for pair_id, mode, seed in tasks
        ]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
