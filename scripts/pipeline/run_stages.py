"""Selective stage runner for the DualDocking pipeline.

Allows running individual pipeline stages or subsets, with preflight
checks for required inputs.

Usage::

    uv run python scripts/pipeline/run_stages.py --stages analyze,report
    uv run python scripts/pipeline/run_stages.py --stages all
    uv run python scripts/pipeline/run_stages.py --stages score --pair CDK2_ERK2 --seeds 42
    uv run python scripts/pipeline/run_stages.py --list-stages
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from loguru import logger

from dualdocking.command_runner import run_project_command

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

STAGE_ORDER = ["fetch", "calibrate", "generate", "score", "analyze", "report", "audit"]

STAGE_DEPS = {
    "fetch": [],
    "calibrate": ["fetch"],
    "generate": ["calibrate"],
    "score": ["generate"],
    "analyze": ["score"],
    "report": ["analyze"],
    "audit": ["analyze"],
}

STAGE_DESCRIPTIONS = {
    "fetch": "Download PDB, FASTA, SMILES, Docker image",
    "calibrate": "Dock calibration set, compute normalization",
    "generate": "Run FREED++ molecule generation",
    "score": "Score molecules with Boltz-2",
    "analyze": "Aggregate results, compute metrics, sensitivity, correlation",
    "report": "Generate plots, render molecules, LaTeX report",
    "audit": "Run claim traceability audit",
}


def _check_fetch_outputs() -> list[str]:
    """Check that fetch stage outputs exist.

    Returns:
        List of filesystem paths for any missing fetch outputs.
    """
    missing = []
    targets_dir = PROJECT_ROOT / "output" / "data" / "targets"
    if not targets_dir.exists():
        missing.append(str(targets_dir))
    cal_dir = PROJECT_ROOT / "output" / "data" / "calibration"
    if not cal_dir.exists():
        missing.append(str(cal_dir))
    return missing


def _check_calibration_outputs() -> list[str]:
    """Check that calibration outputs exist.

    Returns:
        List of filesystem paths for any missing calibration outputs.
    """
    missing = []
    norm = PROJECT_ROOT / "configs" / "normalization.yaml"
    if not norm.exists():
        missing.append(str(norm))
    scores_dir = PROJECT_ROOT / "output" / "data" / "calibration" / "scores"
    if not scores_dir.exists():
        missing.append(str(scores_dir))
    return missing


def _check_generation_outputs(pair: str, mode: str, seed: int) -> list[str]:
    """Check that generation outputs exist for a specific run.

    Args:
        pair: Protein pair identifier (e.g. ``"CDK2_ERK2"``).
        mode: Docking mode (e.g. ``"single_a"``, ``"double"``).
        seed: Random seed for the generation run.

    Returns:
        List of filesystem paths for any missing generated output files.
    """
    from dualdocking.data.io_utils import stem_exists

    missing = []
    run_dir = PROJECT_ROOT / "output" / "results" / pair / mode / f"seed_{seed}"
    if not stem_exists(run_dir / "generated"):
        missing.append(f"{run_dir}/generated.parquet")
    return missing


def _check_score_outputs(pair: str, mode: str, seed: int) -> list[str]:
    """Check that scoring outputs exist for a specific run.

    Args:
        pair: Protein pair identifier (e.g. ``"CDK2_ERK2"``).
        mode: Docking mode (e.g. ``"single_a"``, ``"double"``).
        seed: Random seed for the scoring run.

    Returns:
        List of filesystem paths for any missing scored output files.
    """
    from dualdocking.data.io_utils import stem_exists

    missing = []
    run_dir = PROJECT_ROOT / "output" / "results" / pair / mode / f"seed_{seed}"
    if not stem_exists(run_dir / "scored_joined"):
        missing.append(f"{run_dir}/scored_joined.parquet")
    return missing


def _check_analyze_outputs() -> list[str]:
    """Check that analysis outputs exist.

    Returns:
        List of filesystem paths for any missing analysis output files.
    """
    missing = []
    for name in ("all_results.parquet",):
        p = PROJECT_ROOT / "report" / "data" / name
        if not p.exists():
            missing.append(str(p))
    for name in ("metrics.json", "stats.json"):
        p = PROJECT_ROOT / "report" / name
        if not p.exists():
            missing.append(str(p))
    return missing


def _check_report_outputs() -> list[str]:
    """Check that report outputs exist.

    Returns:
        List of filesystem paths for any missing report output files.
    """
    missing = []
    figs = PROJECT_ROOT / "report" / "figures"
    if not figs.exists():
        missing.append(str(figs))
    return missing


def _run_cmd(cmd: list[str]) -> None:
    """Run a subprocess command, raising on failure.

    Args:
        cmd: Command and arguments to execute as a subprocess.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero return code.
    """
    logger.info("Running: {}", " ".join(cmd))
    result = run_project_command(
        cmd,
        project_root=PROJECT_ROOT,
        allow_inprocess=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (rc={result.returncode}): {' '.join(cmd)}")


def _load_config() -> tuple[dict, dict]:
    """Load pairs and experiment configs.

    Returns:
        A tuple of ``(pairs, exp_cfg)`` where *pairs* is the pair
        definitions dict and *exp_cfg* is the full experiment config dict.
    """
    from dualdocking.config import load_experiment, load_pairs

    pairs = load_pairs(PROJECT_ROOT / "configs")["pairs"]
    exp_cfg = load_experiment(PROJECT_ROOT / "configs")
    return pairs, exp_cfg


def _run_fetch() -> None:
    """Run the fetch stage (download PDB, FASTA, SMILES, Docker image)."""
    _run_cmd(["uv", "run", "python", "scripts/s0_fetch/fetch_all.py"])


def _run_calibrate() -> None:
    """Run the calibration stage (dock calibration set, compute normalization)."""
    _run_cmd(["uv", "run", "python", "scripts/s2_calibrate/dock_calibration.py"])
    _run_cmd(["uv", "run", "python", "scripts/s2_calibrate/compute_normalization.py"])


def _run_generate(pair: str, mode: str, seed: int) -> None:
    """Run FREED++ molecule generation for one pair/mode/seed combination.

    Args:
        pair: Protein pair identifier (e.g. ``"CDK2_ERK2"``).
        mode: Docking mode (e.g. ``"single_a"``, ``"double"``).
        seed: Random seed for reproducibility.
    """
    _run_cmd(
        [
            "uv",
            "run",
            "python",
            "scripts/s3_generate/generate.py",
            "--pair",
            pair,
            "--mode",
            mode,
            "--seed",
            str(seed),
        ]
    )


def _run_score(pair: str, mode: str, seed: int, molecules: str | None = None) -> None:
    """Score molecules with Boltz-2 for one pair/mode/seed combination.

    Args:
        pair: Protein pair identifier (e.g. ``"CDK2_ERK2"``).
        mode: Docking mode (e.g. ``"single_a"``, ``"double"``).
        seed: Random seed for the scoring run.
        molecules: Optional path to an external molecule file to score.
    """
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/s4_score/score.py",
        "--pair",
        pair,
        "--mode",
        mode,
        "--seed",
        str(seed),
    ]
    if molecules:
        cmd.extend(["--molecules", molecules])
    _run_cmd(cmd)


def _run_analyze() -> None:
    """Run the full analysis stage (aggregate, metrics, sensitivity, correlation, summarize)."""
    _run_cmd(["uv", "run", "python", "scripts/s5_analyze/aggregate.py"])
    _run_cmd(["uv", "run", "python", "scripts/s5_analyze/compute_metrics.py"])
    _run_cmd(["uv", "run", "python", "scripts/s5_analyze/threshold_sensitivity.py"])
    _run_cmd(["uv", "run", "python", "scripts/s5_analyze/proxy_correlation.py"])
    _run_cmd(["uv", "run", "python", "scripts/s5_analyze/summarize.py"])


def _run_report(latex_mode: str = "off") -> None:
    """Run the report stage (plots, molecule renders, optional LaTeX report).

    Args:
        latex_mode: LaTeX compilation mode. ``"off"`` skips LaTeX,
            ``"draft"`` or ``"full"`` triggers report generation.
    """
    _run_cmd(["uv", "run", "python", "scripts/s6_report/plot_results.py"])
    _run_cmd(["uv", "run", "python", "scripts/s6_report/render_molecules.py"])
    if latex_mode != "off":
        _run_cmd(
            [
                "uv",
                "run",
                "python",
                "scripts/s6_report/generate_latex_report.py",
                "--mode",
                latex_mode,
            ]
        )


def _run_audit() -> None:
    """Run the claim traceability audit."""
    _run_cmd(["uv", "run", "python", "scripts/s7_audit/claim_audit.py"])


def _preflight_check(stage: str, pairs: dict, exp_cfg: dict, pair_filter: str | None, seeds: list[int]) -> list[str]:
    """Check prerequisites for a stage.

    Args:
        stage: Pipeline stage name (e.g. ``"fetch"``, ``"score"``).
        pairs: Mapping of pair identifiers to their configuration dicts.
        exp_cfg: Full experiment configuration dictionary.
        pair_filter: If set, restrict checks to this single pair identifier.
        seeds: List of random seeds to validate against.

    Returns:
        List of filesystem paths for any missing prerequisite inputs.
    """
    checks = {
        "calibrate": _check_fetch_outputs,
        "analyze": lambda: _check_analyze_prereqs(pairs, exp_cfg, pair_filter, seeds),
        "report": _check_analyze_outputs,
        "audit": _check_analyze_outputs,
    }

    if stage == "generate":
        return _check_calibration_outputs()
    if stage == "score":
        return _check_score_prereqs(pairs, exp_cfg, pair_filter, seeds)

    check_fn = checks.get(stage)
    if check_fn:
        return check_fn()
    return []


def _check_score_prereqs(pairs: dict, exp_cfg: dict, pair_filter: str | None, seeds: list[int]) -> list[str]:
    """Check that generated files exist for scoring.

    Args:
        pairs: Mapping of pair identifiers to their configuration dicts.
        exp_cfg: Full experiment configuration dictionary.
        pair_filter: If set, restrict checks to this single pair identifier.
        seeds: List of random seeds to validate against.

    Returns:
        List of filesystem paths for any missing generation output files
        required by the scoring stage.
    """
    missing = []
    modes = exp_cfg.get("modes", ["single_a", "single_b", "double"])
    for pair_id in pairs:
        if pair_filter and pair_id != pair_filter:
            continue
        for mode in modes:
            for seed in seeds:
                missing.extend(_check_generation_outputs(pair_id, mode, seed))
    return missing


def _check_analyze_prereqs(pairs: dict, exp_cfg: dict, pair_filter: str | None, seeds: list[int]) -> list[str]:
    """Check that scored files exist for analysis.

    Args:
        pairs: Mapping of pair identifiers to their configuration dicts.
            Pairs with ``set == "blocked"`` are skipped.
        exp_cfg: Full experiment configuration dictionary.
        pair_filter: If set, restrict checks to this single pair identifier.
        seeds: List of random seeds to validate against.

    Returns:
        List of filesystem paths for any missing scored output files
        required by the analysis stage.
    """
    missing = []
    modes = exp_cfg.get("modes", ["single_a", "single_b", "double"])
    for pair_id, pair_cfg in pairs.items():
        if pair_filter and pair_id != pair_filter:
            continue
        if pair_cfg.get("set") == "blocked":
            continue
        for mode in modes:
            for seed in seeds:
                missing.extend(_check_score_outputs(pair_id, mode, seed))
    return missing


@click.command()
@click.option("--stages", default=None, help="Comma-separated stages to run (or 'all')")
@click.option("--list-stages", is_flag=True, help="List available stages and exit")
@click.option("--pair", default=None, help="Filter to a specific pair")
@click.option("--seeds", default=None, help="Comma-separated seed list")
@click.option("--molecules", default=None, type=click.Path(exists=True), help="Molecule file for scoring")
@click.option("--latex-mode", type=click.Choice(["off", "draft", "full"]), default="off")
@click.option("--dry-run", is_flag=True, help="Show what would be run without executing")
@click.option("--skip-preflight/--no-skip-preflight", default=False, help="Skip input checks")
def main(
    stages: str | None,
    list_stages: bool,
    pair: str | None,
    seeds: str | None,
    molecules: str | None,
    latex_mode: str,
    dry_run: bool,
    skip_preflight: bool,
) -> None:
    """Run selected pipeline stages with preflight input validation.

    Each stage checks that its required inputs are available before
    executing, giving clear error messages about what to run first.

    Args:
        stages: Comma-separated stage names to run, or ``"all"`` for every
            stage. ``None`` triggers an error with usage guidance.
        list_stages: If ``True``, print available stages and exit.
        pair: Optional pair identifier to restrict execution to a single pair.
        seeds: Comma-separated seed values. Defaults to the seeds defined in
            the experiment config.
        molecules: Optional filesystem path to a molecule file used by the
            scoring stage.
        latex_mode: LaTeX compilation mode (``"off"``, ``"draft"``, or
            ``"full"``).
        dry_run: If ``True``, log what would be run without executing.
        skip_preflight: If ``True``, skip prerequisite input checks before
            each stage.
    """
    if list_stages:
        click.echo("Available stages:\n")
        for name in STAGE_ORDER:
            deps = STAGE_DEPS[name]
            dep_str = f" (requires: {', '.join(deps)})" if deps else ""
            click.echo(f"  {name:<12} {STAGE_DESCRIPTIONS[name]}{dep_str}")
        return

    if stages is None:
        click.echo("Error: --stages is required (use --list-stages to see options)")
        sys.exit(1)

    if stages == "all":
        selected = list(STAGE_ORDER)
    else:
        selected = [s.strip() for s in stages.split(",")]
        for s in selected:
            if s not in STAGE_ORDER:
                click.echo(f"Error: unknown stage '{s}'. Use --list-stages to see options.")
                sys.exit(1)

    pairs, exp_cfg = _load_config()
    all_seeds = exp_cfg.get("seeds", [42, 137, 256, 512, 1024])
    seed_list = [int(s) for s in seeds.split(",")] if seeds else all_seeds
    modes = exp_cfg.get("modes", ["single_a", "single_b", "double"])

    # Exclude blocked pairs from run
    active_pairs = {pid: cfg for pid, cfg in pairs.items() if cfg.get("set") != "blocked"}

    for stage in selected:
        logger.info("=== Stage: {} ===", stage)

        if not skip_preflight:
            missing = _preflight_check(stage, active_pairs, exp_cfg, pair, seed_list)
            if missing:
                logger.error("Preflight check failed for stage '{}'. Missing inputs:", stage)
                for m in missing[:10]:
                    logger.error("  - {}", m)
                if len(missing) > 10:
                    logger.error("  ... and {} more", len(missing) - 10)
                sys.exit(1)

        if dry_run:
            logger.info("[DRY RUN] Would run stage: {}", stage)
            continue

        if stage == "fetch":
            _run_fetch()
        elif stage == "calibrate":
            _run_calibrate()
        elif stage == "generate":
            for pair_id in active_pairs:
                if pair and pair_id != pair:
                    continue
                for mode in modes:
                    for seed in seed_list:
                        _run_generate(pair_id, mode, seed)
        elif stage == "score":
            for pair_id in active_pairs:
                if pair and pair_id != pair:
                    continue
                for mode in modes:
                    for seed in seed_list:
                        _run_score(pair_id, mode, seed, molecules=molecules)
        elif stage == "analyze":
            _run_analyze()
        elif stage == "report":
            _run_report(latex_mode)
        elif stage == "audit":
            _run_audit()

        logger.info("=== Stage {} complete ===", stage)

    logger.info("All selected stages completed successfully")


if __name__ == "__main__":
    main()
