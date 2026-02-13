"""Launch FREED++ molecule generation in single_a, single_b, or double mode.

Usage::

    python scripts/s3_generate/generate.py --pair CDK2_ERK2 --mode single_a
    python scripts/s3_generate/generate.py --pair CDK2_ERK2 --mode double
    python scripts/s3_generate/generate.py --pair CDK2_ERK2 --mode single_a --epochs 2 --n-molecules 10
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import click
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_git_dir(repo_root: Path) -> Path | None:
    """Resolve the real Git directory for a repository root.

    Supports both standard repos (``.git/`` directory) and worktrees
    (``.git`` file pointing to ``gitdir: ...``).

    Args:
        repo_root: Repository root path.

    Returns:
        Path to the Git metadata directory, or ``None`` if unavailable.
    """
    dot_git = repo_root / ".git"
    if dot_git.is_dir():
        return dot_git
    if not dot_git.is_file():
        return None

    try:
        text = dot_git.read_text().strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None

    gitdir_token = text.split(":", 1)[1].strip()
    gitdir = Path(gitdir_token)
    if not gitdir.is_absolute():
        gitdir = (repo_root / gitdir).resolve()
    return gitdir if gitdir.exists() else None


def _read_git_ref(git_dir: Path, ref: str) -> str:
    """Read a full commit hash from a git ref.

    Args:
        git_dir: Git metadata directory.
        ref: Ref path, e.g. ``"refs/heads/main"``.

    Returns:
        Commit hash string, or ``""`` if unresolved.
    """
    ref_path = git_dir / ref
    try:
        if ref_path.exists():
            return ref_path.read_text().strip()
    except OSError:
        return ""

    packed_refs = git_dir / "packed-refs"
    if not packed_refs.exists():
        return ""
    try:
        for line in packed_refs.read_text().splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            parts = line.strip().split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    except OSError:
        return ""
    return ""


def _short_commit(hash_text: str) -> str:
    """Normalize and shorten a commit hash for metadata output.

    Args:
        hash_text: Candidate full commit hash.

    Returns:
        Short hash (7 chars) or ``"unknown"`` if invalid.
    """
    token = hash_text.strip()
    if len(token) < 7:
        return "unknown"
    if not all(ch in "0123456789abcdefABCDEF" for ch in token):
        return "unknown"
    return token[:7].lower()


def _git_commit() -> str:
    """Return the short SHA of the current git HEAD without shelling out."""
    git_dir = _resolve_git_dir(PROJECT_ROOT)
    if git_dir is None:
        return "unknown"

    head_path = git_dir / "HEAD"
    try:
        head = head_path.read_text().strip()
    except OSError:
        return "unknown"

    if head.startswith("ref:"):
        ref = head.split(":", 1)[1].strip()
        return _short_commit(_read_git_ref(git_dir, ref))
    return _short_commit(head)


def _master_config_field(path: str) -> str:
    """Read a dot-delimited field from ``configs/master_config.yaml``.

    Args:
        path: Dot-separated key path (e.g. ``"software.docker_image_digest"``).

    Returns:
        String value at the path, or ``"unknown"`` if not found.
    """
    master_path = PROJECT_ROOT / "configs" / "master_config.yaml"
    if not master_path.exists():
        return "unknown"
    import yaml

    data = yaml.safe_load(master_path.read_text())
    value: str | None = data
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = None
            break
    if not value:
        return "unknown"
    return str(value)


def _docker_image_digest(image: str) -> str:
    """Retrieve FREED++ docker digest from static config metadata.

    Runtime Docker inspection is intentionally avoided here to keep
    generation metadata collection side-effect free.

    Args:
        image: Docker image name/tag (currently informational).

    Returns:
        Digest string from ``master_config.yaml`` or ``"unknown"``.
    """
    _ = image
    digest = _master_config_field("software.docker_image_digest")
    return digest or "unknown"


def _boltz_version() -> str:
    """Return the installed Boltz package version, or ``"unknown"``."""
    try:
        import importlib.metadata as md

        return md.version("boltz")
    except Exception:
        return "unknown"


def _hardware_info() -> dict[str, str]:
    """Detect GPU and CUDA version from torch runtime state.

    Returns:
        Dict with ``"gpu"`` and ``"cuda_version"`` keys.
    """
    try:
        import torch
    except Exception:
        return {"gpu": "unknown", "cuda_version": "unknown"}

    if not torch.cuda.is_available():
        return {"gpu": "unknown", "cuda_version": torch.version.cuda or "unknown"}

    gpu_count = torch.cuda.device_count()
    if gpu_count <= 0:
        return {"gpu": "unknown", "cuda_version": torch.version.cuda or "unknown"}

    device_token = os.environ.get("FREEDPP_CUDA_DEVICE") or os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    try:
        requested_idx = int(device_token.split(",", 1)[0])
    except ValueError:
        requested_idx = 0
    if requested_idx < 0 or requested_idx >= gpu_count:
        requested_idx = 0

    try:
        gpu_name = torch.cuda.get_device_name(requested_idx)
    except Exception:
        gpu_name = "unknown"
    return {"gpu": gpu_name, "cuda_version": torch.version.cuda or "unknown"}


def _normalization_meta() -> tuple[str, str]:
    """Read docking program name and version from ``normalization.yaml``.

    Returns:
        Tuple of ``(docking_program, docking_version)``.
    """
    norm_path = PROJECT_ROOT / "configs" / "normalization.yaml"
    if not norm_path.exists():
        return "unidock_gpu", "unknown"
    import yaml

    data = yaml.safe_load(norm_path.read_text())
    docking_program = data.get("docking_program", "unidock_gpu")
    docking_version = data.get("docking_version", "unknown")
    return docking_program, docking_version


def _master_config_version() -> str:
    """Return the ``version`` field from master_config.yaml."""
    return _master_config_field("version")


@click.command()
@click.option("--pair", required=True, help="Pair ID from configs/pairs.yaml")
@click.option(
    "--mode",
    type=click.Choice(["single_a", "single_b", "double", "linear_equal", "biased_a", "selective_a"]),
    required=True,
)
@click.option("--n-molecules", default=None, type=int, help="Number of molecules (default from experiment.yaml)")
@click.option("--seed", default=42, type=int)
@click.option("--epochs", default=None, type=int, help="FREED++ epochs (default from experiment.yaml)")
@click.option("--steps-per-epoch", default=None, type=int, help="Steps per epoch (default 256)")
@click.option("--config", default="configs/pairs.yaml")
@click.option("--experiment-config", default="configs/experiment.yaml")
def main(
    pair: str,
    mode: str,
    n_molecules: int | None,
    seed: int,
    epochs: int | None,
    steps_per_epoch: int | None,
    config: str,
    experiment_config: str,
) -> None:
    """Generate molecules for a given target pair and mode.

    Loads pair configuration, validates calibration, runs the FREED++
    (or configured alternative) generator, and writes ``generated.csv`` plus ``metadata.json`` with
    provenance information (git commit, docker digest, hardware, diversity).

    Args:
        pair: Pair identifier from ``configs/pairs.yaml``.
        mode: Generation mode — ``single_a``, ``single_b``, or ``double``.
        n_molecules: Override molecule count (default from experiment.yaml).
        seed: Random seed for reproducibility.
        epochs: Override FREED++ training epochs.
        steps_per_epoch: Override steps per training epoch.
        config: Path to pairs.yaml config file.
        experiment_config: Path to experiment.yaml config file.
    """
    from dualdocking.config import load_experiment, load_pairs
    from dualdocking.generation.backends import GenerationRequest, resolve_generation_backend

    pairs = load_pairs(PROJECT_ROOT / config)["pairs"]
    if pair not in pairs:
        logger.error("Pair '{}' not found", pair)
        sys.exit(1)

    pair_cfg = pairs[pair]
    output_dir = PROJECT_ROOT / "output" / "results" / pair / mode / f"seed_{seed}"

    # Load FREED++ parameters from experiment.yaml
    exp_cfg = load_experiment(PROJECT_ROOT / experiment_config)
    generator_cfg = exp_cfg.get("generator", {})

    # Validate calibration normalization for targets
    norm_path = PROJECT_ROOT / "configs" / "normalization.yaml"
    if norm_path.exists():
        import yaml

        norm_data = yaml.safe_load(norm_path.read_text())
        norm_targets = norm_data.get("targets", {})
        for role in ("target_a", "target_b"):
            tid = pair_cfg[role]["id"]
            entry = norm_targets.get(tid, {})
            if not entry:
                logger.error("Target '{}' missing from normalization.yaml. Run compute_normalization.py", tid)
                sys.exit(1)
            if not entry.get("stable_mean") or not entry.get("stable_std"):
                logger.error(
                    "Target '{}' not calibrated (stable=false, n_samples={}). "
                    "Run: dock_calibration.py + compute_normalization.py",
                    tid,
                    entry.get("n_samples", 0),
                )
                sys.exit(1)

    qg_cfg = exp_cfg.get("quality_gate", {})

    backend_name, backend_generate = resolve_generation_backend(generator_cfg)
    logger.info("Generator backend: {}", backend_name)

    request = GenerationRequest(
        pair_id=pair,
        pair_config=pair_cfg,
        mode=mode,
        seed=seed,
        output_dir=output_dir,
        n_molecules=n_molecules or exp_cfg.get("budget", {}).get("n_molecules", 1000),
        epochs_override=epochs,
        steps_per_epoch_override=steps_per_epoch,
        generator_config=generator_cfg,
        quality_gate=qg_cfg,
    )

    start = time.time()
    smiles = backend_generate(request)
    duration = time.time() - start

    from dualdocking.analysis.diversity import compute_all_diversity_metrics
    from dualdocking.data.chem_sanitize import canonicalize

    n_raw = len(smiles)
    canon = [canonicalize(s) for s in smiles]
    n_unique = len({c for c in canon if c})
    duplication_rate = 0.0 if n_raw == 0 else 1.0 - (n_unique / n_raw)

    diversity = compute_all_diversity_metrics(smiles).to_dict()
    if diversity.get("uniqueness", 0.0) < 0.90:
        logger.warning("Low uniqueness: {:.3f}", diversity.get("uniqueness", 0.0))

    # Post-generation drug-likeness filters
    filter_stats_dict: dict = {}
    try:
        from dualdocking.data.filters import compute_filters, load_filter_config

        filters_path = PROJECT_ROOT / "configs" / "filters.yaml"
        if filters_path.exists():
            filter_cfg = load_filter_config(filters_path)
            _filter_results, filter_stats = compute_filters(smiles, filter_cfg)
            filter_stats_dict = filter_stats.to_dict()
        else:
            logger.warning("filters.yaml not found, skipping drug-likeness filters")
    except Exception as exc:
        logger.warning("Filter computation failed ({}), skipping", exc)

    freedpp_cfg = generator_cfg.get("freedpp", {}) if isinstance(generator_cfg, dict) else {}
    docker_image = (
        freedpp_cfg.get("docker_image", "dualdocking/freedpp:latest")
        if backend_name == "freedpp" and isinstance(freedpp_cfg, dict)
        else "unknown"
    )
    docker_digest = _docker_image_digest(str(docker_image)) if docker_image != "unknown" else "unknown"
    docking_program, docking_version = _normalization_meta()

    # Extract training curve from FREED++ log (if available)
    training_curve: list[dict] = []
    if backend_name == "freedpp":
        try:
            from dualdocking.generation.freedpp import parse_training_log

            ws_dir = output_dir / "freedpp_workspace"
            log_files = sorted(ws_dir.glob("freedpp_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            if log_files:
                training_curve = parse_training_log(log_files[0])
        except Exception as exc:
            logger.warning("Training log parsing failed ({}), skipping", exc)

    metadata = {
        "pair_id": pair,
        "mode": mode,
        "seed": seed,
        "generator_backend": backend_name,
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "git_commit": _git_commit(),
        "master_config_version": _master_config_version(),
        "docker_image_digest": docker_digest,
        "boltz_version": _boltz_version(),
        "docking_program": docking_program,
        "docking_version": docking_version,
        "n_molecules_raw": n_raw,
        "n_molecules_unique": n_unique,
        "duplication_rate": round(duplication_rate, 4),
        "diversity": diversity,
        "filter_stats": filter_stats_dict,
        "training_curve": training_curve,
        "training_time_seconds": int(duration),
        "hardware": _hardware_info(),
    }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Metadata -> {}", output_dir / "metadata.json")
    logger.info("Generated {} molecules in mode {} for pair {}", len(smiles), mode, pair)


if __name__ == "__main__":
    main()
