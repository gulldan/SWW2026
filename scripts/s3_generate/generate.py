"""Launch FREED++ molecule generation in single_a, single_b, or double mode.

Usage::

    python scripts/s3_generate/generate.py --pair CDK2_ERK2 --mode single_a
    python scripts/s3_generate/generate.py --pair CDK2_ERK2 --mode double
    python scripts/s3_generate/generate.py --pair CDK2_ERK2 --mode single_a --epochs 2 --n-molecules 10
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import click
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _run_cmd(cmd: list[str]) -> str:
    """Run a shell command and return its stripped stdout/stderr.

    Args:
        cmd: Command and arguments to execute.

    Returns:
        Stripped output string, or empty string on failure.
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    except FileNotFoundError:
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout or result.stderr or "").strip()


def _git_commit() -> str:
    """Return the short SHA of the current git HEAD."""
    out = _run_cmd(["git", "rev-parse", "--short", "HEAD"])
    return out.splitlines()[0] if out else "unknown"


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
    """Retrieve the Docker image digest (or ID) for provenance tracking.

    Args:
        image: Docker image name/tag (e.g. ``"dualdocking/freedpp:latest"``).

    Returns:
        Image digest hash, image ID, or value from master_config as fallback.
    """
    out = _run_cmd(["docker", "image", "inspect", image, "--format", "{{index .RepoDigests 0}}"])
    if "@" in out:
        return out.split("@")[-1]
    if out:
        return out
    # Local builds often have no RepoDigests; fall back to image ID.
    image_id = _run_cmd(["docker", "image", "inspect", image, "--format", "{{.Id}}"])
    if image_id:
        return image_id
    return _master_config_field("software.docker_image_digest")


def _boltz_version() -> str:
    """Return the installed Boltz package version, or ``"unknown"``."""
    try:
        import importlib.metadata as md

        return md.version("boltz")
    except Exception:
        return "unknown"


def _hardware_info() -> dict[str, str]:
    """Detect GPU name and CUDA version from system tools.

    Returns:
        Dict with ``"gpu"`` and ``"cuda_version"`` keys.
    """
    gpu = _run_cmd(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]) or "unknown"
    cuda = _run_cmd(["nvcc", "--version"])
    if cuda:
        # Extract release number if possible
        for line in cuda.splitlines():
            if "release" in line:
                cuda = line.strip().split("release")[-1].split(",")[0].strip()
                break
    return {"gpu": gpu, "cuda_version": cuda or "unknown"}


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
@click.option("--pair", required=True, help="ID пары из configs/pairs.yaml")
@click.option("--mode", type=click.Choice(["single_a", "single_b", "double"]), required=True)
@click.option("--n-molecules", default=None, type=int, help="Кол-во молекул (default из experiment.yaml)")
@click.option("--seed", default=42, type=int)
@click.option("--epochs", default=None, type=int, help="Эпохи FREED++ (default из experiment.yaml)")
@click.option("--steps-per-epoch", default=None, type=int, help="Шагов на эпоху (default 256)")
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
        logger.error("Пара '{}' не найдена", pair)
        sys.exit(1)

    pair_cfg = pairs[pair]
    output_dir = PROJECT_ROOT / "output" / "results" / pair / mode / f"seed_{seed}"

    # Загрузить параметры FREED++ из experiment.yaml
    exp_cfg = load_experiment(PROJECT_ROOT / experiment_config)
    generator_cfg = exp_cfg.get("generator", {})

    # Проверка калибровки нормализации для таргетов
    norm_path = PROJECT_ROOT / "configs" / "normalization.yaml"
    if norm_path.exists():
        import yaml

        norm_data = yaml.safe_load(norm_path.read_text())
        norm_targets = norm_data.get("targets", {})
        for role in ("target_a", "target_b"):
            tid = pair_cfg[role]["id"]
            entry = norm_targets.get(tid, {})
            if not entry:
                logger.error("Таргет '{}' отсутствует в normalization.yaml. Запустите compute_normalization.py", tid)
                sys.exit(1)
            if not entry.get("stable_mean") or not entry.get("stable_std"):
                logger.error(
                    "Таргет '{}' не калиброван (stable=false, n_samples={}). "
                    "Запустите: dock_calibration.py + compute_normalization.py",
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
        logger.warning("Низкая уникальность: {:.3f}", diversity.get("uniqueness", 0.0))

    freedpp_cfg = generator_cfg.get("freedpp", {}) if isinstance(generator_cfg, dict) else {}
    docker_image = (
        freedpp_cfg.get("docker_image", "dualdocking/freedpp:latest")
        if backend_name == "freedpp" and isinstance(freedpp_cfg, dict)
        else "unknown"
    )
    docker_digest = _docker_image_digest(str(docker_image)) if docker_image != "unknown" else "unknown"
    docking_program, docking_version = _normalization_meta()

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
        "training_time_seconds": int(duration),
        "hardware": _hardware_info(),
    }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info("Metadata -> {}", output_dir / "metadata.json")
    logger.info("Сгенерировано {} молекул в режиме {} для пары {}", len(smiles), mode, pair)


if __name__ == "__main__":
    main()
