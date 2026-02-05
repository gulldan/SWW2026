"""Запуск генерации FREED++ в single_a, single_b или double режиме.

Использование:
    python scripts/generate.py --pair CDK2_ERK2 --mode single_a
    python scripts/generate.py --pair CDK2_ERK2 --mode single_b
    python scripts/generate.py --pair CDK2_ERK2 --mode double
    python scripts/generate.py --pair CDK2_ERK2 --mode single_a --epochs 2 --n-molecules 10
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_cmd(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    except FileNotFoundError:
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout or result.stderr or "").strip()


def _git_commit() -> str:
    out = _run_cmd(["git", "rev-parse", "--short", "HEAD"])
    return out.splitlines()[0] if out else "unknown"


def _master_config_field(path: str) -> str:
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
    try:
        import importlib.metadata as md

        return md.version("boltz")
    except Exception:
        return "unknown"


def _hardware_info() -> dict[str, str]:
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
    norm_path = PROJECT_ROOT / "configs" / "normalization.yaml"
    if not norm_path.exists():
        return "unidock_gpu", "unknown"
    import yaml

    data = yaml.safe_load(norm_path.read_text())
    docking_program = data.get("docking_program", "unidock_gpu")
    docking_version = data.get("docking_version", "unknown")
    return docking_program, docking_version


def _master_config_version() -> str:
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
    """Сгенерировать молекулы с помощью FREED++."""
    from dualdocking.config import load_experiment, load_pairs
    from dualdocking.generator_freedpp import GenerationConfig, generate

    pairs = load_pairs(PROJECT_ROOT / config)["pairs"]
    if pair not in pairs:
        logger.error("Пара '{}' не найдена", pair)
        sys.exit(1)

    pair_cfg = pairs[pair]
    output_dir = PROJECT_ROOT / "results" / pair / mode / f"seed_{seed}"

    # Загрузить параметры FREED++ из experiment.yaml
    exp_cfg = load_experiment(PROJECT_ROOT / experiment_config)
    freedpp_cfg = exp_cfg.get("generator", {}).get("freedpp", {})

    second_target_id = None
    if mode in {"single_b", "double"}:
        second_target_id = pair_cfg["target_b"]["id"]

    gen_config = GenerationConfig(
        target_id=pair_cfg["target_a"]["id"],
        n_molecules=n_molecules or exp_cfg.get("budget", {}).get("n_molecules", 1000),
        seed=seed,
        output_dir=output_dir,
        reward_mode=mode,
        second_target_id=second_target_id,
        epochs=epochs or freedpp_cfg.get("epochs", 200),
        num_sub_proc=freedpp_cfg.get("num_sub_proc", 12),
        starting_smile=freedpp_cfg.get("starting_smile", "c1([*:1])c([*:2])ccc([*:3])c1"),
        use_cache=freedpp_cfg.get("use_model_cache", True),
        docker_image=freedpp_cfg.get("docker_image", "dualdocking/freedpp:latest"),
        training_timeout=freedpp_cfg.get("training_timeout", 86400),
        box_padding=freedpp_cfg.get("box_padding", 10.0),
        vina_program=freedpp_cfg.get("vina_program", "unidock_gpu"),
        steps_per_epoch=steps_per_epoch or freedpp_cfg.get("steps_per_epoch", 256),
    )

    start = time.time()
    smiles = generate(gen_config)
    duration = time.time() - start

    from dualdocking.chem_sanitize import canonicalize
    from dualdocking.diversity import compute_all_diversity_metrics

    n_raw = len(smiles)
    canon = [canonicalize(s) for s in smiles]
    n_unique = len({c for c in canon if c})
    duplication_rate = 0.0 if n_raw == 0 else 1.0 - (n_unique / n_raw)

    diversity = compute_all_diversity_metrics(smiles).to_dict()
    if diversity.get("uniqueness", 0.0) < 0.90:
        logger.warning("Низкая уникальность: {:.3f}", diversity.get("uniqueness", 0.0))

    docker_digest = _docker_image_digest(gen_config.docker_image)
    docking_program, docking_version = _normalization_meta()

    metadata = {
        "pair_id": pair,
        "mode": mode,
        "seed": seed,
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
