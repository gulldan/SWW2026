"""Запуск генерации FREED++ в single или double режиме.

Использование:
    python scripts/generate.py --pair CDK2_ERK2 --mode single
    python scripts/generate.py --pair CDK2_ERK2 --mode double
    python scripts/generate.py --pair CDK2_ERK2 --mode single --epochs 2 --n-molecules 10
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@click.command()
@click.option("--pair", required=True, help="ID пары из configs/pairs.yaml")
@click.option("--mode", type=click.Choice(["single", "double"]), required=True)
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

    pairs = load_pairs(PROJECT_ROOT / "configs")["pairs"]
    if pair not in pairs:
        logger.error("Пара '{}' не найдена", pair)
        sys.exit(1)

    pair_cfg = pairs[pair]
    output_dir = PROJECT_ROOT / "results" / pair / mode

    # Загрузить параметры FREED++ из experiment.yaml
    exp_cfg = load_experiment(PROJECT_ROOT / "configs")
    freedpp_cfg = exp_cfg.get("generator", {}).get("freedpp", {})

    gen_config = GenerationConfig(
        target_id=pair_cfg["target_a"]["id"],
        n_molecules=n_molecules or exp_cfg.get("budget", {}).get("n_molecules", 1000),
        seed=seed,
        output_dir=output_dir,
        reward_mode=mode,
        second_target_id=pair_cfg["target_b"]["id"] if mode == "double" else None,
        epochs=epochs or freedpp_cfg.get("epochs", 200),
        num_sub_proc=freedpp_cfg.get("num_sub_proc", 12),
        starting_smile=freedpp_cfg.get("starting_smile", "c1([*:1])c([*:2])ccc([*:3])c1"),
        use_cache=freedpp_cfg.get("use_model_cache", True),
        docker_image=freedpp_cfg.get("docker_image", "dualdocking/freedpp:latest"),
        training_timeout=freedpp_cfg.get("training_timeout", 86400),
        box_padding=freedpp_cfg.get("box_padding", 10.0),
        vina_program=freedpp_cfg.get("vina_program", "qvina02"),
        steps_per_epoch=steps_per_epoch or freedpp_cfg.get("steps_per_epoch", 256),
    )

    smiles = generate(gen_config)
    logger.info("Сгенерировано {} молекул в режиме {} для пары {}", len(smiles), mode, pair)


if __name__ == "__main__":
    main()
