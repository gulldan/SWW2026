"""Smoke-тест Boltz-2.

Запускает минимальный пример скоринга для проверки работоспособности Boltz-2.
Результат: results/smoke/boltz_score.json

Использование:
    uv run python scripts/boltz_smoke.py --target CDK2
    uv run python scripts/boltz_smoke.py --target CDK2 --dry-run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Сокращённый набор: 3 молекулы для быстрой проверки
SMOKE_SMILES = [
    "c1ccccc1",  # бензол
    "CC(=O)Oc1ccccc1C(=O)O",  # аспирин
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ибупрофен
]

# Допустимый диапазон скоров (log10(IC50, μM))
# Пример: 1 nM = 0.001 μM → log10 = -3
SCORE_MIN = -6.0
SCORE_MAX = 2.0


@click.command()
@click.option("--target", default="CDK2", help="ID мишени для скоринга")
@click.option("--dry-run", is_flag=True, default=False, help="Только сгенерировать YAML без запуска Boltz")
def main(target: str, dry_run: bool) -> None:
    """Запустить smoke-тест Boltz-2."""
    if dry_run:
        _run_dry(target)
        return

    from dualdocking.scoring.boltz2 import score_batch

    logger.info("Запуск smoke-теста Boltz-2: {} молекул по мишени {}", len(SMOKE_SMILES), target)

    scores = score_batch(target, SMOKE_SMILES, use_cache=False)

    results = {
        "target": target,
        "n_molecules": len(SMOKE_SMILES),
        "molecules": [{"smiles": smi, "score": float(s)} for smi, s in zip(SMOKE_SMILES, scores, strict=True)],
        "mean_score": float(scores[~np.isnan(scores)].mean()) if not np.all(np.isnan(scores)) else None,
        "status": "ok",
    }

    out_dir = PROJECT_ROOT / "results" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "boltz_score.json"
    out_path.write_text(json.dumps(results, indent=2))

    logger.info("Smoke-тест завершён. Результаты -> {}", out_path)
    n_valid = 0
    for mol in results["molecules"]:
        score = mol["score"]
        logger.info("  {} -> {:.3f}", mol["smiles"], score)
        if SCORE_MIN <= score <= SCORE_MAX:
            n_valid += 1
        else:
            logger.warning("  Скор {:.3f} вне ожидаемого диапазона [{}, {}]", score, SCORE_MIN, SCORE_MAX)

    logger.info("Валидных скоров: {}/{}", n_valid, len(SMOKE_SMILES))


def _run_dry(target: str) -> None:
    """Dry-run: сгенерировать YAML без запуска Boltz для проверки формата."""
    from dualdocking.scoring.boltz2 import _get_msa_path, _load_sequence, _write_boltz_yaml

    logger.info("Dry-run: генерация YAML для мишени {}", target)

    sequence = _load_sequence(target)
    logger.info("Sequence загружена: {} символов", len(sequence))

    msa_path = _get_msa_path(target)
    if msa_path:
        logger.info("MSA найден: {}", msa_path)
    else:
        logger.warning("MSA не найден для {}. Boltz потребует --use_msa_server (онлайн).", target)

    out_dir = PROJECT_ROOT / "results" / "smoke" / "dry_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, smi in enumerate(SMOKE_SMILES):
        yaml_path = out_dir / f"input_{i}.yaml"
        _write_boltz_yaml(sequence, smi, yaml_path, msa_path=msa_path)
        logger.info("YAML сгенерирован: {}", yaml_path)

    logger.info("Dry-run завершён. YAML файлы в {}", out_dir)


if __name__ == "__main__":
    main()
