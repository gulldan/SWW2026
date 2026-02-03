"""Обёртка скоринга Boltz-2.

Предоставляет ``score_batch(target, smiles_list)`` с дисковым кешем.
Вызывает ``boltz predict`` через subprocess (Boltz-2 не имеет публичного Python API).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
DATA_DIR = PROJECT_ROOT / "data" / "targets"
MSA_CACHE_DIR = CACHE_DIR / "boltz_msa"

BOLTZ_TIMEOUT = 600  # секунд на одну молекулу
MAX_LIGAND_ATOMS = 128

# Boltz CLI параметры
BOLTZ_CLI_ARGS: list[str] = [
    "--model",
    "boltz2",
    "--accelerator",
    "gpu",
    "--recycling_steps",
    "3",
    "--sampling_steps",
    "200",
    "--diffusion_samples",
    "1",
    "--sampling_steps_affinity",
    "200",
    "--diffusion_samples_affinity",
    "5",
    "--override",
]

# Кеш sequence в памяти (target_id -> sequence)
_sequence_cache: dict[str, str] = {}


def _cache_key(target_id: str, smi: str) -> str:
    blob = f"{target_id}||{smi}".encode()
    return hashlib.sha256(blob).hexdigest()


def _load_cache(target_id: str) -> dict[str, float]:
    path = CACHE_DIR / f"{target_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_cache(target_id: str, cache: dict[str, float]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{target_id}.json"
    path.write_text(json.dumps(cache, indent=2))


def _load_sequence(target_id: str) -> str:
    """Читает FASTA из data/targets/{target_id}.fasta, кеширует в памяти."""
    if target_id in _sequence_cache:
        return _sequence_cache[target_id]

    fasta_path = DATA_DIR / f"{target_id}.fasta"
    if not fasta_path.exists():
        msg = f"FASTA не найден: {fasta_path}. Запустите: uv run python scripts/prepare_targets.py --prepare-all"
        raise FileNotFoundError(msg)

    lines = fasta_path.read_text().strip().splitlines()
    # Пропустить заголовки FASTA (строки начинающиеся с >)
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    _sequence_cache[target_id] = seq
    return seq


def _get_msa_path(target_id: str) -> Path | None:
    """Ищет pre-computed MSA в cache/boltz_msa/{target_id}.csv."""
    msa_path = MSA_CACHE_DIR / f"{target_id}.csv"
    if msa_path.exists():
        return msa_path
    return None


def _validate_smiles(smiles: str) -> bool:
    """Проверяет SMILES через RDKit: парсинг + <= MAX_LIGAND_ATOMS атомов."""
    try:
        from rdkit import Chem
    except ImportError:
        logger.warning("RDKit недоступен, пропускаем валидацию SMILES")
        return True

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return mol.GetNumAtoms() <= MAX_LIGAND_ATOMS


def _write_boltz_yaml(
    sequence: str,
    smiles: str,
    path: Path,
    msa_path: Path | None = None,
) -> None:
    """Генерирует входной YAML для Boltz predict."""
    import yaml

    protein_entry: dict = {
        "id": "A",
        "sequence": sequence,
    }
    if msa_path is not None:
        protein_entry["msa"] = str(msa_path.resolve())

    data = {
        "version": 1,
        "sequences": [
            {"protein": protein_entry},
            {"ligand": {"id": "B", "smiles": smiles}},
        ],
        "properties": [
            {"affinity": {"binder": "B"}},
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


def _run_boltz_predict(yaml_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
    """Запуск boltz predict через subprocess."""
    cmd = [
        "uv",
        "run",
        "boltz",
        "predict",
        str(yaml_path),
        "--out_dir",
        str(out_dir),
        *BOLTZ_CLI_ARGS,
    ]
    logger.debug("Boltz CMD: {}", " ".join(cmd))

    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        timeout=BOLTZ_TIMEOUT,
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    return result


def _parse_affinity(out_dir: Path, stem: str) -> float:
    """Парсит affinity_pred_value из JSON результата Boltz.

    Boltz-2 возвращает log10(IC50 в микромолях), т.е. меньшие значения = сильнее связывание.
    Путь: {out_dir}/boltz_results_{stem}/predictions/{stem}/affinity_{stem}.json
    """
    affinity_path = out_dir / f"boltz_results_{stem}" / "predictions" / stem / f"affinity_{stem}.json"

    if not affinity_path.exists():
        msg = f"Affinity JSON не найден: {affinity_path}"
        raise FileNotFoundError(msg)

    data = json.loads(affinity_path.read_text())
    return float(data["affinity_pred_value"])


def score_single(target_id: str, smiles: str) -> float:
    """Оценить одну молекулу по мишени через Boltz-2.

    Returns:
    -------
    float
        log10(IC50 в микромолях); меньше = лучше.
        NaN при ошибке.
    """
    # Валидация SMILES
    if not _validate_smiles(smiles):
        logger.warning("Невалидный SMILES или слишком большая молекула (>{} атомов): {}", MAX_LIGAND_ATOMS, smiles)
        return float("nan")

    # Загрузить sequence
    sequence = _load_sequence(target_id)

    # Найти pre-computed MSA
    msa_path = _get_msa_path(target_id)
    if msa_path is None:
        msg = (
            f"Pre-computed MSA не найден для {target_id}. "
            f"Запустите: uv run python scripts/prepare_targets.py --pair <pair> --prepare-all"
        )
        raise FileNotFoundError(msg)

    with tempfile.TemporaryDirectory(prefix=f"boltz_{target_id}_") as tmpdir:
        tmp_path = Path(tmpdir)
        yaml_stem = "input"
        yaml_path = tmp_path / f"{yaml_stem}.yaml"
        out_dir = tmp_path / "output"

        _write_boltz_yaml(sequence, smiles, yaml_path, msa_path=msa_path)

        try:
            result = _run_boltz_predict(yaml_path, out_dir)
        except subprocess.TimeoutExpired:
            logger.error("Boltz таймаут ({} сек) для SMILES: {}", BOLTZ_TIMEOUT, smiles)
            return float("nan")

        if result.returncode != 0:
            stderr = result.stderr or ""
            if "CUDA out of memory" in stderr:
                logger.error("Boltz OOM GPU для SMILES: {}", smiles)
            else:
                logger.error(
                    "Boltz ошибка (rc={}) для SMILES: {}\nstderr: {}",
                    result.returncode,
                    smiles,
                    stderr[-2000:],
                )
            return float("nan")

        try:
            log_ic50_um = _parse_affinity(out_dir, yaml_stem)
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            logger.error("Не удалось распарсить affinity для SMILES {}: {}", smiles, exc)
            return float("nan")

    # Конвенция: log10(IC50 в микромолях), меньше = лучше
    return log_ic50_um


def score_batch(
    target_id: str,
    smiles_list: list[str],
    *,
    use_cache: bool = True,
) -> NDArray[np.float64]:
    """Оценить батч SMILES по мишени *target_id*.

    Скоры кешируются по паре (target, smiles) на диске.
    Каждая молекула скорится отдельным вызовом ``boltz predict``.

    Parameters
    ----------
    target_id:
        Идентификатор, соответствующий ключу в ``configs/pairs.yaml``.
    smiles_list:
        Канонические SMILES-строки.
    use_cache:
        Если True, пропускать уже оценённые молекулы.

    Returns:
    -------
    scores:
        1-D массив скоров (log10(IC50 в μM)), в том же порядке, что и *smiles_list*.
    """
    from tqdm import tqdm

    # Загрузить sequence один раз
    _load_sequence(target_id)

    cache = _load_cache(target_id) if use_cache else {}
    scores: list[float] = []
    to_score: list[tuple[int, str]] = []

    for idx, smi in enumerate(smiles_list):
        key = _cache_key(target_id, smi)
        if key in cache:
            scores.append(cache[key])
        else:
            scores.append(float("nan"))
            to_score.append((idx, smi))

    if to_score:
        logger.info("Скоринг {} новых молекул для мишени {} (из {} всего)", len(to_score), target_id, len(smiles_list))
        for idx, smi in tqdm(to_score, desc=f"Boltz-2 [{target_id}]", unit="mol"):
            s = score_single(target_id, smi)
            key = _cache_key(target_id, smi)
            cache[key] = s
            scores[idx] = s
            # Инкрементальное сохранение кеша после каждой молекулы
            if use_cache:
                _save_cache(target_id, cache)
    else:
        logger.info("Все {} молекул для мишени {} найдены в кеше", len(smiles_list), target_id)

    return np.array(scores, dtype=np.float64)
