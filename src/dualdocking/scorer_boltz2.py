"""Обёртка скоринга Boltz-2.

Предоставляет ``score_batch(target, smiles_list)`` с дисковым кешем.
Вызывает ``boltz predict`` через subprocess (Boltz-2 не имеет публичного Python API).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from dualdocking.cache_io import load_cache, save_cache

if TYPE_CHECKING:
    from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
DATA_DIR = PROJECT_ROOT / "data" / "targets"
MSA_CACHE_DIR = CACHE_DIR / "boltz_msa"

BOLTZ_TIMEOUT = 600  # секунд на одну молекулу
BOLTZ_BATCH_TIMEOUT = 3600  # секунд на батч
MAX_LIGAND_ATOMS = 128
_DEFAULT_BATCH_SIZE = 8  # молекул за один вызов boltz predict


def _auto_batch_size() -> int:
    """Pick batch size based on GPU VRAM; override via BOLTZ_BATCH_SIZE env var."""
    env = os.getenv("BOLTZ_BATCH_SIZE")
    if env is not None:
        return int(env)
    try:
        import torch

        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            if vram_gb >= 48:  # H100 80 GB / A100 40–80 GB
                return 32
            if vram_gb >= 20:  # RTX 4090 24 GB / A5000 24 GB
                return 16
    except Exception:
        logger.debug("GPU detection failed for batch size, using default={}", _DEFAULT_BATCH_SIZE)
    return _DEFAULT_BATCH_SIZE


BATCH_SIZE = _auto_batch_size()

# Boltz CLI параметры
_BOLTZ_CLI_ARGS_BASE: list[str] = [
    "--model",
    "boltz2",
    "--recycling_steps",
    "3",
    "--sampling_steps",
    "200",
    "--diffusion_samples",
    "4",
    "--sampling_steps_affinity",
    "200",
    "--diffusion_samples_affinity",
    "5",
    "--override",
]

_ACCELERATOR: str | None = None

# Scoring context hash — computed once per process
_CONTEXT_HASH: str | None = None

# Docker image for FREED++ (used for digest lookup)
_FREEDPP_DOCKER_IMAGE = "dualdocking/freedpp:latest"


def _resolve_accelerator() -> str:
    """Pick Boltz accelerator; GPU is required unless explicitly overridden."""
    global _ACCELERATOR
    if _ACCELERATOR is not None:
        return _ACCELERATOR

    env = os.getenv("BOLTZ_ACCELERATOR", "gpu").lower()
    if env not in {"gpu", "cpu"}:
        raise ValueError(f"Unsupported BOLTZ_ACCELERATOR={env}; use 'gpu' or 'cpu'")

    if env == "gpu":
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Boltz GPU requested but torch.cuda.is_available() is False. "
                    "Reinstall boltz with CUDA support and ensure NVIDIA drivers are visible."
                )
        except Exception as exc:
            raise RuntimeError(
                "Boltz GPU requested but CUDA is not available. Reinstall boltz with CUDA support and verify torch CUDA."
            ) from exc

    _ACCELERATOR = env
    return _ACCELERATOR


# ---------------------------------------------------------------------------
# Scoring context — v2 cache key
# ---------------------------------------------------------------------------


def _boltz_version() -> str:
    """Return installed Boltz version with fallback to master_config.yaml."""
    try:
        return importlib.metadata.version("boltz")
    except Exception:
        logger.debug("importlib.metadata.version('boltz') failed, trying master_config.yaml")
    try:
        return _master_config_field("software.boltz_version")
    except Exception:
        logger.warning("Cannot determine Boltz version; using 'unknown'")
        return "unknown"


def _master_config_field(path: str) -> str:
    """Read a dotted path from configs/master_config.yaml."""
    import yaml

    master_path = PROJECT_ROOT / "configs" / "master_config.yaml"
    if not master_path.exists():
        msg = f"master_config.yaml not found: {master_path}"
        raise FileNotFoundError(msg)
    data = yaml.safe_load(master_path.read_text())
    value = data
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = None
            break
    if not value:
        msg = f"Field '{path}' not found in master_config.yaml"
        raise KeyError(msg)
    return str(value)


def _run_cmd(cmd: list[str]) -> str:
    """Run a command, return stripped stdout or empty string on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    except FileNotFoundError:
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout or result.stderr or "").strip()


def _docker_digest() -> str:
    """Return FREED++ Docker image digest with fallback chain."""
    out = _run_cmd(["docker", "image", "inspect", _FREEDPP_DOCKER_IMAGE, "--format", "{{index .RepoDigests 0}}"])
    if "@" in out:
        return out.split("@")[-1]
    if out:
        return out
    # Local builds often have no RepoDigests; fall back to image ID.
    image_id = _run_cmd(["docker", "image", "inspect", _FREEDPP_DOCKER_IMAGE, "--format", "{{.Id}}"])
    if image_id:
        return image_id
    try:
        return _master_config_field("software.docker_image_digest")
    except Exception:
        logger.warning("Cannot determine Docker digest; using 'unknown'")
        return "unknown"


def _msa_content_hash(msa_path: Path | None) -> str:
    """SHA256[:16] of MSA file contents, or ``'no_msa'``."""
    if msa_path is None:
        return "no_msa"
    return hashlib.sha256(msa_path.read_bytes()).hexdigest()[:16]


def _sequence_content_hash(sequence: str) -> str:
    """SHA256[:16] of protein sequence."""
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]


def _scoring_context_hash() -> str:
    """Compute (and cache) the global scoring context hash."""
    global _CONTEXT_HASH
    if _CONTEXT_HASH is not None:
        return _CONTEXT_HASH

    boltz_ver = _boltz_version()
    cli_args = "|".join(_BOLTZ_CLI_ARGS_BASE)
    accelerator = os.getenv("BOLTZ_ACCELERATOR", "gpu").lower()
    digest = _docker_digest()

    blob = f"{boltz_ver}||{cli_args}||{accelerator}||{digest}".encode()
    _CONTEXT_HASH = hashlib.sha256(blob).hexdigest()[:16]

    logger.info(
        "Scoring context: boltz={}, cli_hash={}..., accel={}, docker={}... -> ctx={}",
        boltz_ver,
        hashlib.sha256(cli_args.encode()).hexdigest()[:8],
        accelerator,
        digest[:16],
        _CONTEXT_HASH,
    )
    return _CONTEXT_HASH


def get_cache_context_hash() -> str:
    """Public accessor for the scoring context hash."""
    return _scoring_context_hash()


def get_scoring_context_info() -> dict:
    """Return all scoring context components (for metadata persistence)."""
    boltz_ver = _boltz_version()
    cli_args = "|".join(_BOLTZ_CLI_ARGS_BASE)
    accelerator = os.getenv("BOLTZ_ACCELERATOR", "gpu").lower()
    digest = _docker_digest()
    return {
        "cache_context_hash": get_cache_context_hash(),
        "boltz_version": boltz_ver,
        "docker_digest": digest,
        "accelerator": accelerator,
        "cli_args_hash": hashlib.sha256(cli_args.encode()).hexdigest()[:16],
    }


def prune_stale_cache() -> int:
    """Remove cache files not matching the current context hash prefix.

    Returns the count of removed files.
    """
    ctx_prefix = _scoring_context_hash()[:8]
    removed = 0
    for f in CACHE_DIR.glob("*.json"):
        if f.suffix == ".lock":
            continue
        # Current-context files have pattern: {target_id}.{ctx[:8]}.json
        parts = f.stem.rsplit(".", 1)
        if len(parts) == 2 and parts[1] == ctx_prefix:
            continue
        # Legacy files (no context segment) or different context — remove
        f.unlink()
        logger.info("Pruned stale cache file: {}", f.name)
        removed += 1
    return removed


# ---------------------------------------------------------------------------
# Sequence cache + cache key helpers
# ---------------------------------------------------------------------------

# Кеш sequence в памяти (target_id -> sequence)
_sequence_cache: dict[str, str] = {}


def _cache_key(
    target_id: str,
    smi: str,
    *,
    ctx_hash: str = "",
    seq_hash: str = "",
    msa_hash: str = "",
) -> str:
    blob = f"{target_id}||{smi}||{ctx_hash}||{seq_hash}||{msa_hash}".encode()
    return hashlib.sha256(blob).hexdigest()


def _cache_path(target_id: str) -> Path:
    """Return context-segmented cache file path."""
    return CACHE_DIR / f"{target_id}.{_scoring_context_hash()[:8]}.json"


def _load_cache(target_id: str) -> dict[str, float]:
    return load_cache(_cache_path(target_id))


def _save_cache(target_id: str, cache: dict[str, float]) -> None:
    save_cache(_cache_path(target_id), cache)


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


def _boltz_env(device: str | None = None) -> dict[str, str]:
    """Build env dict for boltz subprocess, applying GPU pinning if configured.

    Parameters
    ----------
    device:
        Explicit CUDA device index (e.g. ``"0"``, ``"1"``).  When *None*,
        falls back to the ``BOLTZ_CUDA_DEVICE`` environment variable.
    """
    env = dict(os.environ)
    cuda_device = device if device is not None else os.getenv("BOLTZ_CUDA_DEVICE")
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_device
        logger.debug("Boltz GPU pinned to CUDA_VISIBLE_DEVICES={}", cuda_device)
    return env


def _run_boltz_predict(yaml_path: Path, out_dir: Path, *, device: str | None = None) -> subprocess.CompletedProcess[str]:
    """Запуск boltz predict через subprocess."""
    cmd = [
        "uv",
        "run",
        "boltz",
        "predict",
        str(yaml_path),
        "--out_dir",
        str(out_dir),
        "--accelerator",
        _resolve_accelerator(),
        *_BOLTZ_CLI_ARGS_BASE,
    ]
    logger.debug("Boltz CMD: {}", " ".join(cmd))

    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        timeout=BOLTZ_TIMEOUT,
        cwd=str(PROJECT_ROOT),
        env=_boltz_env(device=device),
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


def _run_boltz_predict_dir(
    input_dir: Path,
    out_dir: Path,
    timeout: int = BOLTZ_BATCH_TIMEOUT,
    *,
    device: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Запуск boltz predict на директории YAML файлов."""
    cmd = [
        "uv",
        "run",
        "boltz",
        "predict",
        str(input_dir),
        "--out_dir",
        str(out_dir),
        "--accelerator",
        _resolve_accelerator(),
        *_BOLTZ_CLI_ARGS_BASE,
    ]
    logger.debug("Boltz batch CMD: {}", " ".join(cmd))

    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
        env=_boltz_env(device=device),
        check=False,
    )
    return result


def _score_batch_internal(
    target_id: str,
    smiles_list: list[str],
    sequence: str,
    msa_path: Path,
    *,
    device: str | None = None,
) -> list[float]:
    """Скорить батч молекул за один вызов boltz predict.

    Создаёт директорию с YAML файлами и запускает boltz predict на ней.
    Модель загружается один раз, что значительно быстрее.
    """
    scores = [float("nan")] * len(smiles_list)

    # Валидация SMILES
    valid_indices: list[int] = []
    for idx, smi in enumerate(smiles_list):
        if _validate_smiles(smi):
            valid_indices.append(idx)
        else:
            logger.warning("Невалидный SMILES или слишком большая молекула (>{} атомов): {}", MAX_LIGAND_ATOMS, smi)

    if not valid_indices:
        return scores

    with tempfile.TemporaryDirectory(prefix=f"boltz_batch_{target_id}_") as tmpdir:
        tmp_path = Path(tmpdir)
        input_dir = tmp_path / "inputs"
        out_dir = tmp_path / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        # Создаём YAML для каждой валидной молекулы
        stem_to_idx: dict[str, int] = {}
        for idx in valid_indices:
            smi = smiles_list[idx]
            stem = f"mol_{idx:04d}"
            yaml_path = input_dir / f"{stem}.yaml"
            _write_boltz_yaml(sequence, smi, yaml_path, msa_path=msa_path)
            stem_to_idx[stem] = idx

        # Один вызов boltz predict на директорию
        timeout = BOLTZ_BATCH_TIMEOUT
        try:
            result = _run_boltz_predict_dir(input_dir, out_dir, timeout=timeout, device=device)
        except subprocess.TimeoutExpired:
            logger.error("Boltz batch таймаут ({} сек) для {} молекул", timeout, len(valid_indices))
            return scores

        if result.returncode != 0:
            stderr = result.stderr or ""
            if "CUDA out of memory" in stderr:
                logger.error("Boltz batch OOM GPU для {} молекул", len(valid_indices))
            else:
                logger.error(
                    "Boltz batch ошибка (rc={}) для {} молекул\nstderr: {}",
                    result.returncode,
                    len(valid_indices),
                    stderr[-2000:],
                )
            # Попробуем распарсить частичные результаты

        # Парсим результаты для каждой молекулы
        # При батчевом запуске структура: boltz_results_{dir_name}/predictions/{stem}/affinity_{stem}.json
        results_dir = out_dir / f"boltz_results_{input_dir.name}"
        for stem, idx in stem_to_idx.items():
            affinity_path = results_dir / "predictions" / stem / f"affinity_{stem}.json"
            try:
                if not affinity_path.exists():
                    logger.debug("Affinity файл не найден: {}", affinity_path)
                    continue
                data = json.loads(affinity_path.read_text())
                scores[idx] = float(data["affinity_pred_value"])
            except (KeyError, json.JSONDecodeError) as exc:
                logger.debug("Не удалось распарсить affinity для {}: {}", stem, exc)

    return scores


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
    device: str | None = None,
    use_cache: bool = True,
    batch_size: int = BATCH_SIZE,
) -> NDArray[np.float64]:
    """Оценить батч SMILES по мишени *target_id*.

    Скоры кешируются по паре (target, smiles) на диске.
    Молекулы группируются в батчи и обрабатываются за один вызов boltz predict,
    что значительно ускоряет работу (модель загружается один раз на батч).

    Parameters
    ----------
    target_id:
        Идентификатор, соответствующий ключу в ``configs/pairs.yaml``.
    smiles_list:
        Канонические SMILES-строки.
    use_cache:
        Если True, пропускать уже оценённые молекулы.
    batch_size:
        Количество молекул в одном батче (по умолчанию BATCH_SIZE).

    Returns:
    -------
    scores:
        1-D массив скоров (log10(IC50 в μM)), в том же порядке, что и *smiles_list*.
    """
    from tqdm import tqdm

    # Загрузить sequence и MSA один раз
    sequence = _load_sequence(target_id)
    msa_path = _get_msa_path(target_id)
    if msa_path is None:
        msg = (
            f"Pre-computed MSA не найден для {target_id}. "
            f"Запустите: uv run python scripts/prepare_targets.py --pair <pair> --prepare-all"
        )
        raise FileNotFoundError(msg)

    # Compute per-target hashes fresh (no stale data risk)
    seq_hash = _sequence_content_hash(sequence)
    msa_hash = _msa_content_hash(msa_path)
    ctx = _scoring_context_hash()

    cache = _load_cache(target_id) if use_cache else {}
    scores: list[float] = [float("nan")] * len(smiles_list)
    to_score: list[tuple[int, str]] = []

    # Проверяем кеш
    for idx, smi in enumerate(smiles_list):
        key = _cache_key(target_id, smi, ctx_hash=ctx, seq_hash=seq_hash, msa_hash=msa_hash)
        if key in cache:
            scores[idx] = cache[key]
        else:
            to_score.append((idx, smi))

    if not to_score:
        logger.info("Все {} молекул для мишени {} найдены в кеше", len(smiles_list), target_id)
        return np.array(scores, dtype=np.float64)

    logger.info(
        "Скоринг {} новых молекул для мишени {} (из {} всего, batch_size={})",
        len(to_score),
        target_id,
        len(smiles_list),
        batch_size,
    )

    # Группируем в батчи
    num_batches = (len(to_score) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc=f"Boltz-2 [{target_id}]", unit="batch"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(to_score))
        batch = to_score[start:end]

        # Извлекаем SMILES для батча
        batch_smiles = [smi for _, smi in batch]

        # Скорим батч за один вызов boltz predict
        batch_scores = _score_batch_internal(target_id, batch_smiles, sequence, msa_path, device=device)

        # Записываем результаты
        for i, (idx, smi) in enumerate(batch):
            s = batch_scores[i]
            scores[idx] = s
            if not np.isnan(s):
                key = _cache_key(target_id, smi, ctx_hash=ctx, seq_hash=seq_hash, msa_hash=msa_hash)
                cache[key] = s

        # Сохраняем кеш после каждого батча (не после каждой молекулы)
        if use_cache:
            _save_cache(target_id, cache)

    return np.array(scores, dtype=np.float64)
