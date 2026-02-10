"""Обёртка генератора FREED++ через Docker.

FREED++ требует Python 3.7 / PyTorch 1.12 / DGL 0.9 — несовместимо с
проектом (Python ≥3.11), поэтому изоляция через Docker.

Reward при обучении: UniDock GPU (unidock_gpu, GPU-ускоренный докинг).
Финальный скоринг: Boltz-2 (scorer_boltz2.py, ~58 сек/мол, GPU).

Ссылки
------
- Статья FREED++: https://arxiv.org/abs/2401.09840
- Оригинал FREED: https://arxiv.org/abs/2110.01219
- Репозиторий: https://github.com/AIRI-Institute/FFREED
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from dualdocking.scoring.binding_box import extract_binding_box

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "targets"
CACHE_DIR = PROJECT_ROOT / "cache" / "freedpp_models"

DEFAULT_DOCKER_IMAGE = "dualdocking/freedpp:latest"
DOCKER_BIN = shutil.which("docker") or "docker"
CACHE_KEY_VERSION = "v3_clean_single_b_20260207"
FREED_DATA_DIR = PROJECT_ROOT / "docker" / "freedpp" / "data"
FREED_DATA_FILES = (
    FREED_DATA_DIR / "zinc_crem.json",
    FREED_DATA_DIR / "alert_collections.csv",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GenerationConfig:
    """Параметры одного запуска генерации FREED++."""

    target_id: str
    n_molecules: int = 1000
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("results/smoke"))
    reward_mode: str = "single_a"  # "single_a" | "single_b" | "double"
    second_target_id: str | None = None

    # FREED++-специфичные
    epochs: int = 200
    num_sub_proc: int = 0  # 0 = auto-detect based on GPU VRAM
    starting_smile: str = "c1([*:1])c([*:2])ccc([*:3])c1"
    use_cache: bool = True
    docker_image: str = DEFAULT_DOCKER_IMAGE
    training_timeout: int = 86400  # 24 ч
    box_padding: float = 10.0  # Å
    vina_program: str = "unidock_gpu"  # "qvina02" | "unidock" | "unidock_gpu"
    steps_per_epoch: int = 256
    qg_max_invalid_fraction: float = 0.10
    qg_max_timeout_fraction: float = 0.05


# ---------------------------------------------------------------------------
# Docker image management
# ---------------------------------------------------------------------------


def _auto_num_sub_proc() -> int:
    """Pick num_sub_proc based on GPU VRAM; override via FREEDPP_NUM_SUB_PROC env var.

    VRAM tiers (concurrent UniDock GPU processes):
        >=80 GB (H100/A100-80)  -> 24
         >=40 GB (A6000 48 GB)  -> 12
         <=24 GB (RTX 4090)     ->  8
    Default (detection failure) ->  8 (safe for any consumer GPU)
    """
    env = os.getenv("FREEDPP_NUM_SUB_PROC")
    if env is not None:
        return int(env)

    vram_gb: float | None = None

    # Attempt 1: torch.cuda (fast, in-process)
    try:
        import torch

        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    except Exception:
        pass

    # Attempt 2: nvidia-smi (works even without CUDA-enabled torch)
    if vram_gb is None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "--id=0"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                vram_gb = float(result.stdout.strip().split("\n")[0]) / 1024
        except Exception:
            pass

    if vram_gb is None:
        logger.debug("GPU VRAM detection failed for num_sub_proc, using default=8")
        return 8

    logger.debug("GPU VRAM detected: {:.1f} GB", vram_gb)
    if vram_gb >= 76:  # H100/H800 80 GB / A100 80 GB (nvidia-smi reports ~79.x GiB)
        return 24
    if vram_gb >= 38:  # A6000 48 GB / A100 40 GB
        return 12
    return 8  # RTX 4090 24 GB and smaller


def _ensure_docker_image(image: str = DEFAULT_DOCKER_IMAGE) -> None:
    """Проверить наличие Docker-образа, собрать при необходимости."""
    result = subprocess.run(  # noqa: S603
        [DOCKER_BIN, "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        logger.debug("Docker-образ {} найден", image)
        return

    stderr = result.stderr or ""
    # Distinguish "image not found" from Docker daemon errors
    if "No such image" not in stderr and "not found" not in stderr.lower():
        raise RuntimeError(f"Docker inspect failed (rc={result.returncode}): {stderr.strip()}")

    logger.info("Docker-образ {} не найден, собираю...", image)
    dockerfile_dir = PROJECT_ROOT / "docker" / "freedpp"
    dockerfile_path = dockerfile_dir / "Dockerfile"
    if not dockerfile_path.exists():
        raise FileNotFoundError(f"Dockerfile не найден: {dockerfile_path}")

    missing_files = [p for p in FREED_DATA_FILES if not p.exists()]
    if missing_files:
        logger.info("Отсутствуют FREED data files, скачиваю: {}", ", ".join(str(p) for p in missing_files))
        fetch_script = PROJECT_ROOT / "scripts" / "fetch_freed_data.py"
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(fetch_script)],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            tail = ((result.stdout or "") + (result.stderr or ""))[-2000:]
            raise RuntimeError(f"Не удалось скачать FREED data files:\n{tail}")

    subprocess.run(  # noqa: S603
        [
            DOCKER_BIN,
            "build",
            "-t",
            image,
            "-f",
            str(dockerfile_path),
            str(PROJECT_ROOT),
        ],
        check=True,
    )
    logger.info("Docker-образ {} собран", image)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _cache_key(config: GenerationConfig) -> str:
    """SHA256-хеш от параметров, влияющих на обученную модель."""
    parts = [
        CACHE_KEY_VERSION,
        config.target_id,
        config.second_target_id or "",
        config.reward_mode,
        str(config.epochs),
        str(config.steps_per_epoch),
        str(config.num_sub_proc),
        str(config.seed),
        config.vina_program,
        config.starting_smile,
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _find_cached_model(key: str) -> Path | None:
    """Найти кеш обученной модели по ключу."""
    model_dir = CACHE_DIR / key
    if model_dir.exists() and any(model_dir.iterdir()):
        logger.info("Найден кеш модели: {}", model_dir)
        return model_dir
    return None


# ---------------------------------------------------------------------------
# PDB resolution
# ---------------------------------------------------------------------------


def _resolve_pdb_path(target_id: str) -> Path:
    """Найти PDB файл для target_id в data/targets/.

    Ищет в директориях ``data/targets/*/A/`` и ``data/targets/*/B/``,
    сопоставляя target_id с meta.yaml.
    """
    import yaml

    for pair_dir in DATA_DIR.iterdir():
        if not pair_dir.is_dir():
            continue
        for role in ("A", "B"):
            role_dir = pair_dir / role
            meta_path = role_dir / "meta.yaml"
            if not meta_path.exists():
                continue
            with meta_path.open() as f:
                meta = yaml.safe_load(f)
            if meta.get("id") == target_id:
                pdb_files = list(role_dir.glob("*.pdb"))
                if pdb_files:
                    return pdb_files[0]

    raise FileNotFoundError(f"PDB файл для target '{target_id}' не найден в {DATA_DIR}")


# ---------------------------------------------------------------------------
# Receptor preparation
# ---------------------------------------------------------------------------


def _prepare_receptor(
    pdb_path: Path,
    out_dir: Path,
    image: str = DEFAULT_DOCKER_IMAGE,
    padding: float = 10.0,
) -> tuple[Path, str, str]:
    """Конвертировать PDB → PDBQT и определить binding box.

    Returns:
    -------
    tuple[Path, str, str]
        (pdbqt_path, center_str, size_str)
    """
    pdbqt_path = out_dir / (pdb_path.stem + ".pdbqt")
    if not pdbqt_path.exists():
        logger.info("Конвертация PDB → PDBQT: {} → {}", pdb_path, pdbqt_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(  # noqa: S603
            [
                DOCKER_BIN,
                "run",
                "--rm",
                "-v",
                f"{pdb_path.parent}:/input:ro",
                "-v",
                f"{out_dir}:/output",
                image,
                "prepare-receptor",
                f"/input/{pdb_path.name}",
                f"/output/{pdbqt_path.name}",
            ],
            check=True,
            timeout=60,
        )
    else:
        logger.debug("PDBQT уже существует: {}", pdbqt_path)

    center_str, size_str = extract_binding_box(pdb_path, padding=padding)
    logger.info("Binding box: center={}, size={}", center_str, size_str)
    return pdbqt_path, center_str, size_str


# ---------------------------------------------------------------------------
# Docking program resolution (параметр vina_program из FREED++ API)
# ---------------------------------------------------------------------------

_VINA_PATHS: dict[str, str] = {
    "qvina02": "/opt/freed/ffreed/env/qvina02",
    "unidock": "/usr/local/bin/unidock",
    # В контейнере Uni-Dock GPU устанавливается как `unidock`.
    # Оставляем ключ `unidock_gpu` для согласованности метаданных.
    "unidock_gpu": "/usr/local/bin/unidock",
}


def _resolve_vina_path(program: str) -> str:
    """Map vina_program name (FREED++ API) to container-internal binary path."""
    if program in _VINA_PATHS:
        return _VINA_PATHS[program]
    # Treat as an explicit path (e.g. user-provided absolute path)
    return program


# ---------------------------------------------------------------------------
# FREED++ CLI args builder
# ---------------------------------------------------------------------------


def _build_freedpp_args(
    config: GenerationConfig,
    pdbqt_a: Path,
    center_a: str,
    size_a: str,
    commands: str = "train,sample",
) -> list[str]:
    """Собрать аргументы CLI для main.py FREED++."""
    num_sub_proc = config.num_sub_proc if config.num_sub_proc > 0 else _auto_num_sub_proc()
    logger.debug("FREED++ num_sub_proc={} (config={}, auto={})", num_sub_proc, config.num_sub_proc, config.num_sub_proc == 0)
    return [
        "--commands",
        commands,
        "--receptor",
        f"/workspace/{pdbqt_a.name}",
        "--box_center",
        center_a,
        "--box_size",
        size_a,
        "--fragments",
        "/opt/freed/data/zinc_crem.json",
        "--alert_collections",
        "/opt/freed/data/alert_collections.csv",
        "--vina_program",
        _resolve_vina_path(config.vina_program),
        "--starting_smile",
        config.starting_smile,
        "--exp_root",
        "/workspace/output",
        "--name",
        "run",
        "--epochs",
        str(config.epochs),
        "--num_sub_proc",
        str(num_sub_proc),
        "--seed",
        str(config.seed),
        "--num_mols",
        str(config.n_molecules),
        "--steps_per_epoch",
        str(config.steps_per_epoch),
        "--reward_version",
        "soft",
        "--num_modes",
        "1",
    ]


# ---------------------------------------------------------------------------
# Docker run
# ---------------------------------------------------------------------------


def _run_freedpp(
    config: GenerationConfig,
    workspace: Path,
    pdbqt_a: Path,
    center_a: str,
    size_a: str,
    pdbqt_b: Path | None = None,
    center_b: str | None = None,
    size_b: str | None = None,
    commands: str = "train,sample",
) -> subprocess.CompletedProcess[str]:
    """Запустить FREED++ внутри Docker контейнера.

    Вывод Docker стримится в реальном времени для наблюдения за прогрессом.
    """
    # For single_b, B is the primary receptor for FREED++ generation + reward.
    # _build_freedpp_args param names (pdbqt_a, center_a, size_a) refer to the
    # primary receptor, not necessarily target A.
    if config.reward_mode == "single_b":
        if pdbqt_b is None:
            raise ValueError("single_b requires receptor B (second_target_id must be set)")
        if center_b is None or size_b is None:
            raise ValueError("single_b requires center_b and size_b")
        args = _build_freedpp_args(config, pdbqt_b, center_b, size_b, commands)
    else:
        args = _build_freedpp_args(config, pdbqt_a, center_a, size_a, commands)

    gpu_device = os.getenv("FREEDPP_CUDA_DEVICE")
    gpu_flag = f'"device={gpu_device}"' if gpu_device is not None else "all"

    docker_cmd: list[str] = [
        DOCKER_BIN,
        "run",
        "--rm",
        "--gpus",
        gpu_flag,
        "-e",
        "PYTHONUNBUFFERED=1",
        "-v",
        f"{workspace}:/workspace",
    ]

    # Передать контекст для reward (нормализация + режим)
    docker_cmd.extend(
        [
            "-e",
            f"REWARD_MODE={config.reward_mode}",
            "-e",
            f"TARGET_ID_A={config.target_id}",
            "-e",
            f"TARGET_ID_B={config.second_target_id or ''}",
            "-e",
            "NORMALIZATION_PATH=/workspace/normalization.yaml",
            "-e",
            f"QG_MAX_INVALID_FRACTION={config.qg_max_invalid_fraction}",
            "-e",
            f"QG_MAX_TIMEOUT_FRACTION={config.qg_max_timeout_fraction}",
            "-e",
            "GPU_BATCH_DOCK=1",
        ]
    )

    # Dual-target: mount второй receptor и конфиг (only needed for double mode)
    if config.reward_mode == "double" and pdbqt_b is not None:
        dual_cfg = {
            "receptor_b": f"/workspace/{pdbqt_b.name}",
            "center_b": center_b,
            "size_b": size_b,
        }
        dual_config_path = workspace / "dual_config.json"
        dual_config_path.write_text(json.dumps(dual_cfg, indent=2))

        docker_cmd.extend(
            [
                "-e",
                f"DUAL_RECEPTOR_B=/workspace/{pdbqt_b.name}",
                "-e",
                "DUAL_CONFIG=/workspace/dual_config.json",
            ]
        )

    docker_cmd.append(config.docker_image)
    docker_cmd.extend(args)

    def _ts() -> str:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")

    run_id = os.environ.get("EXPERIMENT_RUN_ID", "adhoc")
    log_ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = workspace / f"freedpp_{config.reward_mode}_seed_{config.seed}_{run_id}_{log_ts}.log"

    logger.info("Запуск FREED++: {}", " ".join(docker_cmd))
    logger.info("FREED++ лог -> {}", log_path)

    # Стриминг вывода в реальном времени для видимости прогресса
    epoch_bar = None
    log_file = None
    stderr_lines: list[str] = []
    process = None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        log_file.write(f"[{_ts()}] RUN_ID={run_id}\n")
        log_file.write(f"[{_ts()}] CMD: {' '.join(docker_cmd)}\n")
        log_file.flush()

        process = subprocess.Popen(  # noqa: S603
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if "train" in commands:
            epoch_bar = tqdm(total=config.epochs, desc="FREED++ обучение", unit="epoch")

        train_start = time.monotonic()
        last_epoch_logged = -1
        line_count = 0

        # Multiple patterns to catch FREED++ epoch/iteration progress
        _epoch_re = re.compile(
            r"[Ee](?:poch|pisode)\s*[:\[#]?\s*(\d+)"  # Epoch 1, Episode 1, epoch: 1
            r"|#(\d+)"  # #1, #001
            r"|\[(\d+)/\d+\]"  # [1/200]
            r"|[Ii]ter(?:ation)?\s*[:\[#]?\s*(\d+)"  # Iter 1, Iteration: 1
        )
        _noise_patterns = ("DeprecationWarning", "SyntaxWarning", "FutureWarning", "UserWarning", "warnings.warn")

        for line in process.stdout:  # type: ignore[union-attr]
            stripped = line.rstrip()
            log_file.write(f"[{_ts()}] {stripped}\n")
            log_file.flush()
            line_count += 1

            # Skip noisy warnings
            if any(pat in stripped for pat in _noise_patterns):
                stderr_lines.append(line)
                continue

            # Парсинг прогресса эпох/эпизодов из вывода FREED++
            if epoch_bar is not None:
                match = _epoch_re.search(stripped)
                if match:
                    # Take first non-None group
                    epoch_num = int(next(g for g in match.groups() if g is not None))
                    epoch_bar.n = min(epoch_num, config.epochs)
                    epoch_bar.refresh()
                    log_interval = max(1, config.epochs // 10)
                    if epoch_num > 0 and epoch_num % log_interval == 0 and epoch_num != last_epoch_logged:
                        last_epoch_logged = epoch_num
                        elapsed = time.monotonic() - train_start
                        sec_per_epoch = elapsed / epoch_num
                        remaining = (config.epochs - epoch_num) * sec_per_epoch
                        logger.info(
                            "[FREED++] Epoch {}/{} — {:.0f}s elapsed, ~{:.0f}m remaining ({:.1f}s/epoch)",
                            epoch_num,
                            config.epochs,
                            elapsed,
                            remaining / 60,
                            sec_per_epoch,
                        )
                elif line_count % 500 == 0:
                    # Fallback: show line count in postfix if no epoch pattern matched yet
                    elapsed = time.monotonic() - train_start
                    epoch_bar.set_postfix_str(f"{line_count} lines, {elapsed:.0f}s")

            # Показываем ключевые строки в логе (без warnings-шума)
            low = stripped.lower()
            if any(kw in low for kw in ("error", "sample", "saving", "loss", "score", "result")):
                logger.info("[FREED++] {}", stripped)
            stderr_lines.append(line)

        process.wait(timeout=config.training_timeout)
    finally:
        if epoch_bar is not None:
            epoch_bar.close()
        if log_file is not None:
            log_file.close()

    stdout_text = "".join(stderr_lines)

    if process is None or process.returncode != 0:
        rc = process.returncode if process is not None else -1
        logger.error("FREED++ завершился с ошибкой (rc={})", rc)
        logger.error("output: {}", stdout_text[-2000:])
        logger.error("FREED++ лог -> {}", log_path)
        raise subprocess.CalledProcessError(rc, docker_cmd, output=stdout_text)

    return subprocess.CompletedProcess(docker_cmd, process.returncode, stdout=stdout_text, stderr="")


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _parse_output_smiles(workspace: Path) -> list[str]:
    """Прочитать сгенерированные SMILES из выходных файлов FREED++.

    FREED++ сохраняет результаты в ``{exp_root}/{name}/mols/``:
    - ``sample_*.json`` — JSON-массив SMILES (основной формат)
    - ``sample_*.csv`` — CSV со столбцом ``smiles`` (legacy)
    """
    import pandas as pd

    mols_dir = workspace / "output" / "run" / "mols"
    all_smiles: list[str] = []

    # JSON format (primary)
    json_files = sorted(mols_dir.glob("sample_*.json"))
    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text())
            if isinstance(data, list):
                all_smiles.extend(str(s) for s in data if s)
        except Exception as e:
            logger.warning("Error reading {}: {}", json_path, e)

    # CSV format (fallback ONLY if no JSON found)
    if not all_smiles:
        csv_files = sorted(mols_dir.glob("sample_*.csv"))
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                col = "smiles" if "smiles" in df.columns else "Smiles" if "Smiles" in df.columns else None
                if col:
                    all_smiles.extend(df[col].dropna().tolist())
            except Exception as e:
                logger.warning("Error reading {}: {}", csv_path, e)

    if not all_smiles:
        logger.warning("No molecule files found in {}", mols_dir)

    logger.info("Read {} SMILES from output", len(all_smiles))
    return all_smiles


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate(config: GenerationConfig) -> list[str]:
    """Запустить генерацию FREED++ и вернуть список SMILES.

    Полный пайплайн:
    1. Проверить / собрать Docker-образ
    2. Проверить кеш обученных моделей
    3. Подготовить receptor(s): PDB → PDBQT + binding box
    4. Запустить FREED++ (train + sample или только sample)
    5. Прочитать выходные SMILES
    6. Сохранить в кеш и записать CSV

    Parameters
    ----------
    config : GenerationConfig
        Параметры генерации.

    Returns:
    -------
    list[str]
        Список сгенерированных SMILES.
    """
    needs_second = config.reward_mode in {"single_b", "double"}
    if needs_second and not config.second_target_id:
        raise ValueError(f"{config.reward_mode} mode requires second_target_id")
    n_steps = 8 if not needs_second else 9
    pbar = tqdm(
        total=n_steps,
        desc=f"Генерация [{config.target_id}]",
        unit="step",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

    # 1. Docker image
    pbar.set_postfix_str("Docker-образ")
    _ensure_docker_image(config.docker_image)
    pbar.update(1)

    # 2. Cache check
    pbar.set_postfix_str("Проверка кеша")
    key = _cache_key(config)
    cached_model = _find_cached_model(key) if config.use_cache else None
    pbar.update(1)

    # 3. Workspace
    pbar.set_postfix_str("Подготовка workspace")
    workspace = config.output_dir / "freedpp_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    pbar.update(1)

    # 3.1. Нормализация (нужна для reward)
    normalization_src = PROJECT_ROOT / "configs" / "normalization.yaml"
    if not normalization_src.exists():
        raise FileNotFoundError(f"normalization.yaml не найден: {normalization_src}")
    normalization_dst = workspace / "normalization.yaml"
    if not normalization_dst.exists():
        shutil.copy2(normalization_src, normalization_dst)
        logger.info("normalization.yaml -> {}", normalization_dst)

    # 4. Prepare receptor A
    pbar.set_postfix_str(f"Receptor A → PDBQT ({config.target_id})")
    pdb_a = _resolve_pdb_path(config.target_id)
    pdbqt_a, center_a, size_a = _prepare_receptor(
        pdb_a,
        workspace,
        config.docker_image,
        config.box_padding,
    )
    pbar.update(1)

    # 5. Prepare receptor B (single_b / double mode)
    pdbqt_b = center_b = size_b = None
    if needs_second and config.second_target_id:
        pbar.set_postfix_str(f"Receptor B → PDBQT ({config.second_target_id})")
        pdb_b = _resolve_pdb_path(config.second_target_id)
        pdbqt_b, center_b, size_b = _prepare_receptor(
            pdb_b,
            workspace,
            config.docker_image,
            config.box_padding,
        )
        pbar.update(1)

    # 6. Determine commands
    if cached_model is not None:
        # Восстановить модель из кеша → только sampling
        model_target = workspace / "output" / "run"
        if not model_target.exists():
            shutil.copytree(cached_model, model_target, dirs_exist_ok=True)
        commands = "sample"
        logger.info("Используется кешированная модель ({}) → только sampling", key)
    else:
        commands = "train,sample"
        # Очистить stale output от предыдущего прерванного запуска
        stale_output = workspace / "output" / "run"
        if stale_output.exists():
            logger.warning("Обнаружен stale output от прерванного запуска: {}", stale_output)
            shutil.rmtree(stale_output, ignore_errors=True)
            # Fallback: Docker (root-owned файлы)
            if stale_output.exists():
                logger.info("Удаление root-owned файлов через Docker...")
                subprocess.run(  # noqa: S603
                    [
                        DOCKER_BIN,
                        "run",
                        "--rm",
                        "-v",
                        f"{workspace}:/ws",
                        "alpine",
                        "rm",
                        "-rf",
                        "/ws/output/run",
                    ],
                    check=False,
                    timeout=30,
                )
            logger.info("Stale output удалён")

    # 7. Run FREED++
    pbar.set_postfix_str(f"FREED++ ({commands})")
    _run_freedpp(
        config,
        workspace,
        pdbqt_a,
        center_a,
        size_a,
        pdbqt_b,
        center_b,
        size_b,
        commands=commands,
    )
    pbar.update(1)

    # 8. Parse output
    pbar.set_postfix_str("Парсинг SMILES")
    smiles = _parse_output_smiles(workspace)
    pbar.update(1)

    # 9. Cache trained model
    pbar.set_postfix_str("Кеширование модели")
    if config.use_cache and commands == "train,sample":
        model_dir = workspace / "output" / "run"
        if model_dir.exists():
            cache_dest = CACHE_DIR / key
            cache_dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(model_dir, cache_dest, dirs_exist_ok=True)
            logger.info("Модель сохранена в кеш: {}", cache_dest)
    pbar.update(1)

    # 10. Write output CSV
    pbar.set_postfix_str("Запись CSV")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.output_dir / "generated.csv"
    out_path.write_text("smiles\n" + "\n".join(smiles) + "\n")
    logger.info("Сгенерировано {} молекул -> {}", len(smiles), out_path)
    pbar.close()

    return smiles
