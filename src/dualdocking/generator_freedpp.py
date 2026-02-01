"""Обёртка генератора FREED++ через Docker.

FREED++ требует Python 3.7 / PyTorch 1.12 / DGL 0.9 — несовместимо с
проектом (Python ≥3.11), поэтому изоляция через Docker.

Reward при обучении: QuickVina2 (встроен в FREED++, ~1 сек/мол, CPU).
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
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from dualdocking.binding_box import extract_binding_box

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "targets"
CACHE_DIR = PROJECT_ROOT / "cache" / "freedpp_models"

DEFAULT_DOCKER_IMAGE = "dualdocking/freedpp:latest"

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
    reward_mode: str = "single"  # "single" | "double"
    second_target_id: str | None = None

    # FREED++-специфичные
    epochs: int = 200
    num_sub_proc: int = 12
    starting_smile: str = "c1([*:1])c([*:2])ccc([*:3])c1"
    use_cache: bool = True
    docker_image: str = DEFAULT_DOCKER_IMAGE
    training_timeout: int = 86400  # 24 ч
    box_padding: float = 10.0  # Å
    vina_program: str = "qvina02"  # "qvina02" | "unidock"
    steps_per_epoch: int = 256


# ---------------------------------------------------------------------------
# Docker image management
# ---------------------------------------------------------------------------


def _ensure_docker_image(image: str = DEFAULT_DOCKER_IMAGE) -> None:
    """Проверить наличие Docker-образа, собрать при необходимости."""
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
    )
    if result.returncode == 0:
        logger.debug("Docker-образ {} найден", image)
        return

    logger.info("Docker-образ {} не найден, собираю...", image)
    dockerfile_dir = PROJECT_ROOT / "docker" / "freedpp"
    if not (dockerfile_dir / "Dockerfile").exists():
        raise FileNotFoundError(f"Dockerfile не найден: {dockerfile_dir / 'Dockerfile'}")

    subprocess.run(
        ["docker", "build", "-t", image, str(dockerfile_dir)],
        check=True,
    )
    logger.info("Docker-образ {} собран", image)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _cache_key(config: GenerationConfig) -> str:
    """SHA256-хеш от параметров, влияющих на обученную модель."""
    parts = [
        config.target_id,
        config.second_target_id or "",
        config.reward_mode,
        str(config.epochs),
        str(config.seed),
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
        subprocess.run(
            [
                "docker",
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
# Vina program resolution
# ---------------------------------------------------------------------------

_VINA_PATHS: dict[str, str] = {
    "qvina02": "/opt/freed/ffreed/env/qvina02",
    "unidock": "/usr/local/bin/unidock",
}


def _resolve_vina_path(program: str) -> str:
    """Map vina_program name to container-internal binary path."""
    if program in _VINA_PATHS:
        return _VINA_PATHS[program]
    # Treat as an explicit path (e.g. user-provided absolute path)
    return program


# ---------------------------------------------------------------------------
# FREED++ CLI args builder
# ---------------------------------------------------------------------------


def _build_freedpp_args(
    config: GenerationConfig,
    workspace: Path,
    pdbqt_a: Path,
    center_a: str,
    size_a: str,
    commands: str = "train,sample",
) -> list[str]:
    """Собрать аргументы CLI для main.py FREED++."""
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
        str(config.num_sub_proc),
        "--seed",
        str(config.seed),
        "--num_mols",
        str(config.n_molecules),
        "--steps_per_epoch",
        str(config.steps_per_epoch),
        "--reward_version",
        "soft",
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
    """Запустить FREED++ внутри Docker контейнера."""
    args = _build_freedpp_args(config, workspace, pdbqt_a, center_a, size_a, commands)

    docker_cmd: list[str] = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{workspace}:/workspace",
    ]

    # Dual-target: mount второй receptor и конфиг
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

    logger.info("Запуск FREED++: {}", " ".join(docker_cmd))
    result = subprocess.run(
        docker_cmd,
        capture_output=True,
        text=True,
        timeout=config.training_timeout,
    )

    if result.returncode != 0:
        logger.error("FREED++ завершился с ошибкой (rc={})", result.returncode)
        logger.error("stderr: {}", (result.stderr or "")[-2000:])
        result.check_returncode()

    return result


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

    # JSON формат (sample_*.json) — основной выход FREED++
    json_files = sorted(mols_dir.glob("sample_*.json"))
    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text())
            if isinstance(data, list):
                all_smiles.extend(str(s) for s in data if s)
        except Exception as e:
            logger.warning("Ошибка чтения {}: {}", json_path, e)

    # CSV формат (sample_*.csv) — fallback
    csv_files = sorted(mols_dir.glob("sample_*.csv"))
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            if "smiles" in df.columns:
                all_smiles.extend(df["smiles"].dropna().tolist())
            elif "Smiles" in df.columns:
                all_smiles.extend(df["Smiles"].dropna().tolist())
        except Exception as e:
            logger.warning("Ошибка чтения {}: {}", csv_path, e)

    if not all_smiles:
        logger.warning("Файлы с молекулами не найдены в {}", mols_dir)

    logger.info("Прочитано {} SMILES из {} файлов", len(all_smiles), len(json_files) + len(csv_files))
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
    # 1. Docker image
    _ensure_docker_image(config.docker_image)

    # 2. Cache check
    key = _cache_key(config)
    cached_model = _find_cached_model(key) if config.use_cache else None

    # 3. Workspace
    workspace = config.output_dir / "freedpp_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    # 4. Prepare receptor A
    pdb_a = _resolve_pdb_path(config.target_id)
    pdbqt_a, center_a, size_a = _prepare_receptor(
        pdb_a,
        workspace,
        config.docker_image,
        config.box_padding,
    )

    # 5. Prepare receptor B (double mode)
    pdbqt_b = center_b = size_b = None
    if config.reward_mode == "double" and config.second_target_id:
        pdb_b = _resolve_pdb_path(config.second_target_id)
        pdbqt_b, center_b, size_b = _prepare_receptor(
            pdb_b,
            workspace,
            config.docker_image,
            config.box_padding,
        )

    # 6. Determine commands
    if cached_model is not None:
        # Восстановить модель из кеша → только sampling
        import shutil

        model_target = workspace / "output" / "run"
        if not model_target.exists():
            shutil.copytree(cached_model, model_target, dirs_exist_ok=True)
        commands = "sample"
        logger.info("Используется кешированная модель ({}) → только sampling", key)
    else:
        commands = "train,sample"

    # 7. Run FREED++
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

    # 8. Parse output
    smiles = _parse_output_smiles(workspace)

    # 9. Cache trained model
    if config.use_cache and commands == "train,sample":
        model_dir = workspace / "output" / "run"
        if model_dir.exists():
            cache_dest = CACHE_DIR / key
            cache_dest.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copytree(model_dir, cache_dest, dirs_exist_ok=True)
            logger.info("Модель сохранена в кеш: {}", cache_dest)

    # 10. Write output CSV
    config.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.output_dir / "generated.csv"
    out_path.write_text("smiles\n" + "\n".join(smiles) + "\n")
    logger.info("Сгенерировано {} молекул -> {}", len(smiles), out_path)

    return smiles
