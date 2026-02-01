"""Скачивание и подготовка структур белковых мишеней.

Читает пары из configs/pairs.yaml, скачивает PDB-файлы и раскладывает
в data/targets/<pair_id>/{A,B}/.

С флагом ``--prepare-all`` дополнительно:
- Скачивает FASTA-последовательности из UniProt → data/targets/{target_id}.fasta
- Pre-compute MSA для каждого белка → cache/boltz_msa/{target_id}.csv
- Скачивает Boltz checkpoints (автоматически при первом запуске)

После ``--prepare-all`` весь пайплайн скоринга работает OFFLINE.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import click
import requests
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "targets"
MSA_CACHE_DIR = PROJECT_ROOT / "cache" / "boltz_msa"

MSA_RETRY_ATTEMPTS = 3
MSA_RETRY_BACKOFF = 30  # секунд

FREEDPP_DOCKER_IMAGE = "dualdocking/freedpp:latest"


def download_pdb(pdb_id: str, output_path: Path) -> Path:
    """Скачать PDB-файл из RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    logger.info("Скачивание {} -> {}", url, output_path)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(resp.text)
    return output_path


def fetch_sequence(uniprot_id: str, target_id: str) -> Path:
    """Скачать FASTA-последовательность из UniProt.

    GET https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta
    → data/targets/{target_id}.fasta
    """
    fasta_path = DATA_DIR / f"{target_id}.fasta"
    if fasta_path.exists():
        logger.info("FASTA уже существует: {}", fasta_path)
        return fasta_path

    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    logger.info("Скачивание FASTA {} -> {}", url, fasta_path)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    fasta_path.write_text(resp.text)
    logger.info("FASTA сохранён: {} ({} байт)", fasta_path, len(resp.text))
    return fasta_path


def _read_sequence(fasta_path: Path) -> str:
    """Прочитать последовательность из FASTA файла (пропуская заголовки)."""
    lines = fasta_path.read_text().strip().splitlines()
    return "".join(line.strip() for line in lines if not line.startswith(">"))


def _write_warmup_yaml(sequence: str, yaml_path: Path) -> None:
    """Сгенерировать YAML для MSA warmup (protein + бензол + affinity)."""
    import yaml

    data = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": sequence}},
            {"ligand": {"id": "B", "smiles": "c1ccccc1"}},  # бензол как dummy
        ],
        "properties": [
            {"affinity": {"binder": "B"}},
        ],
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


def precompute_msa(target_id: str, sequence: str) -> Path | None:
    """Pre-compute MSA для белка через Boltz + ColabFold API.

    Запускает ``boltz predict`` с ``--use_msa_server`` и dummy лигандом.
    MSA кешируется в cache/boltz_msa/{target_id}.csv.

    Returns:
    -------
    Path | None
        Путь к MSA .csv файлу или None при ошибке.
    """
    msa_csv = MSA_CACHE_DIR / f"{target_id}.csv"
    if msa_csv.exists():
        logger.info("MSA уже существует: {}", msa_csv)
        return msa_csv

    out_dir = MSA_CACHE_DIR / target_id
    out_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = out_dir / "msa_warmup.yaml"
    _write_warmup_yaml(sequence, yaml_path)

    cmd = [
        "uv",
        "run",
        "boltz",
        "predict",
        str(yaml_path),
        "--out_dir",
        str(out_dir),
        "--model",
        "boltz2",
        "--accelerator",
        "gpu",
        "--use_msa_server",
    ]

    for attempt in range(1, MSA_RETRY_ATTEMPTS + 1):
        logger.info("Pre-computing MSA для {} (попытка {}/{})", target_id, attempt, MSA_RETRY_ATTEMPTS)
        logger.debug("CMD: {}", " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(PROJECT_ROOT),
        )

        # MSA генерируется до GPU-инференса. Даже если predict упал
        # (например, нет cuequivariance_torch), MSA .csv уже создан.
        csv_files = list(out_dir.rglob("msa/*.csv"))
        if csv_files:
            logger.info("MSA .csv найден после попытки {} (rc={})", attempt, result.returncode)
            if result.returncode != 0:
                logger.info("Boltz predict упал после MSA-этапа — это ожидаемо при warmup")
            break

        if result.returncode != 0:
            logger.warning(
                "Boltz MSA ошибка (попытка {}/{}): rc={}\nstderr: {}",
                attempt,
                MSA_RETRY_ATTEMPTS,
                result.returncode,
                (result.stderr or "")[-1000:],
            )
        if attempt < MSA_RETRY_ATTEMPTS:
            logger.info("Повтор через {} сек...", MSA_RETRY_BACKOFF * attempt)
            time.sleep(MSA_RETRY_BACKOFF * attempt)
    else:
        logger.error("Не удалось pre-compute MSA для {} после {} попыток", target_id, MSA_RETRY_ATTEMPTS)
        return None

    csv_files = list(out_dir.rglob("msa/*.csv"))
    if not csv_files:
        logger.error("MSA .csv файл не найден в {}", out_dir)
        return None

    # Скопировать первый найденный .csv → cache/boltz_msa/{target_id}.csv
    src_csv = csv_files[0]
    shutil.copy2(src_csv, msa_csv)
    logger.info("MSA скопирован: {} -> {}", src_csv, msa_csv)
    return msa_csv


def prepare_receptor_pdbqt(pdb_path: Path, target_id: str) -> Path:
    """Конвертировать PDB → PDBQT через Docker OpenBabel.

    Parameters
    ----------
    pdb_path : Path
        Путь к PDB файлу.
    target_id : str
        ID мишени (для имени выходного файла).

    Returns:
    -------
    Path
        Путь к PDBQT файлу.
    """
    pdbqt_path = DATA_DIR / f"{target_id}.pdbqt"
    if pdbqt_path.exists():
        logger.info("PDBQT уже существует: {}", pdbqt_path)
        return pdbqt_path

    # Проверить наличие Docker-образа
    inspect = subprocess.run(
        ["docker", "image", "inspect", FREEDPP_DOCKER_IMAGE],
        capture_output=True,
    )
    if inspect.returncode != 0:
        logger.warning(
            "Docker-образ {} не найден. Сборка: docker build -t {} docker/freedpp/",
            FREEDPP_DOCKER_IMAGE,
            FREEDPP_DOCKER_IMAGE,
        )
        return pdbqt_path

    logger.info("Конвертация PDB → PDBQT: {} → {}", pdb_path, pdbqt_path)
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{pdb_path.parent}:/input:ro",
            "-v",
            f"{DATA_DIR}:/output",
            FREEDPP_DOCKER_IMAGE,
            "prepare-receptor",
            f"/input/{pdb_path.name}",
            f"/output/{target_id}.pdbqt",
        ],
        check=True,
        timeout=300,
    )
    logger.info("PDBQT сохранён: {}", pdbqt_path)
    return pdbqt_path


def prepare_pair(pair_id: str, pair_cfg: dict, *, prepare_all: bool = False) -> None:
    """Подготовить входные данные для одной пары."""
    base = PROJECT_ROOT / "data" / "targets" / pair_id

    for role in ("A", "B"):
        target = pair_cfg[f"target_{role.lower()}"]
        pdb_id = target["pdb_id"]
        target_id = target["id"]
        uniprot_id = target["uniprot"]
        out_dir = base / role
        out_dir.mkdir(parents=True, exist_ok=True)

        # Шаг 1: PDB
        pdb_path = out_dir / f"{pdb_id.lower()}.pdb"
        if pdb_path.exists():
            logger.info("Уже существует: {}", pdb_path)
        else:
            download_pdb(pdb_id, pdb_path)

        # Сохранить метаданные
        meta_path = out_dir / "meta.yaml"
        if not meta_path.exists():
            import yaml

            meta_path.write_text(yaml.dump(target, default_flow_style=False))

        if prepare_all:
            # Шаг 2: FASTA
            fasta_path = fetch_sequence(uniprot_id, target_id)

            # Шаг 3: MSA pre-compute
            sequence = _read_sequence(fasta_path)
            precompute_msa(target_id, sequence)

            # Шаг 4: PDB → PDBQT (для FREED++ / QuickVina2)
            prepare_receptor_pdbqt(pdb_path, target_id)


@click.command()
@click.option("--config", default="configs/pairs.yaml", help="Путь к конфигу пар")
@click.option("--pair", default=None, help="Обработать только эту пару")
@click.option(
    "--prepare-all",
    is_flag=True,
    default=False,
    help="Полная подготовка: PDB + FASTA + MSA + Boltz checkpoints (нужен интернет)",
)
def main(config: str, pair: str | None, prepare_all: bool) -> None:
    """Скачать и подготовить белковые мишени."""
    import yaml

    config_path = PROJECT_ROOT / config
    with config_path.open() as f:
        pairs = yaml.safe_load(f)["pairs"]

    if pair:
        if pair not in pairs:
            logger.error("Пара '{}' не найдена в конфиге", pair)
            sys.exit(1)
        pairs = {pair: pairs[pair]}

    for pair_id, pair_cfg in pairs.items():
        logger.info("Подготовка пары: {}", pair_id)
        prepare_pair(pair_id, pair_cfg, prepare_all=prepare_all)

    if prepare_all:
        logger.info("Полная подготовка завершена. Пайплайн скоринга готов к работе OFFLINE.")
    logger.info("Готово. Мишени сохранены в data/targets/")


if __name__ == "__main__":
    main()
