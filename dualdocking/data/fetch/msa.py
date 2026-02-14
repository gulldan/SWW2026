"""Pre-compute MSA via Boltz + ColabFold API.

Extracted from ``cli/s1_prepare/prepare_targets.py``.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import yaml
from loguru import logger

MSA_RETRY_ATTEMPTS: int = 3
MSA_RETRY_BACKOFF: int = 30  # seconds


def _write_warmup_yaml(sequence: str, yaml_path: Path) -> None:
    """Generate a Boltz input YAML for MSA warmup (protein + benzene + affinity).

    Args:
        sequence: Protein amino acid sequence.
        yaml_path: Output path for the YAML file.
    """
    data = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": sequence}},
            {"ligand": {"id": "B", "smiles": "c1ccccc1"}},  # benzene as dummy
        ],
        "properties": [
            {"affinity": {"binder": "B"}},
        ],
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


def compute_msa(
    target_id: str,
    sequence: str,
    msa_cache_dir: Path,
    project_root: Path,
    *,
    retry_attempts: int = MSA_RETRY_ATTEMPTS,
    retry_backoff: int = MSA_RETRY_BACKOFF,
) -> Path | None:
    """Pre-compute MSA for a protein via Boltz + ColabFold API.

    Runs ``boltz predict`` with ``--use_msa_server`` and a dummy ligand.
    The MSA is cached in ``{msa_cache_dir}/{target_id}.csv``.

    Args:
        target_id: Internal target identifier.
        sequence: Protein amino acid sequence.
        msa_cache_dir: Directory for cached MSA CSV files.
        project_root: Project root for subprocess cwd.
        retry_attempts: Maximum number of attempts.
        retry_backoff: Base backoff in seconds between retries.

    Returns:
        Path to the MSA CSV file, or None on failure.
    """
    msa_csv = msa_cache_dir / f"{target_id}.csv"
    if msa_csv.exists():
        logger.info("MSA already exists: {}", msa_csv)
        return msa_csv

    out_dir = msa_cache_dir / target_id
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

    for attempt in range(1, retry_attempts + 1):
        logger.info("Pre-computing MSA for {} (attempt {}/{})", target_id, attempt, retry_attempts)
        logger.debug("CMD: {}", " ".join(cmd))

        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(project_root),
            check=False,
        )

        csv_files = list(out_dir.rglob("msa/*.csv"))
        if csv_files:
            logger.info("MSA .csv found after attempt {} (rc={})", attempt, result.returncode)
            if result.returncode != 0:
                logger.info("Boltz predict crashed after MSA stage -- expected during warmup")
            break

        if result.returncode != 0:
            logger.warning(
                "Boltz MSA error (attempt {}/{}): rc={}\nstderr: {}",
                attempt,
                retry_attempts,
                result.returncode,
                (result.stderr or "")[-1000:],
            )
        if attempt < retry_attempts:
            logger.info("Retrying in {} sec...", retry_backoff * attempt)
            time.sleep(retry_backoff * attempt)
    else:
        logger.error("Failed to pre-compute MSA for {} after {} attempts", target_id, retry_attempts)
        return None

    csv_files = list(out_dir.rglob("msa/*.csv"))
    if not csv_files:
        logger.error("MSA .csv file not found in {}", out_dir)
        return None

    src_csv = csv_files[0]
    shutil.copy2(src_csv, msa_csv)
    logger.info("MSA copied: {} -> {}", src_csv, msa_csv)
    return msa_csv
