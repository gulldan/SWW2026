"""Compute SHA256 checksums and update configs/master_config.yaml."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCKER_BIN = shutil.which("docker") or "docker"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _docker_image_digest(image: str) -> str:
    try:
        result = subprocess.run(  # noqa: S603
            [DOCKER_BIN, "image", "inspect", image, "--format", "{{index .RepoDigests 0}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (result.stdout or "").strip()
        if "@" in out:
            return out.split("@")[-1]
        return out or "UNKNOWN"
    except FileNotFoundError:
        return "UNKNOWN"


def _boltz_version() -> str:
    try:
        import importlib.metadata as md

        return md.version("boltz")
    except Exception:
        return "UNKNOWN"


def _normalization_meta() -> tuple[str, str]:
    norm_path = PROJECT_ROOT / "configs" / "normalization.yaml"
    if not norm_path.exists():
        return "unidock_gpu", "UNKNOWN"
    data = yaml.safe_load(norm_path.read_text())
    return data.get("docking_program", "unidock_gpu"), data.get("docking_version", "UNKNOWN")


def main() -> None:
    master_path = PROJECT_ROOT / "configs" / "master_config.yaml"
    if not master_path.exists():
        raise FileNotFoundError(f"master_config.yaml not found: {master_path}")

    master = yaml.safe_load(master_path.read_text())

    for section in ("data_sources", "configs"):
        for entry in master.get(section, {}).values():
            path_str = entry.get("path")
            if not path_str:
                continue
            path = PROJECT_ROOT / path_str
            if not path.exists():
                logger.warning("Missing file for checksum: {}", path)
                entry["sha256"] = "MISSING"
            else:
                entry["sha256"] = _sha256(path)

    # Update software versions
    docking_program, docking_version = _normalization_meta()
    master.setdefault("software", {})["docking_program"] = docking_program
    master.setdefault("software", {})["docking_version"] = docking_version
    master.setdefault("software", {})["boltz_version"] = _boltz_version()

    # Docker image digest from experiment config
    exp_path = PROJECT_ROOT / "configs" / "experiment.yaml"
    if exp_path.exists():
        exp = yaml.safe_load(exp_path.read_text())
        image = exp.get("generator", {}).get("freedpp", {}).get("docker_image", "dualdocking/freedpp:latest")
        master.setdefault("software", {})["docker_image_digest"] = _docker_image_digest(image)

    master_path.write_text(yaml.safe_dump(master, sort_keys=False))
    logger.info("Updated checksums -> {}", master_path)


if __name__ == "__main__":
    main()
