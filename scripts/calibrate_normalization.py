"""Calibrate normalization statistics for docking scores."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import click
import numpy as np
import yaml
from loguru import logger

from dualdocking.baseline import bootstrap_stability_check
from dualdocking.config import load_pairs

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _boltz_version() -> str:
    try:
        import importlib.metadata as md

        return md.version("boltz")
    except Exception:
        return "unknown"


_PROGRAM_ALIASES: dict[str, str] = {
    "unidock_gpu": "unidock",
}

_ALLOWED_BINARIES = {"unidock", "qvina02"}


def _resolve_program(program: str, override: str | None = None) -> str:
    if override:
        return override
    return _PROGRAM_ALIASES.get(program, program)


def _docking_version(program: str) -> str:
    try:
        binary = program
        if binary not in _ALLOWED_BINARIES and not Path(binary).is_absolute():
            return "unknown"
        result = subprocess.run([binary, "--version"], capture_output=True, text=True, check=False)  # noqa: S603
        output = (result.stdout or result.stderr or "").strip()
        return output.splitlines()[0] if output else "unknown"
    except Exception:
        return "unknown"


def _load_scores(scores_dir: Path, target_id: str) -> np.ndarray:
    path = scores_dir / f"{target_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Scores not found for {target_id}: {path}")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 0:
        data = np.array([float(data)])
    return data.astype(float)


@click.command()
@click.option("--calibration-config", default="configs/calibration.yaml")
@click.option("--calibration-set", default=None)
@click.option("--subset-name", default="zinc_druglike_calibration_1000_v2024")
@click.option("--n", default=1000, type=int)
@click.option("--seed", default=137, type=int)
@click.option("--scores-dir", default="data/calibration/scores")
@click.option("--output", default="configs/normalization.yaml")
@click.option("--docking-program", default="unidock_gpu")
@click.option("--docking-binary", default=None, help="Binary to query for version (optional)")
@click.option("--docking-version", default=None, help="Override docking version (optional)")
def main(
    calibration_config: str,
    calibration_set: str | None,
    subset_name: str,
    n: int,
    seed: int,
    scores_dir: str,
    output: str,
    docking_program: str,
    docking_binary: str | None,
    docking_version: str | None,
) -> None:
    """Compute normalization statistics and write normalization.yaml."""
    if calibration_set is None:
        cfg = yaml.safe_load((PROJECT_ROOT / calibration_config).read_text())
        calibration_set = cfg.get("output_path", "data/calibration/zinc_druglike_calibration_1000_v2024.smi")
    calibration_path = PROJECT_ROOT / calibration_set
    scores_path = PROJECT_ROOT / scores_dir

    if calibration_path.exists():
        checksum = _sha256(calibration_path)
    else:
        checksum = "unknown"
        logger.warning("Calibration set file not found: {}", calibration_path)

    pairs = load_pairs(PROJECT_ROOT / "configs")["pairs"]
    target_ids: list[str] = []
    for pair in pairs.values():
        target_ids.append(pair["target_a"]["id"])
        target_ids.append(pair["target_b"]["id"])

    manifest_path = scores_path / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("docking_program") and manifest.get("docking_program") != docking_program:
                logger.warning(
                    "Docking program mismatch: manifest={} vs args={}",
                    manifest.get("docking_program"),
                    docking_program,
                )
            if docking_version is None:
                docking_version = manifest.get("docking_version")
        except Exception as exc:
            logger.warning("Failed to read docking manifest {}: {}", manifest_path, exc)

    if docking_version is None:
        binary = _resolve_program(docking_program, docking_binary)
        docking_version = _docking_version(binary)

    normalization = {
        "version": "1.0",
        "calibration_set": str(calibration_path),
        "subset_name": subset_name,
        "n": int(n),
        "seed": int(seed),
        "checksum": checksum,
        "date": datetime.now(UTC).date().isoformat(),
        "boltz_version": _boltz_version(),
        "docking_program": docking_program,
        "docking_version": docking_version,
        "targets": {},
    }

    for target_id in target_ids:
        scores = _load_scores(scores_path, target_id)
        stats = bootstrap_stability_check(scores, n_bootstrap=1000)
        normalization["targets"][target_id] = {
            "mean": float(stats["mean"]),
            "std": float(stats["std"]),
            "ci_mean": stats["ci_mean"],
            "ci_std": stats["ci_std"],
            "n_samples": int(len(scores)),
            "stable_mean": bool(stats["stable_mean"]),
            "stable_std": bool(stats["stable_std"]),
        }

    out_path = PROJECT_ROOT / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(normalization, sort_keys=False))

    unstable = [t for t, entry in normalization["targets"].items() if not entry["stable_mean"] or not entry["stable_std"]]
    if unstable:
        logger.warning("Unstable targets detected: {}", ", ".join(unstable))

    logger.info("Normalization saved -> {}", out_path)


if __name__ == "__main__":
    main()
