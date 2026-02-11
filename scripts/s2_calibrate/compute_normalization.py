"""Calibrate normalization statistics for docking scores.

Loads per-target calibration scores, runs bootstrap stability checks,
and writes a ``normalization.yaml`` config used by downstream scoring
to z-score-normalize docking results.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import click
import numpy as np
import yaml
from loguru import logger

from dualdocking.config import load_pairs
from dualdocking.data.baseline import bootstrap_stability_check

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _boltz_version() -> str:
    """Return the installed Boltz package version.

    Returns:
        Version string, or ``"unknown"`` if not installed.
    """
    try:
        import importlib.metadata as md

        return md.version("boltz")
    except Exception:
        return "unknown"


_PROGRAM_ALIASES: dict[str, str] = {
    "unidock_gpu": "unidock",
}


def _resolve_program(program: str, override: str | None = None) -> str:
    """Resolve a docking program name to its binary.

    Args:
        program: Logical program name (may be aliased).
        override: Optional explicit binary path.

    Returns:
        Resolved binary name or path.
    """
    if override:
        return override
    return _PROGRAM_ALIASES.get(program, program)


def _docking_version(program: str) -> str:
    """Return docking version fallback without probing external binaries.

    Args:
        program: Binary name or absolute path.

    Returns:
        ``"unknown"`` (runtime version probing is intentionally disabled).
    """
    _ = program
    return "unknown"


def _load_scores(scores_dir: Path, target_id: str) -> np.ndarray:
    """Load calibration docking scores from a Parquet or CSV file.

    Args:
        scores_dir: Directory containing per-target score files.
        target_id: Target identifier (filename stem).

    Returns:
        1-D float array of scores.

    Raises:
        FileNotFoundError: If the score file does not exist.
    """
    from dualdocking.data.io_utils import read_df, stem_exists

    stem = scores_dir / target_id
    if not stem_exists(stem):
        raise FileNotFoundError(f"Scores not found for {target_id}: tried {stem}.parquet and {stem}.csv")
    df = read_df(stem)
    return df["score"].to_numpy(dtype=float)


@click.command()
@click.option("--calibration-config", default="configs/calibration.yaml")
@click.option("--calibration-set", default=None)
@click.option("--subset-name", default="zinc_druglike_calibration_1000_v2024")
@click.option("--n", default=1000, type=int)
@click.option("--seed", default=137, type=int)
@click.option("--scores-dir", default="output/data/calibration/scores")
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
    """Compute normalization statistics and write normalization.yaml.

    Loads calibration docking scores for each target, runs a bootstrap
    stability check, and writes per-target mean/std with metadata to a
    YAML file consumed by the scoring pipeline.

    Args:
        calibration_config: Path to the calibration YAML config.
        calibration_set: Override path to the calibration SMILES file.
        subset_name: Name label for the calibration subset.
        n: Expected number of calibration molecules.
        seed: Random seed used during calibration.
        scores_dir: Directory containing per-target score CSV files.
        output: Output path for the normalization YAML.
        docking_program: Logical docking program name.
        docking_binary: Override binary path for version detection.
        docking_version: Manually specify the docking version string.
    """
    if calibration_set is None:
        cfg = yaml.safe_load((PROJECT_ROOT / calibration_config).read_text())
        calibration_set = cfg.get("output_path", "output/data/calibration/zinc_druglike_calibration_1000_v2024.smi")
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
        if docking_version == "unknown":
            logger.warning(
                "Docking version unavailable from manifest/override for binary '{}'; writing docking_version=unknown.",
                binary,
            )

    normalization = {
        "version": "1.0",
        "calibration_set": str(calibration_set),
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
