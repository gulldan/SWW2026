"""Compute z-score normalization statistics from calibration docking scores.

Loads per-target calibration scores, runs bootstrap stability checks,
and produces a normalization config dict used by downstream scoring
to z-score-normalize docking results.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger

from dualdocking.config import load_pairs
from dualdocking.data.baseline import bootstrap_stability_check
from dualdocking.data.http_utils import sha256_file
from dualdocking.scoring.docking import resolve_program


def boltz_version() -> str:
    """Return the installed Boltz package version.

    Returns:
        Version string, or ``"unknown"`` if not installed.
    """
    try:
        import importlib.metadata as md

        return md.version("boltz")
    except Exception:
        return "unknown"


def docking_version_fallback(program: str) -> str:
    """Return docking version fallback without probing external binaries.

    Args:
        program: Binary name or absolute path.

    Returns:
        ``"unknown"`` (runtime version probing is intentionally disabled).
    """
    _ = program
    return "unknown"


def load_scores(scores_dir: Path, target_id: str) -> np.ndarray:
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


def compute_normalization(  # noqa: PLR0913
    project_root: Path,
    calibration_config: str = "configs/calibration.yaml",
    calibration_set: str | None = None,
    subset_name: str = "zinc_druglike_calibration_1000_v2024",
    n: int = 1000,
    seed: int = 137,
    scores_dir: str = "output/data/calibration/scores",
    output: str = "configs/normalization.yaml",
    docking_program: str = "unidock_gpu",
    docking_binary: str | None = None,
    docking_version_override: str | None = None,
) -> dict[str, Any]:
    """Compute normalization statistics and return the normalization dict.

    Loads calibration docking scores for each target, runs a bootstrap
    stability check, and produces per-target mean/std with metadata.

    Args:
        project_root: Project root directory.
        calibration_config: Path to the calibration YAML config.
        calibration_set: Override path to the calibration SMILES file.
        subset_name: Name label for the calibration subset.
        n: Expected number of calibration molecules.
        seed: Random seed used during calibration.
        scores_dir: Directory containing per-target score CSV files.
        output: Output path for the normalization YAML.
        docking_program: Logical docking program name.
        docking_binary: Override binary path for version detection.
        docking_version_override: Manually specify the docking version string.

    Returns:
        The normalization dict (also written to *output* YAML).
    """
    dock_ver = docking_version_override

    if calibration_set is None:
        cfg = yaml.safe_load((project_root / calibration_config).read_text())
        calibration_set = cfg.get("output_path", "output/data/calibration/zinc_druglike_calibration_1000_v2024.smi")
    calibration_path = project_root / calibration_set
    scores_path = project_root / scores_dir

    if calibration_path.exists():
        checksum = sha256_file(calibration_path)
    else:
        checksum = "unknown"
        logger.warning("Calibration set file not found: {}", calibration_path)

    pairs = load_pairs(project_root / "configs")["pairs"]
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
            if dock_ver is None:
                dock_ver = manifest.get("docking_version")
        except Exception as exc:
            logger.warning("Failed to read docking manifest {}: {}", manifest_path, exc)

    if dock_ver is None:
        binary = resolve_program(docking_program, docking_binary)
        dock_ver = docking_version_fallback(binary)
        if dock_ver == "unknown":
            logger.warning(
                "Docking version unavailable from manifest/override for binary '{}'; writing docking_version=unknown.",
                binary,
            )

    normalization: dict[str, Any] = {
        "version": "1.0",
        "calibration_set": str(calibration_set),
        "subset_name": subset_name,
        "n": int(n),
        "seed": int(seed),
        "checksum": checksum,
        "date": datetime.now(UTC).date().isoformat(),
        "boltz_version": boltz_version(),
        "docking_program": docking_program,
        "docking_version": dock_ver,
        "targets": {},
    }

    skipped: list[str] = []
    for tid in target_ids:
        try:
            scores = load_scores(scores_path, tid)
        except FileNotFoundError:
            skipped.append(tid)
            continue
        stats = bootstrap_stability_check(scores, n_bootstrap=1000)
        normalization["targets"][tid] = {
            "mean": float(stats["mean"]),
            "std": float(stats["std"]),
            "ci_mean": stats["ci_mean"],
            "ci_std": stats["ci_std"],
            "n_samples": int(len(scores)),
            "stable_mean": bool(stats["stable_mean"]),
            "stable_std": bool(stats["stable_std"]),
        }
    if skipped:
        logger.info("Skipped {} target(s) without scores: {}", len(skipped), ", ".join(skipped))

    out_path = project_root / output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(normalization, sort_keys=False))

    unstable = [t for t, entry in normalization["targets"].items() if not entry["stable_mean"] or not entry["stable_std"]]
    if unstable:
        logger.warning("Unstable targets detected: {}", ", ".join(unstable))

    logger.info("Normalization saved -> {}", out_path)

    return normalization
