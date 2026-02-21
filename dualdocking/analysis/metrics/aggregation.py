"""Aggregate scored results across all pairs into a single analysis dataset.

Reads ``results/<pair>/<mode>/seed_<seed>/scored_joined.csv`` for every
pair, mode, and seed, validates quality gates, concatenates the frames,
and writes a Parquet file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from loguru import logger


@dataclass(frozen=True)
class AggregateConfig:
    """Configuration for result aggregation.

    Attributes:
        project_root: Absolute path to project root.
        pairs: Pair configuration dict (pair_id -> config), excluding blocked.
        modes: Generation modes (e.g. ``["single_a", "single_b", "double"]``).
        seeds: Random seeds used across runs.
        quality_gate: Quality gate thresholds dict (``max_invalid_fraction``,
            ``max_timeout_fraction``).
        results_root: Root directory containing per-pair results.
        output_path: Output Parquet file path (absolute).
    """

    project_root: Path
    pairs: dict[str, dict]
    modes: list[str]
    seeds: list[int]
    quality_gate: dict[str, float] = field(default_factory=dict)
    results_root: Path = Path("output/results")
    output_path: Path = Path("report/data/all_results.parquet")


def aggregate(config: AggregateConfig) -> Path:
    """Aggregate scored results from all (pair, mode, seed) combinations.

    Iterates over every combination, applies the quality gate,
    and concatenates valid runs into a single Parquet file.

    Args:
        config: Aggregation configuration.

    Returns:
        Absolute path to the written Parquet file.

    Raises:
        SystemExit: If no valid results are found.
    """
    from dualdocking.analysis.fairness import validate_reward_quality
    from dualdocking.data.io_utils import read_df, stem_exists

    results_root = config.results_root if config.results_root.is_absolute() else config.project_root / config.results_root
    frames: list[pd.DataFrame] = []

    for pair_id in config.pairs:
        for mode in config.modes:
            for seed in config.seeds:
                run_dir = results_root / pair_id / mode / f"seed_{seed}"
                scored_stem = run_dir / "scored_joined"
                if not stem_exists(scored_stem):
                    logger.warning("Missing: {}", scored_stem)
                    continue

                rq = validate_reward_quality(
                    run_dir,
                    max_invalid_fraction=config.quality_gate.get("max_invalid_fraction", 0.10),
                    max_timeout_fraction=config.quality_gate.get("max_timeout_fraction", 0.05),
                )
                if not rq["run_valid"]:
                    logger.warning("Quality gate failed, skipping {}: {}", run_dir, rq["issues"])
                    continue

                df = read_df(scored_stem)

                # Backward compat: ensure filter columns exist
                filter_cols = [
                    "passes_lipinski",
                    "passes_qed",
                    "passes_pains",
                    "passes_sa",
                    "passes_veber",
                    "passes_all_filters",
                ]
                for col in filter_cols:
                    if col not in df.columns:
                        df[col] = True

                # Merge GNINA cross-validation scores if available
                gnina_stem = run_dir / "gnina_scores"
                if stem_exists(gnina_stem):
                    gnina_df = read_df(gnina_stem)
                    gnina_cols = [c for c in gnina_df.columns if c.startswith("gnina_")]
                    if "smiles" in gnina_df.columns and gnina_cols:
                        df = df.merge(gnina_df[["smiles", *gnina_cols]], on="smiles", how="left")
                        logger.debug("Merged {} GNINA columns from {}", len(gnina_cols), gnina_stem)

                df["pair"] = pair_id
                df["mode"] = mode
                df["seed"] = seed
                frames.append(df)
                logger.info("Loaded {} rows from {}", len(df), scored_stem)

    if not frames:
        import sys

        logger.error("No results found. Run generation and scoring first.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    out_path = config.output_path if config.output_path.is_absolute() else config.project_root / config.output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    logger.info("Aggregated {} rows -> {}", len(combined), out_path)
    return out_path
