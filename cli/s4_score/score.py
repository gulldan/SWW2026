"""Score generated molecules against both targets of a pair.

Reads molecules from ``results/<pair>/<mode>/seed_<seed>/generated.parquet``
(with CSV fallback), scores against both targets, and writes
``scores_A.parquet``, ``scores_B.parquet``, and ``scored_joined.parquet``
(plus CSV copies for compatibility).

Supports checkpointing: if ``scores_A.parquet`` (or ``.csv``) already exists,
target A scoring is skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if TYPE_CHECKING:
    from dualdocking.scoring.backends import ScoringBackend


def _score_one_target(
    target_id: str,
    smiles_list: list[str],
    scores_path: Path,
    force: bool,
    scoring_backend: ScoringBackend,
    *,
    use_cache: bool = True,
) -> np.ndarray:
    """Score a list of molecules against a single target via configured scorer.

    If ``scores_path`` already exists and *force* is ``False``, loads cached
    scores from disk instead of re-computing.

    Returns:
        Scores array aligned with *smiles_list*.
    """
    from dualdocking.scoring.checkpoint import load_checkpoint_scores, save_checkpoint_scores

    if not force:
        cached = load_checkpoint_scores(scores_path, smiles_list)
        if cached is not None:
            logger.info("Checkpoint: scores for target {} already exist ({}), skipping", target_id, scores_path)
            return cached

    logger.info("Scoring {} molecules against target {}", len(smiles_list), target_id)
    scores = np.asarray(scoring_backend.score_batch(target_id, smiles_list, use_cache=use_cache), dtype=float)

    save_checkpoint_scores(scores_path, smiles_list, scores)
    logger.info("Scores for target {} saved: {}", target_id, scores_path)
    return scores


def _build_joined(
    smiles_list: list[str],
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    pair: str,
    exp_cfg: dict,
) -> tuple[pd.DataFrame, str]:
    """Build the scored_joined DataFrame with objectives and hit flags."""
    from dualdocking.config import load_metrics
    from dualdocking.data.chem_sanitize import canonicalize
    from dualdocking.scoring.objectives import compute_combined, parse_objectives_config

    objectives = parse_objectives_config(exp_cfg)

    metrics_cfg = load_metrics(PROJECT_ROOT / "configs")
    thresholds = metrics_cfg.get("thresholds", {}).get(pair, {})
    thr_a = thresholds.get("target_a", 0.0)
    thr_b = thresholds.get("target_b", 0.0)

    smiles_canon = [canonicalize(s) or "" for s in smiles_list]
    is_hit_both = (scores_a <= thr_a) & (scores_b <= thr_b)

    joined = pd.DataFrame(
        {
            "smiles": smiles_list,
            "smiles_canonical": smiles_canon,
            "affinity_a": scores_a,
            "affinity_b": scores_b,
            "is_hit_both": is_hit_both,
        }
    )

    primary_col = "combined_min"
    for obj in objectives:
        joined[obj.column_name] = compute_combined(scores_a, scores_b, obj.kind, lower_is_better=True, **obj.params)
        if obj.is_primary:
            primary_col = obj.column_name
    joined = joined.sort_values(primary_col)

    return joined, primary_col


def _apply_filters(
    joined: pd.DataFrame,
    smiles_list: list[str],
    result_dir: Path,
) -> pd.DataFrame:
    """Apply drug-likeness filters and write filter_stats.json."""
    filter_cfg_path = PROJECT_ROOT / "configs" / "filters.yaml"
    if not filter_cfg_path.exists():
        return joined

    import json

    from dualdocking.data.filters import compute_filters, filter_results_to_columns, load_filter_config

    filter_cfg = load_filter_config(filter_cfg_path)
    filter_results, filter_stats = compute_filters(smiles_list, filter_cfg)
    joined = joined.assign(**filter_results_to_columns(filter_results))

    (result_dir / "filter_stats.json").write_text(json.dumps(filter_stats.to_dict(), indent=2))
    logger.info("Filter stats -> {}", result_dir / "filter_stats.json")

    return joined


@click.command()
@click.option("--pair", required=True, help="Pair ID from configs/pairs.yaml")
@click.option(
    "--mode",
    type=click.Choice(["single_a", "single_b", "double", "linear_equal", "biased_a", "selective_a"]),
    required=True,
)
@click.option("--seed", default=42, type=int)
@click.option("--config", default="configs/pairs.yaml")
@click.option("--experiment-config", default="configs/experiment.yaml", help="Experiment config for objectives")
@click.option("--force", is_flag=True, default=False, help="Re-score even if scores already exist")
@click.option("--prune-cache", is_flag=True, default=False, help="Remove stale cache files before scoring")
@click.option(
    "--molecules",
    default=None,
    type=click.Path(exists=True),
    help="Path to molecule file (.parquet/.csv/.smi). Overrides generated.parquet.",
)
def main(
    pair: str,
    mode: str,
    seed: int,
    config: str,
    experiment_config: str,
    force: bool,
    prune_cache: bool,
    molecules: str | None,
) -> None:
    """Score molecules against both targets and compute combined objectives.

    Multi-GPU parallelism is handled by the scoring backend (e.g. NIM
    async pool with round-robin across instances), not at target level.
    """
    from dualdocking.config import load_experiment, load_pairs
    from dualdocking.scoring.backends import resolve_scoring_backend

    exp_cfg = load_experiment(PROJECT_ROOT / experiment_config)
    scorer_cfg = exp_cfg.get("scorer", {})
    scorer_engine, scoring_backend = resolve_scoring_backend(scorer_cfg)
    logger.info("Scoring engine: {}", scorer_engine)

    use_cache = bool(scorer_cfg.get("cache_enabled", True))
    if prune_cache:
        n = scoring_backend.prune_stale_cache()
        logger.info("Pruned {} stale cache file(s)", n)

    pairs = load_pairs(PROJECT_ROOT / config)["pairs"]
    if pair not in pairs:
        logger.error("Pair '{}' not found", pair)
        sys.exit(1)

    pair_cfg = pairs[pair]
    from dualdocking.data.io_utils import read_df, read_molecules, stem_exists, write_df

    result_dir = PROJECT_ROOT / "output" / "results" / pair / mode / f"seed_{seed}"

    if molecules:
        smiles_list = read_molecules(Path(molecules))
        logger.info("Loaded {} molecules from {}", len(smiles_list), molecules)
    else:
        generated_stem = result_dir / "generated"
        if not stem_exists(generated_stem):
            logger.error("Molecules not found: {}.parquet/.csv", generated_stem)
            sys.exit(1)
        gen_df = read_df(generated_stem)
        smiles_list = gen_df["smiles"].tolist()

    target_a_id = pair_cfg["target_a"]["id"]
    target_b_id = pair_cfg["target_b"]["id"]

    scores_a = _score_one_target(
        target_a_id, smiles_list, result_dir / "scores_A.csv", force,
        scoring_backend, use_cache=use_cache,
    )
    scores_b = _score_one_target(
        target_b_id, smiles_list, result_dir / "scores_B.csv", force,
        scoring_backend, use_cache=use_cache,
    )

    joined, _primary_col = _build_joined(smiles_list, scores_a, scores_b, pair, exp_cfg)
    joined = _apply_filters(joined, smiles_list, result_dir)

    write_df(joined, result_dir / "scored_joined", csv_too=True)

    # Write scoring context metadata
    import json
    from datetime import UTC, datetime

    scoring_meta = scoring_backend.get_scoring_context_info()
    scoring_meta["scorer_engine"] = scorer_engine
    scoring_meta["timestamp"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    (result_dir / "scoring_context.json").write_text(json.dumps(scoring_meta, indent=2))
    logger.info("Scoring context -> {}", result_dir / "scoring_context.json")

    logger.info("Scores saved to {}", result_dir)
    logger.info(
        "Score stats — A: mean={:.2f} std={:.2f} | B: mean={:.2f} std={:.2f}",
        np.nanmean(scores_a),
        np.nanstd(scores_a),
        np.nanmean(scores_b),
        np.nanstd(scores_b),
    )


if __name__ == "__main__":
    main()
