"""Score generated molecules against both targets of a pair.

Reads molecules from ``results/<pair>/<mode>/seed_<seed>/generated.parquet``
(with CSV fallback), scores against both targets, and writes
``scores_A.parquet``, ``scores_B.parquet``, and ``scored_joined.parquet``
(plus CSV copies for compatibility).

Supports checkpointing: if ``scores_A.parquet`` (or ``.csv``) already exists,
target A scoring is skipped.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if TYPE_CHECKING:
    from dualdocking.scoring.backends import ScoringBackend


def _detect_gpu_count() -> int:
    """Detect number of CUDA GPUs available.

    Returns:
        Number of CUDA devices, or 1 if detection fails.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        logger.debug("GPU detection failed for parallel scoring, assuming 1 GPU")
    return 1


def _score_one_target(
    target_id: str,
    smiles_list: list[str],
    scores_path: Path,
    force: bool,
    label: str,
    scoring_backend: ScoringBackend,
    *,
    use_cache: bool = True,
    device: str | None = None,
) -> tuple[str, np.ndarray]:
    """Score a list of molecules against a single target via configured scorer.

    If ``scores_path`` already exists and *force* is ``False``, loads cached
    scores from disk instead of re-computing.

    Args:
        target_id: Target identifier (e.g. ``"CDK2"``).
        smiles_list: SMILES strings to score.
        scores_path: Path to write/read the per-target score Parquet (with CSV fallback).
        force: If ``True``, re-score even when checkpoint exists.
        label: Short label for logging (``"A"`` or ``"B"``).
        scoring_backend: Resolved scoring backend implementation.
        use_cache: Whether backend-level cache should be used.
        device: CUDA device identifier for parallel scoring.

    Returns:
        Tuple of ``(label, scores_array)``.
    """
    from dualdocking.data.io_utils import stem_exists

    if not force and stem_exists(scores_path.with_suffix("")):
        logger.info("Checkpoint: scores for target {} already exist ({}), skipping", target_id, scores_path)
        from dualdocking.data.io_utils import read_df

        saved = read_df(scores_path.with_suffix(""))
        return label, saved["score"].to_numpy()

    logger.info("Scoring {} molecules against target {} (device={})", len(smiles_list), target_id, device)
    scores = np.asarray(scoring_backend.score_batch(target_id, smiles_list, use_cache=use_cache, device=device), dtype=float)

    from dualdocking.data.io_utils import write_df

    write_df(pd.DataFrame({"smiles": smiles_list, "score": scores}), scores_path.with_suffix(""), csv_too=True)
    logger.info("Scores for target {} saved: {}", target_id, scores_path)
    return label, scores


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
    "--parallel-scoring",
    type=click.Choice(["auto", "on", "off"]),
    default="auto",
    show_default=True,
    help="Parallel target scoring: 'auto' enables when >=2 GPUs detected, 'on'/'off' force.",
)
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
    parallel_scoring: str,
    molecules: str | None,
) -> None:
    """Score molecules against both targets and compute combined objectives.

    Loads generated SMILES, runs configured scorer for each target,
    computes hit flags and combined objective scores, then writes
    ``scored_joined.parquet`` (plus CSV copy) and ``scoring_context.json``.

    Args:
        pair: Pair identifier from ``configs/pairs.yaml``.
        mode: Generation mode (``single_a``, ``single_b``, or ``double``).
        seed: Random seed identifying the run.
        config: Path to pairs.yaml.
        experiment_config: Path to experiment.yaml.
        force: Re-score even if checkpoint CSVs exist.
        prune_cache: Remove stale scorer cache files before scoring.
        parallel_scoring: Parallel scoring mode (``auto``/``on``/``off``).
        molecules: Optional path to molecule file overriding generated output.
    """
    from dualdocking.config import load_experiment, load_pairs
    from dualdocking.scoring.backends import resolve_scoring_backend
    from dualdocking.scoring.objectives import compute_combined, parse_objectives_config

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
    from dualdocking.data.io_utils import read_df, read_molecules, stem_exists

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

    scores_a_path = result_dir / "scores_A.csv"
    scores_b_path = result_dir / "scores_B.csv"

    # --- Target scoring (adaptive: parallel on 2+ GPUs, sequential on 1 GPU) ---
    targets = [
        (target_a_id, scores_a_path, "A"),
        (target_b_id, scores_b_path, "B"),
    ]

    all_scores: dict[str, np.ndarray] = {}

    # Determine whether to use parallel scoring
    n_gpus = _detect_gpu_count()
    pinned_device = os.getenv("SCORER_CUDA_DEVICE") or os.getenv("BOLTZ_CUDA_DEVICE")
    use_parallel = parallel_scoring == "on" or (parallel_scoring == "auto" and n_gpus >= 2 and pinned_device is None)

    if use_parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info("Parallel scoring of two targets on {} GPUs", n_gpus)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {}
            for i, (target_id, scores_path, label) in enumerate(targets):
                gpu_id = str(i % n_gpus)
                f = pool.submit(
                    _score_one_target,
                    target_id,
                    smiles_list,
                    scores_path,
                    force,
                    label,
                    scoring_backend,
                    use_cache=use_cache,
                    device=gpu_id,
                )
                futures[f] = label
            for f in as_completed(futures):
                label, scores = f.result()
                all_scores[label] = scores
    else:
        if parallel_scoring == "auto" and n_gpus < 2:
            logger.info("Single GPU detected — sequential scoring")
        progress = tqdm(targets, desc=f"Scoring [{pair}/{mode}]", unit="target")
        for target_id, scores_path, label in progress:
            progress.set_postfix(target=target_id)
            _, scores = _score_one_target(
                target_id,
                smiles_list,
                scores_path,
                force,
                label,
                scoring_backend,
                use_cache=use_cache,
            )
            all_scores[label] = scores

    scores_a = all_scores["A"]
    scores_b = all_scores["B"]

    # Parse objectives from experiment config
    objectives = parse_objectives_config(exp_cfg)

    # Build joined table (schema: smiles_canonical, affinity_a, affinity_b, is_hit_both)
    from dualdocking.config import load_metrics
    from dualdocking.data.chem_sanitize import canonicalize

    metrics_cfg = load_metrics(PROJECT_ROOT / "configs")
    thresholds = metrics_cfg.get("thresholds", {}).get(pair, {})
    thr_a = thresholds.get("target_a", 0.0)
    thr_b = thresholds.get("target_b", 0.0)

    smiles_canon = [canonicalize(s) or "" for s in smiles_list]
    affinity_a = scores_a
    affinity_b = scores_b
    is_hit_both = (affinity_a <= thr_a) & (affinity_b <= thr_b)

    joined = pd.DataFrame(
        {
            "smiles": smiles_list,
            "smiles_canonical": smiles_canon,
            "affinity_a": affinity_a,
            "affinity_b": affinity_b,
            "score_a": affinity_a,
            "score_b": affinity_b,
            "is_hit_both": is_hit_both,
        }
    )

    # Config-driven objective computation
    primary_col = "combined_min"
    for obj in objectives:
        joined[obj.column_name] = compute_combined(scores_a, scores_b, obj.kind, lower_is_better=True, **obj.params)
        if obj.is_primary:
            primary_col = obj.column_name
    joined = joined.sort_values(primary_col)

    # Drug-likeness filters (analytical only, does not affect RL reward)
    filter_cfg_path = PROJECT_ROOT / "configs" / "filters.yaml"
    if filter_cfg_path.exists():
        from dualdocking.data.filters import compute_filters, load_filter_config

        filter_cfg = load_filter_config(filter_cfg_path)
        filter_results, filter_stats = compute_filters(smiles_list, filter_cfg)
        joined["passes_lipinski"] = [r.passes_lipinski for r in filter_results]
        joined["passes_qed"] = [r.passes_qed for r in filter_results]
        joined["passes_pains"] = [r.passes_pains for r in filter_results]
        joined["passes_sa"] = [r.passes_sa for r in filter_results]
        joined["passes_veber"] = [r.passes_veber for r in filter_results]
        joined["passes_all_filters"] = [r.passes_all for r in filter_results]
        joined["mw"] = [r.mw for r in filter_results]
        joined["logp"] = [r.logp for r in filter_results]
        joined["qed_score"] = [r.qed_score for r in filter_results]
        joined["sa_score"] = [r.sa_score for r in filter_results]
        joined["rotatable_bonds"] = [r.rotatable_bonds for r in filter_results]
        joined["tpsa"] = [r.tpsa for r in filter_results]

        import json as _json_filters

        (result_dir / "filter_stats.json").write_text(_json_filters.dumps(filter_stats.to_dict(), indent=2))
        logger.info("Filter stats -> {}", result_dir / "filter_stats.json")

    from dualdocking.data.io_utils import write_df

    write_df(joined, result_dir / "scored_joined", csv_too=True)

    # Write scoring context metadata
    import json as _json
    from datetime import UTC, datetime

    scoring_meta = scoring_backend.get_scoring_context_info()
    scoring_meta["scorer_engine"] = scorer_engine
    scoring_meta["timestamp"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    (result_dir / "scoring_context.json").write_text(_json.dumps(scoring_meta, indent=2))
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
