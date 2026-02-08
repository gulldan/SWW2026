"""Скоринг сгенерированных молекул по обеим мишеням пары.

Читает молекулы из results/<pair>/<mode>/seed_<seed>/generated.csv, оценивает по
обеим мишеням, записывает scores_A.csv, scores_B.csv, scored_joined.csv.

Поддерживает чекпоинт: если scores_A.csv уже существует, скоринг мишени A
пропускается. При прерывании скоринга отдельных молекул кеш Boltz-2
(cache/{target_id}.json) сохраняется инкрементально.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _detect_gpu_count() -> int:
    """Detect number of CUDA GPUs; returns 1 if detection fails."""
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
    *,
    device: str | None = None,
) -> tuple[str, np.ndarray]:
    """Score molecules against one target, with checkpoint support."""
    from dualdocking.scorer_boltz2 import score_batch

    if not force and scores_path.exists():
        logger.info("Чекпоинт: скоры мишени {} уже существуют ({}), пропускаем", target_id, scores_path)
        saved = pd.read_csv(scores_path)
        return label, saved["score"].to_numpy()

    logger.info("Скоринг {} молекул по мишени {} (device={})", len(smiles_list), target_id, device)
    scores = score_batch(target_id, smiles_list, device=device)

    pd.DataFrame({"smiles": smiles_list, "score": scores}).to_csv(scores_path, index=False)
    logger.info("Скоры мишени {} сохранены: {}", target_id, scores_path)
    return label, scores


@click.command()
@click.option("--pair", required=True, help="ID пары из configs/pairs.yaml")
@click.option("--mode", type=click.Choice(["single_a", "single_b", "double"]), required=True)
@click.option("--seed", default=42, type=int)
@click.option("--config", default="configs/pairs.yaml")
@click.option("--experiment-config", default="configs/experiment.yaml", help="Experiment config for objectives")
@click.option("--force", is_flag=True, default=False, help="Пересчитать даже если scores уже есть")
@click.option("--prune-cache", is_flag=True, default=False, help="Remove stale cache files before scoring")
@click.option(
    "--parallel-scoring",
    type=click.Choice(["auto", "on", "off"]),
    default="auto",
    show_default=True,
    help="Parallel target scoring: 'auto' enables when >=2 GPUs detected, 'on'/'off' force.",
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
) -> None:
    """Оценить молекулы по обеим мишеням."""
    from dualdocking.config import load_experiment, load_pairs
    from dualdocking.objectives import compute_combined, parse_objectives_config
    from dualdocking.scorer_boltz2 import get_scoring_context_info, prune_stale_cache

    if prune_cache:
        n = prune_stale_cache()
        logger.info("Pruned {} stale cache file(s)", n)

    pairs = load_pairs(PROJECT_ROOT / config)["pairs"]
    if pair not in pairs:
        logger.error("Пара '{}' не найдена", pair)
        sys.exit(1)

    pair_cfg = pairs[pair]
    result_dir = PROJECT_ROOT / "results" / pair / mode / f"seed_{seed}"
    mol_path = result_dir / "generated.csv"

    if not mol_path.exists():
        logger.error("Молекулы не найдены: {}", mol_path)
        sys.exit(1)

    df = pd.read_csv(mol_path)
    smiles_list = df["smiles"].tolist()

    target_a_id = pair_cfg["target_a"]["id"]
    target_b_id = pair_cfg["target_b"]["id"]

    scores_a_path = result_dir / "scores_A.csv"
    scores_b_path = result_dir / "scores_B.csv"

    # --- Скоринг мишеней (адаптивно: параллельно на 2+ GPU, последовательно на 1 GPU) ---
    targets = [
        (target_a_id, scores_a_path, "A"),
        (target_b_id, scores_b_path, "B"),
    ]

    all_scores: dict[str, np.ndarray] = {}

    # Determine whether to use parallel scoring
    n_gpus = _detect_gpu_count()
    use_parallel = parallel_scoring == "on" or (
        parallel_scoring == "auto" and n_gpus >= 2 and os.getenv("BOLTZ_CUDA_DEVICE") is None
    )

    if use_parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info("Параллельный скоринг двух мишеней на {} GPU", n_gpus)
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
                    device=gpu_id,
                )
                futures[f] = label
            for f in as_completed(futures):
                label, scores = f.result()
                all_scores[label] = scores
    else:
        if parallel_scoring == "auto" and n_gpus < 2:
            logger.info("Один GPU обнаружен — последовательный скоринг")
        progress = tqdm(targets, desc=f"Скоринг [{pair}/{mode}]", unit="target")
        for target_id, scores_path, label in progress:
            progress.set_postfix(target=target_id)
            _, scores = _score_one_target(
                target_id,
                smiles_list,
                scores_path,
                force,
                label,
            )
            all_scores[label] = scores

    scores_a = all_scores["A"]
    scores_b = all_scores["B"]

    # Parse objectives from experiment config
    exp_cfg = load_experiment(PROJECT_ROOT / experiment_config)
    objectives = parse_objectives_config(exp_cfg)

    # Сохранить объединённую таблицу (schema contract: smiles_canonical, affinity_a, affinity_b, is_hit_both)
    from dualdocking.chem_sanitize import canonicalize
    from dualdocking.config import load_metrics

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
    joined.to_csv(result_dir / "scored_joined.csv", index=False)

    # Write scoring context metadata
    import json as _json
    from datetime import UTC, datetime

    scoring_meta = get_scoring_context_info()
    scoring_meta["timestamp"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    (result_dir / "scoring_context.json").write_text(_json.dumps(scoring_meta, indent=2))
    logger.info("Scoring context -> {}", result_dir / "scoring_context.json")

    logger.info("Скоры сохранены в {}", result_dir)
    logger.info(
        "Статистика скоров - A: mean={:.2f} std={:.2f} | B: mean={:.2f} std={:.2f}",
        np.nanmean(scores_a),
        np.nanstd(scores_a),
        np.nanmean(scores_b),
        np.nanstd(scores_b),
    )


if __name__ == "__main__":
    main()
