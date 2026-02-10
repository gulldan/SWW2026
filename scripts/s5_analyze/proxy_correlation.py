"""Proxy vs final correlation analysis (SCI-07 / KVA-271).

Computes Spearman rho and Pearson r between UniDock docking reward (proxy,
used during RL training) and Boltz-2 affinity (final evaluation scorer)
per pair/mode.

Data sources:
  - Boltz-2 affinities: ``results/{pair}/{mode}/seed_{seed}/scored_joined.csv``
    columns: ``affinity_a``, ``affinity_b``
  - UniDock docking scores: ``results/{pair}/{mode}/seed_{seed}/
    freedpp_workspace/output/run/mols/train_*.csv``
    column: ``DockingScoreProperty`` (single mode), or
    ``docking_a``/``docking_b`` (double)

Molecules are joined by canonical SMILES.  Only training-epoch molecules
that also appear in the final scored set are used.

Outputs:
  - ``report/data/proxy_final_corr.csv``  (flat table)
  - ``report/data/proxy_final_corr.json`` (structured)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_N_BOOTSTRAP = 5_000


def _bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
    n_bootstrap: int | None = None,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Compute correlation with 95% bootstrap confidence interval.

    Args:
        x: First variable array.
        y: Second variable array (same length as *x*).
        method: ``"spearman"`` or ``"pearson"``.
        n_bootstrap: Number of bootstrap resamples (defaults to module-level).
        rng_seed: Seed for the bootstrap RNG.

    Returns:
        Tuple of ``(observed_correlation, ci_lower, ci_upper)``.
    """
    if n_bootstrap is None:
        n_bootstrap = _N_BOOTSTRAP

    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")

    corr_fn = sp_stats.spearmanr if method == "spearman" else sp_stats.pearsonr
    observed = float(corr_fn(x, y).statistic)

    rng = np.random.default_rng(rng_seed)
    boot = np.empty(n_bootstrap, dtype=float)
    n = len(x)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(corr_fn(x[idx], y[idx]).statistic)

    ci_lo, ci_hi = np.nanpercentile(boot, [2.5, 97.5])
    return observed, float(ci_lo), float(ci_hi)


def _load_boltz_scores(scored_csv: Path) -> pd.DataFrame:
    """Load Boltz-2 affinities from ``scored_joined.csv``.

    Args:
        scored_csv: Path to the scored CSV file.

    Returns:
        DataFrame with columns ``smiles_canonical``, ``affinity_a``, ``affinity_b``.

    Raises:
        ValueError: If ``smiles_canonical`` is missing and cannot be derived.
    """
    df = pd.read_csv(scored_csv)
    if "affinity_a" not in df.columns and "score_a" in df.columns:
        df = df.rename(columns={"score_a": "affinity_a"})
    if "affinity_b" not in df.columns and "score_b" in df.columns:
        df = df.rename(columns={"score_b": "affinity_b"})
    if "smiles_canonical" not in df.columns:
        if "smiles" in df.columns:
            from dualdocking.data.chem_sanitize import canonicalize

            df["smiles_canonical"] = [canonicalize(s) or "" for s in df["smiles"].astype(str).tolist()]
        else:
            raise ValueError(f"smiles_canonical missing in {scored_csv}")
    df = df[df["smiles_canonical"].astype(str).str.len() > 0]
    return df[["smiles_canonical", "affinity_a", "affinity_b"]].drop_duplicates(subset=["smiles_canonical"])


def _load_docking_scores(run_dir: Path) -> pd.DataFrame:
    """Load docking scores from FREED++ training CSVs.

    Searches ``freedpp_workspace/output/run/mols/train_*.csv`` for
    docking score columns.

    Args:
        run_dir: Run directory containing the ``freedpp_workspace`` subdirectory.

    Returns:
        DataFrame with columns ``smiles_canonical``, ``docking_a``,
        and optionally ``docking_b``.  Empty DataFrame if no data found.
    """
    from dualdocking.data.chem_sanitize import canonicalize

    mols_dir = run_dir / "freedpp_workspace" / "output" / "run" / "mols"
    if not mols_dir.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(mols_dir.glob("train_*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            logger.warning("Failed to read {}", csv_path)
            continue

        smi_col = "Smiles" if "Smiles" in df.columns else "smiles" if "smiles" in df.columns else None
        if smi_col is None:
            continue

        row: dict[str, list] = {"smiles_raw": df[smi_col].tolist()}

        # Single mode: DockingScoreProperty is for the primary target
        if "DockingScoreProperty" in df.columns:
            row["docking_a"] = df["DockingScoreProperty"].tolist()
        # Double mode: may have separate columns
        if "docking_a" in df.columns:
            row["docking_a"] = df["docking_a"].tolist()
        if "docking_b" in df.columns:
            row["docking_b"] = df["docking_b"].tolist()

        if "docking_a" not in row:
            continue

        chunk = pd.DataFrame(row)
        frames.append(chunk)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result["smiles_canonical"] = [canonicalize(s) or "" for s in result["smiles_raw"].astype(str).tolist()]
    result = result[result["smiles_canonical"].str.len() > 0]
    # Keep last occurrence (later epochs are more relevant)
    result = result.drop_duplicates(subset=["smiles_canonical"], keep="last")
    return result


@click.command()
@click.option("--results-root", default="output/results")
@click.option("--pairs-config", default="configs/pairs.yaml")
@click.option("--experiment-config", default="configs/experiment.yaml")
@click.option("--output-csv", default="report/data/proxy_final_corr.csv")
@click.option("--output-json", default="report/data/proxy_final_corr.json")
def main(
    results_root: str,
    pairs_config: str,
    experiment_config: str,
    output_csv: str,
    output_json: str,
) -> None:
    """Compute proxy (docking) vs final (Boltz-2) correlation per pair/mode.

    Joins FREED++ training docking scores with Boltz-2 affinities by
    canonical SMILES, then computes Spearman rho and Pearson r with
    bootstrap 95% CIs.

    Args:
        results_root: Root directory with per-pair results.
        pairs_config: Path to pairs.yaml.
        experiment_config: Path to experiment.yaml.
        output_csv: Output CSV path relative to project root.
        output_json: Output JSON path relative to project root.
    """
    from dualdocking.config import load_experiment, load_pairs

    pairs = load_pairs(PROJECT_ROOT / pairs_config)["pairs"]
    exp_cfg = load_experiment(PROJECT_ROOT / experiment_config)

    modes = exp_cfg.get("modes", ["single_a", "single_b", "double"])
    seeds = exp_cfg.get("seeds", [42, 137, 256, 512, 1024])
    results_root_path = PROJECT_ROOT / results_root

    rows: list[dict] = []
    json_out: dict = {"pairs": {}}

    for pair_id in pairs:
        json_out["pairs"][pair_id] = {}
        for mode in modes:
            seed_results: list[dict] = []
            total_boltz = 0
            total_docking = 0
            total_joined = 0

            for seed in seeds:
                run_dir = results_root_path / pair_id / mode / f"seed_{seed}"
                scored_path = run_dir / "scored_joined.csv"
                if not scored_path.exists():
                    continue

                boltz_df = _load_boltz_scores(scored_path)
                docking_df = _load_docking_scores(run_dir)

                total_boltz += len(boltz_df)

                if docking_df.empty:
                    continue

                total_docking += len(docking_df)

                # Join by canonical SMILES
                merged = boltz_df.merge(docking_df, on="smiles_canonical", how="inner")
                total_joined += len(merged)

                if len(merged) < 3:
                    continue

                seed_results.append(
                    {
                        "seed": seed,
                        "n_boltz": len(boltz_df),
                        "n_docking": len(docking_df),
                        "n_joined": len(merged),
                        "merged": merged,
                    }
                )

            if not seed_results:
                if total_boltz > 0:
                    logger.warning("{}/{}: {} Boltz scores but no docking data", pair_id, mode, total_boltz)
                continue

            # Aggregate across seeds
            all_merged = pd.concat([r["merged"] for r in seed_results], ignore_index=True)
            all_merged = all_merged.drop_duplicates(subset=["smiles_canonical"], keep="first")

            pair_mode_stats: dict = {
                "n_seeds_with_data": len(seed_results),
                "n_boltz_total": total_boltz,
                "n_docking_total": total_docking,
                "n_joined_total": total_joined,
                "n_unique_joined": len(all_merged),
                "targets": {},
            }

            # Correlations per target
            for target_label, aff_col, dock_col in [
                ("target_a", "affinity_a", "docking_a"),
                ("target_b", "affinity_b", "docking_b"),
            ]:
                if dock_col not in all_merged.columns:
                    continue

                valid = all_merged[[aff_col, dock_col]].dropna()
                x = valid[dock_col].to_numpy()
                y = valid[aff_col].to_numpy()

                if len(x) < 3:
                    continue

                spearman_rho, sp_ci_lo, sp_ci_hi = _bootstrap_correlation(x, y, "spearman")
                pearson_r, pe_ci_lo, pe_ci_hi = _bootstrap_correlation(x, y, "pearson")

                target_stats = {
                    "n": len(x),
                    "spearman_rho": spearman_rho,
                    "spearman_ci": [sp_ci_lo, sp_ci_hi],
                    "pearson_r": pearson_r,
                    "pearson_ci": [pe_ci_lo, pe_ci_hi],
                }
                pair_mode_stats["targets"][target_label] = target_stats

                rows.append(
                    {
                        "pair": pair_id,
                        "mode": mode,
                        "target": target_label,
                        "n": len(x),
                        "spearman_rho": spearman_rho,
                        "spearman_ci_lo": sp_ci_lo,
                        "spearman_ci_hi": sp_ci_hi,
                        "pearson_r": pearson_r,
                        "pearson_ci_lo": pe_ci_lo,
                        "pearson_ci_hi": pe_ci_hi,
                        "n_boltz": total_boltz,
                        "n_docking": total_docking,
                        "n_joined": total_joined,
                        "join_rate": total_joined / total_boltz if total_boltz > 0 else 0.0,
                    }
                )

            json_out["pairs"][pair_id][mode] = pair_mode_stats

    # Suitability summary
    all_rhos = [r["spearman_rho"] for r in rows if np.isfinite(r.get("spearman_rho", float("nan")))]
    json_out["summary"] = {
        "n_pair_mode_target_combos": len(rows),
        "median_spearman_rho": float(np.median(all_rhos)) if all_rhos else float("nan"),
        "min_spearman_rho": float(np.min(all_rhos)) if all_rhos else float("nan"),
        "max_spearman_rho": float(np.max(all_rhos)) if all_rhos else float("nan"),
        "proxy_suitable": bool(all_rhos and float(np.median(all_rhos)) > 0.3),
    }

    # Write outputs
    csv_path = PROJECT_ROOT / output_csv
    json_path = PROJECT_ROOT / output_json
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info("CSV -> {}", csv_path)
    else:
        logger.warning("No proxy-final pairs found (no docking data available)")

    json_path.write_text(json.dumps(json_out, indent=2))
    logger.info("JSON -> {}", json_path)

    if all_rhos:
        logger.info(
            "Median Spearman ρ = {:.3f} [{:.3f}, {:.3f}], proxy suitable: {}",
            float(np.median(all_rhos)),
            float(np.min(all_rhos)),
            float(np.max(all_rhos)),
            json_out["summary"]["proxy_suitable"],
        )


if __name__ == "__main__":
    main()
