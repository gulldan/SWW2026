"""Proxy vs final correlation analysis (SCI-07 / KVA-271).

Computes Spearman rho and Pearson r between UniDock docking reward (proxy,
used during RL training) and Boltz-2 affinity (final evaluation scorer)
per pair/mode.

Molecules are joined by canonical SMILES.  Only training-epoch molecules
that also appear in the final scored set are used.

Outputs structured dicts and flat rows suitable for CSV/Parquet and JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as sp_stats

from dualdocking.analysis.metrics.common import load_scored_affinity

_N_BOOTSTRAP = 5_000


def _is_constant(arr: np.ndarray) -> bool:
    """Return ``True`` when all values in *arr* are identical."""
    if arr.size == 0:
        return True
    return bool(np.all(arr == arr[0]))


@dataclass(frozen=True)
class CorrelationConfig:
    """Configuration for proxy-final correlation analysis.

    Attributes:
        project_root: Absolute path to project root.
        pairs: Pair configuration dict (pair_id -> config).
        modes: Generation modes.
        seeds: Random seeds.
        results_root: Root results directory.
        n_bootstrap: Number of bootstrap resamples for CIs.
    """

    project_root: Path
    pairs: dict[str, dict]
    modes: list[str]
    seeds: list[int]
    results_root: Path = Path("output/results")
    n_bootstrap: int = _N_BOOTSTRAP


@dataclass(frozen=True)
class CorrelationResult:
    """Result of proxy-final correlation analysis.

    Attributes:
        json_out: Structured results dict (pairs, summary).
        rows: Flat list of dicts for CSV/Parquet output.
    """

    json_out: dict[str, Any]
    rows: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bootstrap_correlation(
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
        n_bootstrap: Number of bootstrap resamples.
        rng_seed: Seed for the bootstrap RNG.

    Returns:
        Tuple of ``(observed_correlation, ci_lower, ci_upper)``.
    """
    if n_bootstrap is None:
        n_bootstrap = _N_BOOTSTRAP

    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")

    corr_fn = sp_stats.spearmanr if method == "spearman" else sp_stats.pearsonr
    observed = float("nan") if _is_constant(x) or _is_constant(y) else float(corr_fn(x, y).statistic)

    rng = np.random.default_rng(rng_seed)
    boot = np.empty(n_bootstrap, dtype=float)
    n = len(x)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        x_boot = x[idx]
        y_boot = y[idx]
        if _is_constant(x_boot) or _is_constant(y_boot):
            boot[i] = float("nan")
            continue
        boot[i] = float(corr_fn(x_boot, y_boot).statistic)

    ci_lo, ci_hi = np.nanpercentile(boot, [2.5, 97.5])
    return observed, float(ci_lo), float(ci_hi)


def load_boltz_scores(scored_csv: Path) -> pd.DataFrame:
    """Load Boltz-2 affinities from ``scored_joined``.

    Args:
        scored_csv: Stem path (without extension) to ``scored_joined``.

    Returns:
        DataFrame with columns ``smiles_canonical``, ``affinity_a``, ``affinity_b``.

    Raises:
        ValueError: If ``smiles_canonical`` is missing and cannot be derived.
    """
    df = load_scored_affinity(scored_csv)
    return df[["smiles_canonical", "affinity_a", "affinity_b"]]


def load_docking_scores(run_dir: Path) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_correlation(config: CorrelationConfig) -> CorrelationResult:
    """Compute proxy (docking) vs final (Boltz-2) correlation per pair/mode.

    Joins FREED++ training docking scores with Boltz-2 affinities by
    canonical SMILES, then computes Spearman rho and Pearson r with
    bootstrap 95% CIs.

    Args:
        config: Correlation analysis configuration.

    Returns:
        CorrelationResult with structured JSON and flat CSV data.
    """
    from dualdocking.data.io_utils import stem_exists

    results_root = config.results_root if config.results_root.is_absolute() else config.project_root / config.results_root

    rows: list[dict[str, Any]] = []
    json_out: dict[str, Any] = {"pairs": {}}

    for pair_id in config.pairs:
        json_out["pairs"][pair_id] = {}
        for mode in config.modes:
            seed_results: list[dict[str, Any]] = []
            total_boltz = 0
            total_docking = 0
            total_joined = 0

            for seed in config.seeds:
                run_dir = results_root / pair_id / mode / f"seed_{seed}"
                scored_stem = run_dir / "scored_joined"
                if not stem_exists(scored_stem):
                    continue

                boltz_df = load_boltz_scores(scored_stem)
                docking_df = load_docking_scores(run_dir)

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

            pair_mode_stats: dict[str, Any] = {
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

                spearman_rho, sp_ci_lo, sp_ci_hi = bootstrap_correlation(x, y, "spearman", n_bootstrap=config.n_bootstrap)
                pearson_r, pe_ci_lo, pe_ci_hi = bootstrap_correlation(x, y, "pearson", n_bootstrap=config.n_bootstrap)

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

    return CorrelationResult(json_out=json_out, rows=rows)
