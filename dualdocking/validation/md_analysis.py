"""Trajectory analysis and MM-GBSA for MD validation."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from dualdocking.validation.md_config import MDConfig, _run


# ── Trajectory preprocessing ─────────────────────────────────────────
def _preprocess_trajectory(
    tpr_path: Path,
    xtc_path: Path,
    work_dir: Path,
    *,
    cmd_timeout: int = 600,
) -> Path:
    """Fix PBC, center, and fit trajectory for analysis.

    Runs 3 sequential ``gmx trjconv`` calls:
    1. ``-pbc whole`` — fix broken molecules across PBC
    2. ``-center -pbc mol`` — center protein in box
    3. ``-fit rot+trans`` — least-squares fit to backbone

    Returns:
        Path to the fitted trajectory (``md_fit.xtc``).
    """
    xtc_whole = work_dir / "md_whole.xtc"
    xtc_center = work_dir / "md_center.xtc"
    xtc_fit = work_dir / "md_fit.xtc"

    # Step 1: fix PBC
    _run(
        [
            "gmx", "trjconv",
            "-s", str(tpr_path),
            "-f", str(xtc_path),
            "-o", str(xtc_whole),
            "-pbc", "whole",
        ],
        cwd=work_dir,
        input_text="System\n",
        timeout=cmd_timeout,
    )

    # Step 2: center complex
    _run(
        [
            "gmx", "trjconv",
            "-s", str(tpr_path),
            "-f", str(xtc_whole),
            "-o", str(xtc_center),
            "-center", "-pbc", "mol",
        ],
        cwd=work_dir,
        input_text="Protein\nSystem\n",
        timeout=cmd_timeout,
    )
    xtc_whole.unlink(missing_ok=True)

    # Step 3: least-squares fit
    _run(
        [
            "gmx", "trjconv",
            "-s", str(tpr_path),
            "-f", str(xtc_center),
            "-o", str(xtc_fit),
            "-fit", "rot+trans",
        ],
        cwd=work_dir,
        input_text="Backbone\nSystem\n",
        timeout=cmd_timeout,
    )
    xtc_center.unlink(missing_ok=True)

    return xtc_fit


# ── Analysis helpers ─────────────────────────────────────────────────
def _parse_xvg(path: Path, columns: list[str]) -> pd.DataFrame:
    """Parse a GROMACS XVG file into a DataFrame."""
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith(("#", "@")):
            continue
        parts = line.split()
        if len(parts) >= len(columns):
            rows.append([float(x) for x in parts[: len(columns)]])
    return pd.DataFrame(rows, columns=columns)


def _assess_convergence(
    rmsd_series: pd.Series,  # type: ignore[type-arg]
    *,
    alpha: float = 0.05,
    mean_diff_threshold: float = 0.5,
) -> tuple[bool, float]:
    """Assess RMSD convergence by 2-sample KS test on halves.

    Args:
        rmsd_series: RMSD values in Angstrom.
        alpha: Significance level for KS test.
        mean_diff_threshold: Maximum allowed |mean_diff| in Angstrom.

    Returns:
        Tuple of (converged, ks_pvalue).
    """
    from scipy.stats import ks_2samp

    n = len(rmsd_series)
    if n < 10:
        return False, float("nan")

    half = n // 2
    first_half = rmsd_series.iloc[:half].to_numpy()
    second_half = rmsd_series.iloc[half:].to_numpy()

    stat = ks_2samp(first_half, second_half)
    pvalue = float(stat.pvalue)
    mean_diff = abs(float(np.mean(second_half)) - float(np.mean(first_half)))

    converged = pvalue > alpha and mean_diff < mean_diff_threshold
    return converged, pvalue


def _analyze_trajectory(
    work_dir: Path,
    ndx_path: Path,
    xtc_fit: Path,
    out_dir: Path,
    config: MDConfig,
) -> dict:
    """Extract comprehensive MD analysis from preprocessed trajectory.

    Uses ``gmx hbond-legacy`` (GROMACS 2024+) with fallback to ``gmx hbond``.
    Interaction energy code removed — ``energygrps`` is not set in the
    production MDP (incompatible with GPU NB kernels), so ``gmx energy``
    would always fail.

    Returns:
        Dictionary with all analysis metrics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tpr = work_dir / "md.tpr"
    t = config.cmd_timeout

    # ── Backbone RMSD ────────────────────────────────────────────
    bb_rmsd_xvg = work_dir / "rmsd_backbone.xvg"
    _run(
        [
            "gmx", "rms",
            "-s", str(tpr), "-f", str(xtc_fit),
            "-n", str(ndx_path),
            "-o", str(bb_rmsd_xvg), "-tu", "ns",
        ],
        cwd=work_dir,
        input_text="Backbone\nBackbone\n",
        timeout=t,
    )
    bb_rmsd_df = _parse_xvg(bb_rmsd_xvg, columns=["time_ns", "rmsd_nm"])
    bb_rmsd_df["rmsd_A"] = bb_rmsd_df["rmsd_nm"] * 10.0
    bb_rmsd_df[["time_ns", "rmsd_A"]].to_csv(out_dir / "rmsd_backbone.csv", index=False)

    # ── Ligand RMSD ──────────────────────────────────────────────
    lig_rmsd_xvg = work_dir / "rmsd_ligand.xvg"
    _run(
        [
            "gmx", "rms",
            "-s", str(tpr), "-f", str(xtc_fit),
            "-n", str(ndx_path),
            "-o", str(lig_rmsd_xvg), "-tu", "ns",
        ],
        cwd=work_dir,
        input_text="LIG\nLIG\n",
        timeout=t,
    )
    lig_rmsd_df = _parse_xvg(lig_rmsd_xvg, columns=["time_ns", "rmsd_nm"])
    lig_rmsd_df["rmsd_A"] = lig_rmsd_df["rmsd_nm"] * 10.0
    lig_rmsd_df[["time_ns", "rmsd_A"]].to_csv(out_dir / "rmsd_ligand.csv", index=False)

    # ── RMSF ─────────────────────────────────────────────────────
    rmsf_xvg = work_dir / "rmsf.xvg"
    _run(
        [
            "gmx", "rmsf",
            "-s", str(tpr), "-f", str(xtc_fit),
            "-n", str(ndx_path),
            "-o", str(rmsf_xvg), "-res",
        ],
        cwd=work_dir,
        input_text="Backbone\n",
        timeout=t,
    )
    rmsf_df = _parse_xvg(rmsf_xvg, columns=["residue", "rmsf_nm"])
    rmsf_df["rmsf_A"] = rmsf_df["rmsf_nm"] * 10.0
    rmsf_df[["residue", "rmsf_A"]].to_csv(out_dir / "rmsf.csv", index=False)

    # ── Radius of gyration ───────────────────────────────────────
    rg_xvg = work_dir / "gyrate.xvg"
    _run(
        [
            "gmx", "gyrate",
            "-s", str(tpr), "-f", str(xtc_fit),
            "-n", str(ndx_path),
            "-o", str(rg_xvg),
        ],
        cwd=work_dir,
        input_text="Protein\n",
        timeout=t,
    )
    rg_df = _parse_xvg(rg_xvg, columns=["time_ps", "rg_nm"])
    rg_df["rg_A"] = rg_df["rg_nm"] * 10.0
    rg_df[["time_ps", "rg_A"]].to_csv(out_dir / "radius_gyration.csv", index=False)

    # ── Hydrogen bonds (hbond-legacy for GROMACS 2024+, fallback to hbond) ──
    hbond_xvg = work_dir / "hbond.xvg"
    hbond_df: pd.DataFrame
    try:
        _run(
            [
                "gmx", "hbond-legacy",
                "-s", str(tpr), "-f", str(xtc_fit),
                "-n", str(ndx_path),
                "-num", str(hbond_xvg), "-tu", "ns",
            ],
            cwd=work_dir,
            input_text="Protein\nLIG\n",
            timeout=t,
        )
        hbond_df = _parse_xvg(hbond_xvg, columns=["time_ns", "n_hbonds"])
    except RuntimeError:
        logger.debug("gmx hbond-legacy not available, trying gmx hbond")
        try:
            _run(
                [
                    "gmx", "hbond",
                    "-s", str(tpr), "-f", str(xtc_fit),
                    "-n", str(ndx_path),
                    "-num", str(hbond_xvg), "-tu", "ns",
                ],
                cwd=work_dir,
                input_text="Protein\nLIG\n",
                timeout=t,
            )
            hbond_df = _parse_xvg(hbond_xvg, columns=["time_ns", "n_hbonds"])
        except RuntimeError:
            logger.warning("gmx hbond failed (LIG may lack donors/acceptors) — skipping")
            hbond_df = pd.DataFrame(columns=["time_ns", "n_hbonds"])
    hbond_df.to_csv(out_dir / "hbonds.csv", index=False)

    # Copy trajectory + reference structure (needed for visualization)
    if xtc_fit.exists():
        shutil.copy2(xtc_fit, out_dir / "trajectory.xtc")
    tpr = work_dir / "md.tpr"
    if tpr.exists():
        shutil.copy2(tpr, out_dir / "reference.tpr")
    npt_gro = work_dir / "npt.gro"
    if npt_gro.exists():
        shutil.copy2(npt_gro, out_dir / "reference.gro")

    # ── Compute summary metrics ──────────────────────────────────
    # Ligand RMSD stability (last N ns)
    tail = lig_rmsd_df[
        lig_rmsd_df["time_ns"] >= (config.simulation_time_ns - config.stable_window_ns)
    ]
    mean_rmsd = float(tail["rmsd_A"].mean()) if len(tail) > 0 else float("nan")
    max_rmsd = float(tail["rmsd_A"].max()) if len(tail) > 0 else float("nan")
    stable = bool(mean_rmsd < config.rmsd_threshold_a)

    # Backbone RMSD
    bb_mean = float(bb_rmsd_df["rmsd_A"].mean()) if len(bb_rmsd_df) > 0 else float("nan")
    bb_final = float(bb_rmsd_df["rmsd_A"].iloc[-1]) if len(bb_rmsd_df) > 0 else float("nan")

    # RMSF (mean as binding pocket proxy)
    rmsf_bp = float(rmsf_df["rmsf_A"].mean()) if len(rmsf_df) > 0 else float("nan")

    # Radius of gyration
    rg_m = float(rg_df["rg_A"].mean()) if len(rg_df) > 0 else float("nan")
    rg_s = float(rg_df["rg_A"].std()) if len(rg_df) > 0 else float("nan")

    # H-bonds
    mean_hbonds = float(hbond_df["n_hbonds"].mean()) if len(hbond_df) > 0 else float("nan")

    # Convergence assessment on backbone RMSD
    converged, ks_pvalue = _assess_convergence(
        bb_rmsd_df["rmsd_A"],
        alpha=config.convergence_ks_alpha,
    )

    return {
        "mean_rmsd": round(mean_rmsd, 3),
        "max_rmsd": round(max_rmsd, 3),
        "stable": stable,
        "mean_hbonds": round(mean_hbonds, 2),
        "mean_energy": float("nan"),
        "backbone_rmsd_mean": round(bb_mean, 3),
        "backbone_rmsd_final": round(bb_final, 3),
        "rmsf_binding_pocket": round(rmsf_bp, 3),
        "rg_mean": round(rg_m, 3),
        "rg_std": round(rg_s, 3),
        "converged": converged,
        "convergence_ks_pvalue": round(ks_pvalue, 4),
        "coulomb_sr": float("nan"),
        "lj_sr": float("nan"),
    }


# ── MM-GBSA (optional) ──────────────────────────────────────────────
def _run_mmpbsa(
    work_dir: Path,
    ndx_path: Path,
    out_dir: Path,
    config: MDConfig,
) -> float:
    """Run gmx_MMPBSA for MM-GBSA binding free energy estimation.

    Returns:
        Delta G in kcal/mol, or ``float("nan")`` on failure.
    """
    if not shutil.which("gmx_MMPBSA"):
        logger.warning("gmx_MMPBSA not found, skipping MM-GBSA")
        return float("nan")

    # Frame selection: last 20% of trajectory
    total_frames = int(config.simulation_time_ns * 1e3 / (config.dt * config.nstxout_compressed))
    start_frame = int(total_frames * 0.8)
    end_frame = total_frames

    mmpbsa_in = work_dir / "mmpbsa.in"
    mmpbsa_in.write_text(
        f"&general\n  startframe={start_frame}\n  endframe={end_frame}\n  interval=10\n/\n"
        "&gb\n  igb=5\n  saltcon=0.15\n/\n"
    )

    result = _run(
        [
            "gmx_MMPBSA",
            "-O",
            "-i", str(mmpbsa_in),
            "-cs", str(work_dir / "md.tpr"),
            "-ct", str(work_dir / "md.xtc"),
            "-cp", str(work_dir / "topol.top"),
            "-ci", str(ndx_path),
            "-cg", "Protein", "LIG",
        ],
        cwd=work_dir,
        check=False,
        timeout=config.cmd_timeout,
    )
    if result.returncode != 0:
        logger.warning("gmx_MMPBSA failed: {}", (result.stderr or "")[:500])
        return float("nan")

    # Parse FINAL_RESULTS_MMPBSA.dat
    final = work_dir / "FINAL_RESULTS_MMPBSA.dat"
    if not final.exists():
        logger.warning("FINAL_RESULTS_MMPBSA.dat not found")
        return float("nan")

    shutil.copy2(final, out_dir / "mmpbsa.csv")

    dg = float("nan")
    for line in final.read_text().splitlines():
        if "DELTA TOTAL" in line:
            parts = line.split()
            for part in parts:
                try:
                    dg = float(part)
                    break
                except ValueError:
                    continue
            break

    return dg
