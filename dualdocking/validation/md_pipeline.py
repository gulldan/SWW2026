"""MD validation orchestration: molecule selection, single-molecule pipeline, and summary."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from loguru import logger

from dualdocking.validation.md_analysis import (
    _analyze_trajectory,
    _preprocess_trajectory,
    _run_mmpbsa,
)
from dualdocking.validation.md_config import MDConfig, MDResult
from dualdocking.validation.md_setup import _parameterize_ligand, _setup_system


# ── Top-N molecule selection ─────────────────────────────────────────
def select_top_molecules(
    pair: str,
    mode: str,
    top_n: int,
    seeds: list[int],
    results_dir: Path,
) -> pd.DataFrame:
    """Select top-N molecules by combined Boltz-2 affinity across seeds.

    Returns:
        DataFrame with ``smiles``, ``combined_min``, ``seed`` columns,
        deduplicated by SMILES and sorted by best ``combined_min``.
    """
    from dualdocking.data.io_utils import read_df, stem_exists

    frames = []
    for seed in seeds:
        stem = results_dir / pair / mode / f"seed_{seed}" / "scored_joined"
        if not stem_exists(stem):
            continue
        df = read_df(stem)
        df["seed"] = seed
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)

    # Deduplicate by SMILES, keep best combined score
    score_col = "combined_min" if "combined_min" in all_df.columns else "affinity_a"
    best = all_df.sort_values(score_col).drop_duplicates(subset="smiles", keep="first")
    return best.head(top_n).reset_index(drop=True)


# ── Single molecule pipeline ────────────────────────────────────────
def run_single_molecule_md(
    mol_id: int,
    smiles: str,
    pair: str,
    mode: str,
    target_id: str,
    target_label: str,
    protein_pdb: Path,
    out_base: Path,
    config: MDConfig,
) -> MDResult | None:
    """Run the full MD pipeline for one molecule x one target.

    Returns:
        MDResult on success, or ``None`` on failure.
    """
    out_dir = out_base / f"target_{target_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already completed
    summary_file = out_dir / "md_result.json"
    if summary_file.exists():
        logger.info("  Skipping mol_{}/target_{} (already done)", mol_id, target_label)
        data = json.loads(summary_file.read_text())
        return MDResult(**{k: data[k] for k in MDResult.__dataclass_fields__})

    with tempfile.TemporaryDirectory(prefix=f"md_{pair}_{mode}_mol{mol_id}_{target_label}_") as tmp:
        work = Path(tmp)
        try:
            logger.info("  Parameterizing ligand...")
            acpype_dir = _parameterize_ligand(smiles, work, cmd_timeout=config.cmd_timeout)

            logger.info("  Setting up GROMACS system...")
            _setup_system(protein_pdb, acpype_dir, work, config)

            logger.info("  Preprocessing trajectory...")
            ndx = work / "index.ndx"
            xtc_fit = _preprocess_trajectory(
                work / "md.tpr", work / "md.xtc", work,
                cmd_timeout=config.cmd_timeout,
            )

            logger.info("  Analyzing trajectory...")
            analysis = _analyze_trajectory(work, ndx, xtc_fit, out_dir, config)

            mmpbsa_dg = float("nan")
            if config.mmpbsa:
                logger.info("  Running MM-GBSA...")
                mmpbsa_dg = _run_mmpbsa(work, ndx, out_dir, config)
            analysis["mmpbsa_dg"] = mmpbsa_dg

        except Exception:
            logger.exception("  MD failed for mol_{}/target_{}", mol_id, target_label)
            return None

    result = MDResult(
        pair=pair,
        mode=mode,
        mol_id=mol_id,
        target=target_id,
        target_label=target_label,
        smiles=smiles,
        **analysis,
    )
    summary_file.write_text(json.dumps(asdict(result), indent=2))
    return result


# ── GROMACS command generation (for testing) ─────────────────────────
def build_gromacs_commands() -> list[list[str]]:
    """Build the sequence of GROMACS commands for the MD pipeline.

    Pure function for unit testing — does not execute commands.
    Modern protocol: amber14sb, 1.0 nm cutoffs, C-rescale NPT barostat,
    Protein_LIG / Water_and_ions tc-grps, hbond-legacy.

    Returns:
        List of command argument lists.
    """
    w = Path("/work")
    ndx = str(w / "index.ndx")
    commands: list[list[str]] = []

    # ── Setup (14 commands) ──────────────────────────────────────
    # 1. pdb2gmx — amber14sb (Maier et al. 2015)
    commands.append([
        "gmx", "pdb2gmx",
        "-f", str(w / "protein.pdb"),
        "-o", str(w / "protein.gro"),
        "-p", str(w / "topol.top"),
        "-ignh", "-ff", "amber14sb", "-water", "tip3p",
    ])
    # 2. editconf
    commands.append([
        "gmx", "editconf",
        "-f", str(w / "complex.gro"),
        "-o", str(w / "box.gro"),
        "-c", "-d", "1.2", "-bt", "dodecahedron",
    ])
    # 3. solvate
    commands.append([
        "gmx", "solvate",
        "-cp", str(w / "box.gro"),
        "-cs", "spc216.gro",
        "-o", str(w / "solv.gro"),
        "-p", str(w / "topol.top"),
    ])
    # 4. grompp ions
    commands.append([
        "gmx", "grompp",
        "-f", str(w / "ions.mdp"),
        "-c", str(w / "solv.gro"),
        "-p", str(w / "topol.top"),
        "-o", str(w / "ions.tpr"),
        "-maxwarn", "0",
    ])
    # 5. genion
    commands.append([
        "gmx", "genion",
        "-s", str(w / "ions.tpr"),
        "-o", str(w / "solv_ions.gro"),
        "-p", str(w / "topol.top"),
        "-pname", "NA", "-nname", "CL",
        "-neutral", "-conc", "0.15",
    ])
    # 6. grompp EM
    commands.append([
        "gmx", "grompp",
        "-f", str(w / "em.mdp"),
        "-c", str(w / "solv_ions.gro"),
        "-p", str(w / "topol.top"),
        "-o", str(w / "em.tpr"),
        "-maxwarn", "0",
    ])
    # 7. mdrun EM
    commands.append(["gmx", "mdrun", "-deffnm", str(w / "em"), "-ntmpi", "1"])
    # 8. make_ndx
    commands.append([
        "gmx", "make_ndx",
        "-f", str(w / "em.tpr"),
        "-o", ndx,
    ])
    # 9. grompp NVT
    commands.append([
        "gmx", "grompp",
        "-f", str(w / "nvt.mdp"),
        "-c", str(w / "em.gro"),
        "-r", str(w / "em.gro"),
        "-p", str(w / "topol.top"),
        "-n", ndx,
        "-o", str(w / "nvt.tpr"),
        "-maxwarn", "1",
    ])
    # 10. mdrun NVT
    commands.append(["gmx", "mdrun", "-deffnm", str(w / "nvt"), "-ntmpi", "1"])
    # 11. grompp NPT
    commands.append([
        "gmx", "grompp",
        "-f", str(w / "npt.mdp"),
        "-c", str(w / "nvt.gro"),
        "-r", str(w / "nvt.gro"),
        "-p", str(w / "topol.top"),
        "-n", ndx,
        "-t", str(w / "nvt.cpt"),
        "-o", str(w / "npt.tpr"),
        "-maxwarn", "1",
    ])
    # 12. mdrun NPT
    commands.append(["gmx", "mdrun", "-deffnm", str(w / "npt"), "-ntmpi", "1"])
    # 13. grompp prod
    commands.append([
        "gmx", "grompp",
        "-f", str(w / "md.mdp"),
        "-c", str(w / "npt.gro"),
        "-t", str(w / "npt.cpt"),
        "-p", str(w / "topol.top"),
        "-n", ndx,
        "-o", str(w / "md.tpr"),
        "-maxwarn", "1",
    ])
    # 14. mdrun prod
    commands.append(["gmx", "mdrun", "-deffnm", str(w / "md"), "-ntmpi", "1"])

    # ── Preprocessing (3 commands) ───────────────────────────────
    # 15. trjconv -pbc whole
    commands.append([
        "gmx", "trjconv",
        "-s", str(w / "md.tpr"),
        "-f", str(w / "md.xtc"),
        "-o", str(w / "md_whole.xtc"),
        "-pbc", "whole",
    ])
    # 16. trjconv -center -pbc mol
    commands.append([
        "gmx", "trjconv",
        "-s", str(w / "md.tpr"),
        "-f", str(w / "md_whole.xtc"),
        "-o", str(w / "md_center.xtc"),
        "-center", "-pbc", "mol",
    ])
    # 17. trjconv -fit rot+trans
    commands.append([
        "gmx", "trjconv",
        "-s", str(w / "md.tpr"),
        "-f", str(w / "md_center.xtc"),
        "-o", str(w / "md_fit.xtc"),
        "-fit", "rot+trans",
    ])

    # ── Analysis (5 commands) ────────────────────────────────────
    # 18. backbone RMSD
    commands.append([
        "gmx", "rms",
        "-s", str(w / "md.tpr"),
        "-f", str(w / "md_fit.xtc"),
        "-n", ndx,
        "-o", str(w / "rmsd_backbone.xvg"),
        "-tu", "ns",
    ])
    # 19. ligand RMSD
    commands.append([
        "gmx", "rms",
        "-s", str(w / "md.tpr"),
        "-f", str(w / "md_fit.xtc"),
        "-n", ndx,
        "-o", str(w / "rmsd_ligand.xvg"),
        "-tu", "ns",
    ])
    # 20. RMSF
    commands.append([
        "gmx", "rmsf",
        "-s", str(w / "md.tpr"),
        "-f", str(w / "md_fit.xtc"),
        "-n", ndx,
        "-o", str(w / "rmsf.xvg"),
        "-res",
    ])
    # 21. Rg
    commands.append([
        "gmx", "gyrate",
        "-s", str(w / "md.tpr"),
        "-f", str(w / "md_fit.xtc"),
        "-n", ndx,
        "-o", str(w / "gyrate.xvg"),
    ])
    # 22. hbond-legacy
    commands.append([
        "gmx", "hbond-legacy",
        "-s", str(w / "md.tpr"),
        "-f", str(w / "md_fit.xtc"),
        "-n", ndx,
        "-num", str(w / "hbond.xvg"),
        "-tu", "ns",
    ])

    return commands


# ── Main entry point ─────────────────────────────────────────────────
def run_md_validation(
    config: MDConfig,
    pair_cfg: dict,
) -> list[MDResult]:
    """Run MD validation for a protein pair.

    Args:
        config: MD configuration.
        pair_cfg: Pair configuration dict from ``pairs.yaml`` (contains
            ``target_a`` and ``target_b`` with ``id`` and ``pdb`` fields).

    Returns:
        List of MDResult for each successful molecule x target run.
    """
    target_a_id = pair_cfg["target_a"]["id"]
    target_b_id = pair_cfg["target_b"]["id"]

    targets_dir = config.targets_dir / config.pair_id
    pdb_a = targets_dir / "A" / f"{pair_cfg['target_a']['pdb_id'].lower()}.pdb"
    pdb_b = targets_dir / "B" / f"{pair_cfg['target_b']['pdb_id'].lower()}.pdb"

    for pdb, tid in [(pdb_a, target_a_id), (pdb_b, target_b_id)]:
        if not pdb.exists():
            msg = f"Protein PDB not found for {tid}: {pdb}"
            raise FileNotFoundError(msg)

    all_results: list[MDResult] = []
    md_base = config.output_dir / config.pair_id

    for m in config.modes:
        logger.info("=== Mode: {} ===", m)
        top_df = select_top_molecules(
            config.pair_id, m, config.top_n, config.seeds, config.results_dir
        )
        if top_df.empty:
            logger.warning("No scored molecules found for {}/{}", config.pair_id, m)
            continue

        logger.info("Selected {} molecules for MD", len(top_df))

        for idx, row in top_df.iterrows():
            mol_idx = int(str(idx))
            mol_out = md_base / m / f"mol_{mol_idx}"
            logger.info("Processing mol_{}: {:.60s}...", mol_idx, row["smiles"])

            for target_id, target_label, pdb in [
                (target_a_id, "A", pdb_a),
                (target_b_id, "B", pdb_b),
            ]:
                result = run_single_molecule_md(
                    mol_id=mol_idx,
                    smiles=row["smiles"],
                    pair=config.pair_id,
                    mode=m,
                    target_id=target_id,
                    target_label=target_label,
                    protein_pdb=pdb,
                    out_base=mol_out,
                    config=config,
                )
                if result:
                    all_results.append(result)

    return all_results


def write_summary(results: list[MDResult], summary_path: Path) -> None:
    """Write or append MD results to a parquet summary file."""
    if not results:
        return

    summary_df = pd.DataFrame([asdict(r) for r in results])

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if summary_path.exists():
        existing = pd.read_parquet(summary_path)
        summary_df = pd.concat([existing, summary_df], ignore_index=True)
        summary_df = summary_df.drop_duplicates(
            subset=["pair", "mode", "mol_id", "target"], keep="last"
        )

    summary_df.to_parquet(summary_path, index=False)
    logger.info("MD summary written: {} ({} rows)", summary_path, len(summary_df))
