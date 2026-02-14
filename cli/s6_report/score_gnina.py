"""Score generated molecules with GNINA CNN-based scoring.

Provides a third independent scorer alongside Boltz-2 and UniDock.
Reads scored molecules from ``scored_joined.parquet``, re-scores each
molecule against both targets with GNINA, and writes per-target score
Parquet files.

GNINA (McNutt et al., J Cheminform 2021) uses a convolutional neural
network trained on binding poses to predict binding affinity.  It
provides three scores:

- ``gnina_score``: Vina-like empirical score (kcal/mol)
- ``gnina_affinity``: CNN-predicted binding affinity (pK units)
- ``cnn_score``: CNN confidence that the pose is a true binder (0-1)

Prerequisites:
    GNINA binary at ``output/bin/gnina`` (downloaded by Stage 0 fetch).

Outputs::

    output/results/{pair}/{mode}/seed_{seed}/gnina_scores_A.parquet
    output/results/{pair}/{mode}/seed_{seed}/gnina_scores_B.parquet
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

GNINA_TIMEOUT = 120  # seconds per molecule
MAX_ATOMS = 150  # skip very large ligands


def _find_gnina() -> Path:
    """Locate the GNINA binary.

    Returns:
        Path to the GNINA executable.

    Raises:
        FileNotFoundError: If no GNINA binary can be found.
    """
    candidates = [
        PROJECT_ROOT / "output" / "bin" / "gnina",
        Path("/usr/local/bin/gnina"),
    ]
    for p in candidates:
        if p.exists() and p.stat().st_mode & 0o111:
            return p

    import shutil

    which = shutil.which("gnina")
    if which:
        return Path(which)

    msg = "GNINA binary not found. Run cli/s0_fetch/fetch_all.py first."
    raise FileNotFoundError(msg)


def _prepare_ligand_sdf(smiles: str, work_dir: Path) -> Path | None:
    """Generate a 3D SDF file from SMILES for GNINA input.

    Args:
        smiles: SMILES string.
        work_dir: Temporary working directory.

    Returns:
        Path to the SDF file, or ``None`` if generation fails.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if mol.GetNumHeavyAtoms() > MAX_ATOMS:
            logger.debug("Skipping large ligand ({} heavy atoms): {:.60s}", mol.GetNumHeavyAtoms(), smiles)
            return None

        mol = Chem.AddHs(mol)
        status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if status != 0:
            # Fallback: random coordinates
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3(), randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

        sdf_path = work_dir / "ligand.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()
        return sdf_path

    except Exception:
        logger.debug("Failed to prepare SDF for {:.60s}", smiles)
        return None


def _score_one_gnina(
    gnina_bin: Path,
    receptor_pdb: Path,
    ligand_sdf: Path,
    *,
    autobox_ligand: Path | None = None,
    center: tuple[float, float, float] | None = None,
    box_size: tuple[float, float, float] | None = None,
) -> dict[str, float]:
    """Run GNINA on a single ligand and parse scores.

    Args:
        gnina_bin: Path to the GNINA binary.
        receptor_pdb: Path to the receptor PDB.
        ligand_sdf: Path to the ligand SDF.
        autobox_ligand: Optional PDB with reference ligand for autobox.
        center: Optional box center ``(x, y, z)``.
        box_size: Optional box size ``(sx, sy, sz)``.

    Returns:
        Dictionary with ``gnina_score``, ``gnina_affinity``, ``cnn_score``.
        All ``NaN`` on failure.
    """
    nan_result = {"gnina_score": float("nan"), "gnina_affinity": float("nan"), "cnn_score": float("nan")}

    cmd = [
        str(gnina_bin),
        "-r", str(receptor_pdb),
        "-l", str(ligand_sdf),
        "--minimize",
        "--num_modes", "1",
        "--no_gpu",  # CPU for reproducibility; use --device for GPU
    ]

    if autobox_ligand and autobox_ligand.exists():
        cmd.extend(["--autobox_ligand", str(autobox_ligand)])
    elif center and box_size:
        cmd.extend([
            "--center_x", str(center[0]),
            "--center_y", str(center[1]),
            "--center_z", str(center[2]),
            "--size_x", str(box_size[0]),
            "--size_y", str(box_size[1]),
            "--size_z", str(box_size[2]),
        ])
    else:
        cmd.append("--autobox_ligand")
        cmd.append(str(receptor_pdb))  # whole-protein fallback

    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=GNINA_TIMEOUT,
            check=False,
        )
        if result.returncode != 0:
            logger.debug("GNINA failed: {}", result.stderr[:200])
            return nan_result

        # Parse GNINA output (SDF with scored properties)
        return _parse_gnina_output(result.stdout)

    except subprocess.TimeoutExpired:
        logger.debug("GNINA timed out")
        return nan_result
    except Exception:
        logger.debug("GNINA exception")
        return nan_result


def _parse_gnina_output(stdout: str) -> dict[str, float]:
    """Parse GNINA stdout for the top-ranked pose scores.

    GNINA prints a table with columns:
    mode | affinity | intramol | CNN_score | CNN_affinity

    Args:
        stdout: GNINA standard output.

    Returns:
        Dictionary with ``gnina_score``, ``gnina_affinity``, ``cnn_score``.
    """
    nan_result = {"gnina_score": float("nan"), "gnina_affinity": float("nan"), "cnn_score": float("nan")}

    for raw_line in stdout.splitlines():
        stripped = raw_line.strip()
        # Skip header and empty lines
        if not stripped or stripped.startswith(("-", "mode", "Using", "Reading", "Refining")):
            continue
        parts = stripped.split()
        if len(parts) >= 5:
            try:
                mode_idx = int(parts[0])
                if mode_idx == 1:
                    return {
                        "gnina_score": float(parts[1]),
                        "gnina_affinity": float(parts[4]) if len(parts) > 4 else float("nan"),
                        "cnn_score": float(parts[3]) if len(parts) > 3 else float("nan"),
                    }
            except (ValueError, IndexError):
                continue

    return nan_result


def _score_target(
    gnina_bin: Path,
    target_id: str,
    _target_label: str,
    receptor_pdb: Path,
    smiles_list: list[str],
    out_path: Path,
    *,
    force: bool = False,
    center: tuple[float, float, float] | None = None,
    box_size: tuple[float, float, float] | None = None,
) -> pd.DataFrame:
    """Score all molecules against one target with GNINA.

    Args:
        gnina_bin: Path to the GNINA binary.
        target_id: Target identifier.
        target_label: ``"A"`` or ``"B"``.
        receptor_pdb: Path to the receptor PDB file.
        smiles_list: SMILES strings to score.
        out_path: Path to write the output Parquet.
        force: Re-score even if output already exists.
        center: Optional box center.
        box_size: Optional box size.

    Returns:
        DataFrame with ``smiles``, ``gnina_score``, ``gnina_affinity``,
        ``cnn_score`` columns.
    """
    from dualdocking.data.io_utils import stem_exists

    if not force and stem_exists(out_path.with_suffix("")):
        logger.info("GNINA scores for target {} already exist, loading", target_id)
        from dualdocking.data.io_utils import read_df

        return read_df(out_path.with_suffix(""))

    logger.info("Scoring {} molecules against {} with GNINA", len(smiles_list), target_id)
    results = []

    for i, smi in enumerate(smiles_list):
        if (i + 1) % 50 == 0:
            logger.info("  Progress: {}/{}", i + 1, len(smiles_list))

        with tempfile.TemporaryDirectory(prefix=f"gnina_{target_id}_") as tmp:
            work = Path(tmp)
            sdf = _prepare_ligand_sdf(smi, work)
            if sdf is None:
                results.append({"smiles": smi, "gnina_score": np.nan, "gnina_affinity": np.nan, "cnn_score": np.nan})
                continue

            scores = _score_one_gnina(
                gnina_bin,
                receptor_pdb,
                sdf,
                center=center,
                box_size=box_size,
            )
            results.append({"smiles": smi, **scores})

    df = pd.DataFrame(results)

    from dualdocking.data.io_utils import write_df

    write_df(df, out_path.with_suffix(""), csv_too=True)
    logger.info("GNINA scores for target {} saved: {}", target_id, out_path)
    return df


@click.command()
@click.option("--pair", required=True, help="Pair ID from configs/pairs.yaml")
@click.option(
    "--mode",
    type=click.Choice(["single_a", "single_b", "double", "linear_equal", "biased_a", "selective_a"]),
    required=True,
)
@click.option("--seed", default=42, type=int)
@click.option("--config", default="configs/pairs.yaml")
@click.option("--experiment-config", default="configs/experiment.yaml")
@click.option("--force", is_flag=True, default=False, help="Re-score even if results exist")
def main(
    pair: str,
    mode: str,
    seed: int,
    config: str,
    experiment_config: str,
    force: bool,
) -> None:
    """Score generated molecules with GNINA CNN-based scoring.

    Reads molecules from ``scored_joined.parquet``, scores each against
    both targets with GNINA, and writes ``gnina_scores_A.parquet`` and
    ``gnina_scores_B.parquet``.

    Args:
        pair: Protein pair identifier from ``configs/pairs.yaml``.
        mode: Generation mode.
        seed: Random seed identifying the run.
        config: Path to ``pairs.yaml``.
        experiment_config: Path to ``experiment.yaml``.
        force: Re-score even if output already exists.
    """
    _ = experiment_config  # reserved for future scorer config
    gnina_bin = _find_gnina()
    logger.info("Using GNINA: {}", gnina_bin)

    from dualdocking.config import load_pairs
    from dualdocking.data.io_utils import read_df, stem_exists

    pairs_cfg = load_pairs(PROJECT_ROOT / config)["pairs"]
    if pair not in pairs_cfg:
        logger.error("Pair '{}' not found", pair)
        sys.exit(1)

    pair_cfg = pairs_cfg[pair]
    result_dir = PROJECT_ROOT / "output" / "results" / pair / mode / f"seed_{seed}"

    scored_stem = result_dir / "scored_joined"
    if not stem_exists(scored_stem):
        logger.error("Scored molecules not found: {}", scored_stem)
        sys.exit(1)

    scored_df = read_df(scored_stem)
    smiles_list = scored_df["smiles"].tolist()

    # Resolve receptor PDB paths
    targets_dir = PROJECT_ROOT / "output" / "data" / "targets" / pair
    target_a_id = pair_cfg["target_a"]["id"]
    target_b_id = pair_cfg["target_b"]["id"]
    pdb_a = targets_dir / "A" / f"{pair_cfg['target_a']['pdb']}.pdb"
    pdb_b = targets_dir / "B" / f"{pair_cfg['target_b']['pdb']}.pdb"

    for pdb, tid in [(pdb_a, target_a_id), (pdb_b, target_b_id)]:
        if not pdb.exists():
            logger.error("Receptor PDB not found for {}: {}", tid, pdb)
            sys.exit(1)

    # Resolve binding box from pairs.yaml (if available)
    def _get_box(target_cfg: dict) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
        box = target_cfg.get("box", {})
        if "center" in box and "size" in box:
            c = box["center"]
            s = box["size"]
            return tuple(c), tuple(s)
        return None, None

    center_a, size_a = _get_box(pair_cfg["target_a"])
    center_b, size_b = _get_box(pair_cfg["target_b"])

    # Score both targets
    _score_target(
        gnina_bin, target_a_id, "A", pdb_a, smiles_list,
        result_dir / "gnina_scores_A.parquet",
        force=force, center=center_a, box_size=size_a,
    )
    _score_target(
        gnina_bin, target_b_id, "B", pdb_b, smiles_list,
        result_dir / "gnina_scores_B.parquet",
        force=force, center=center_b, box_size=size_b,
    )

    logger.info("GNINA scoring complete for {}/{}/seed_{}", pair, mode, seed)


if __name__ == "__main__":
    main()
