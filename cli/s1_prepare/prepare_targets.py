"""Download and prepare protein target structures for docking experiments.

Thin CLI wrapper -- core logic lives in :mod:`dualdocking.data.prep.targets`
and :mod:`dualdocking.data.fetch.pdb`.

Reads pairs from ``configs/pairs.yaml``, downloads PDB files, and
organizes them in ``data/targets/<pair_id>/{A,B}/``.

With ``--prepare-all``, also:
    - Downloads FASTA sequences from UniProt
    - Pre-computes MSA for each protein via ColabFold
    - Converts PDB to PDBQT using Docker (RDKit + Meeko)

After ``--prepare-all``, the entire scoring pipeline works OFFLINE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

from dualdocking.data.fetch.pdb import record_manifest
from dualdocking.data.prep.targets import PrepCheckpoint, prepare_pair

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "output" / "data" / "targets"
MSA_CACHE_DIR: Path = PROJECT_ROOT / "output" / "cache" / "boltz_msa"
CHECKPOINT_PATH: Path = PROJECT_ROOT / "output" / "cache" / "prepare_checkpoint.json"
MANIFEST_PATH: Path = DATA_DIR / "fetch_manifest.json"


# ── Backward-compat shims (used by tests) ────────────────────────────────


def _load_checkpoint() -> dict:
    """Load the target preparation checkpoint from disk."""
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text())
    return {}


def _save_checkpoint(checkpoint: dict) -> None:
    """Persist the preparation checkpoint to disk."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(checkpoint, indent=2))


def _step_done(checkpoint: dict, pair_id: str, role: str, step: str) -> bool:
    """Check whether a specific preparation step has been completed."""
    return checkpoint.get(f"{pair_id}/{role}/{step}") is True


def _mark_step(checkpoint: dict, pair_id: str, role: str, step: str) -> None:
    """Mark a preparation step as complete and persist the checkpoint."""
    checkpoint[f"{pair_id}/{role}/{step}"] = True
    _save_checkpoint(checkpoint)


def _record_manifest(path: Path, *, url: str) -> None:
    """Record a downloaded file in the fetch manifest with SHA-256."""
    record_manifest(path, url=url, manifest_path=MANIFEST_PATH, project_root=PROJECT_ROOT)


@click.command()
@click.option("--config", default="configs/pairs.yaml", help="Path to pairs config")
@click.option("--pair", default=None, help="Process only this pair")
@click.option(
    "--prepare-all",
    is_flag=True,
    default=False,
    help="Full preparation: PDB + FASTA + MSA + PDBQT (needs internet)",
)
def main(config: str, pair: str | None, prepare_all: bool) -> None:
    """Download and prepare protein target structures.

    Reads target pairs from the configuration file, downloads PDB
    structures from RCSB, and optionally performs full offline
    preparation including FASTA download, MSA computation, and
    PDB-to-PDBQT conversion.
    """
    import yaml

    config_path = PROJECT_ROOT / config
    with config_path.open() as f:
        pairs = yaml.safe_load(f)["pairs"]

    if pair:
        if pair not in pairs:
            logger.error("Pair '{}' not found in config", pair)
            sys.exit(1)
        pairs = {pair: pairs[pair]}

    checkpoint = PrepCheckpoint.load(CHECKPOINT_PATH)
    n_steps_per_pair = 4 if prepare_all else 1
    total_steps = len(pairs) * 2 * n_steps_per_pair

    done_steps = sum(
        1
        for pair_id in pairs
        for role in ("A", "B")
        for step in (["pdb", "fasta", "msa", "pdbqt"] if prepare_all else ["pdb"])
        if checkpoint.is_done(pair_id, role, step)
    )

    with tqdm(total=total_steps, initial=done_steps, desc="Preparing targets", unit="step") as pbar:
        for pair_id, pair_cfg in pairs.items():
            logger.info("Preparing pair: {}", pair_id)
            steps_before = sum(1 for v in checkpoint.data.values() if v is True)
            prepare_pair(
                pair_id,
                pair_cfg,
                data_dir=DATA_DIR,
                msa_cache_dir=MSA_CACHE_DIR,
                project_root=PROJECT_ROOT,
                checkpoint=checkpoint,
                full=prepare_all,
            )
            steps_after = sum(1 for v in checkpoint.data.values() if v is True)
            pbar.update(steps_after - steps_before)

    if prepare_all:
        logger.info("Full preparation complete. Scoring pipeline ready for OFFLINE operation.")
    logger.info("Done. Targets saved to data/targets/")


if __name__ == "__main__":
    main()
