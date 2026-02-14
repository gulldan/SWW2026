"""Prepare protein targets: PDB, FASTA, MSA, and PDBQT conversion.

Extracted from ``cli/s1_prepare/prepare_targets.py``.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from dualdocking.data.fetch.docker import DOCKER_BIN, docker_image_exists
from dualdocking.data.fetch.msa import compute_msa
from dualdocking.data.fetch.pdb import download_fasta, download_pdb, read_sequence

FREEDPP_DOCKER_IMAGE: str = "dualdocking/freedpp:latest"


@dataclass
class TargetPrepConfig:
    """Configuration for preparing a single target.

    Attributes:
        target_id: Internal target identifier (e.g. ``"CDK2"``).
        pdb_id: Four-character PDB accession code.
        uniprot_id: UniProt accession for FASTA download.
        pair_id: Parent pair identifier (e.g. ``"CDK2_ERK2"``).
        role: Target role within the pair (``"A"`` or ``"B"``).
        data_dir: Base directory for target output files.
        msa_cache_dir: Directory for cached MSA files.
        project_root: Project root directory.
        docker_image: Docker image for PDB-to-PDBQT conversion.
    """

    target_id: str
    pdb_id: str
    uniprot_id: str
    pair_id: str
    role: str
    data_dir: Path
    msa_cache_dir: Path
    project_root: Path
    docker_image: str = FREEDPP_DOCKER_IMAGE


@dataclass
class PrepCheckpoint:
    """Track completed preparation steps with persistence.

    Attributes:
        path: Path to the checkpoint JSON file.
        data: Mutable mapping of step keys to completion booleans.
    """

    path: Path
    data: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> PrepCheckpoint:
        """Load checkpoint from disk, or create empty if missing.

        Args:
            path: Path to the checkpoint JSON file.

        Returns:
            Loaded or new checkpoint instance.
        """
        data: dict[str, bool] = {}
        if path.exists():
            data = json.loads(path.read_text())
        return cls(path=path, data=data)

    def save(self) -> None:
        """Persist checkpoint to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))

    def is_done(self, pair_id: str, role: str, step: str) -> bool:
        """Check whether a specific step has been completed.

        Args:
            pair_id: Pair identifier.
            role: Target role (``"A"`` or ``"B"``).
            step: Step name (``"pdb"``, ``"fasta"``, ``"msa"``, ``"pdbqt"``).

        Returns:
            True if the step is recorded as complete.
        """
        return self.data.get(f"{pair_id}/{role}/{step}") is True

    def mark(self, pair_id: str, role: str, step: str) -> None:
        """Mark a step as complete and persist.

        Args:
            pair_id: Pair identifier.
            role: Target role.
            step: Step name.
        """
        self.data[f"{pair_id}/{role}/{step}"] = True
        self.save()


def prepare_receptor_pdbqt(pdb_path: Path, target_id: str, data_dir: Path, docker_image: str) -> Path:
    """Convert a PDB file to PDBQT via Docker (RDKit + Meeko).

    Args:
        pdb_path: Path to the input PDB file.
        target_id: Target identifier (used for the output filename).
        data_dir: Directory for the output PDBQT file.
        docker_image: Docker image to use for conversion.

    Returns:
        Path to the generated PDBQT file.
    """
    pdbqt_path = data_dir / f"{target_id}.pdbqt"
    if pdbqt_path.exists():
        logger.info("PDBQT already exists: {}", pdbqt_path)
        return pdbqt_path

    if not docker_image_exists(docker_image):
        logger.warning(
            "Docker image {} not found. Build: docker build -f docker/freedpp/Dockerfile -t {} .",
            docker_image,
            docker_image,
        )
        return pdbqt_path

    logger.info("Converting PDB -> PDBQT: {} -> {}", pdb_path, pdbqt_path)
    subprocess.run(  # noqa: S603
        [
            DOCKER_BIN,
            "run",
            "--rm",
            "-v",
            f"{pdb_path.parent}:/input:ro",
            "-v",
            f"{data_dir}:/output",
            docker_image,
            "prepare-receptor",
            f"/input/{pdb_path.name}",
            f"/output/{target_id}.pdbqt",
        ],
        check=True,
        timeout=300,
    )
    logger.info("PDBQT saved: {}", pdbqt_path)
    return pdbqt_path


def prepare_target(config: TargetPrepConfig, checkpoint: PrepCheckpoint, *, full: bool = False) -> None:
    """Prepare a single target (PDB, and optionally FASTA + MSA + PDBQT).

    Args:
        config: Target preparation configuration.
        checkpoint: Checkpoint tracker for resumable preparation.
        full: If True, run all steps (PDB + FASTA + MSA + PDBQT).
    """
    pair_id = config.pair_id
    role = config.role
    out_dir = config.data_dir / config.pair_id / role
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: PDB
    pdb_path = out_dir / f"{config.pdb_id.lower()}.pdb"
    if checkpoint.is_done(pair_id, role, "pdb"):
        logger.debug("Checkpoint: PDB {} already done", config.target_id)
    else:
        if pdb_path.exists():
            logger.info("Already exists: {}", pdb_path)
        else:
            download_pdb(config.pdb_id, pdb_path)
        checkpoint.mark(pair_id, role, "pdb")

    # Save metadata
    meta_path = out_dir / "meta.yaml"
    if not meta_path.exists():
        import yaml

        meta = {
            "id": config.target_id,
            "pdb_id": config.pdb_id,
            "uniprot": config.uniprot_id,
        }
        meta_path.write_text(yaml.dump(meta, default_flow_style=False))

    if not full:
        return

    # Step 2: FASTA
    fasta_path = config.data_dir / f"{config.target_id}.fasta"
    if checkpoint.is_done(pair_id, role, "fasta"):
        logger.debug("Checkpoint: FASTA {} already done", config.target_id)
    else:
        download_fasta(config.uniprot_id, fasta_path)
        checkpoint.mark(pair_id, role, "fasta")

    # Step 3: MSA
    if checkpoint.is_done(pair_id, role, "msa"):
        logger.debug("Checkpoint: MSA {} already done", config.target_id)
    else:
        sequence = read_sequence(fasta_path)
        compute_msa(config.target_id, sequence, config.msa_cache_dir, config.project_root)
        checkpoint.mark(pair_id, role, "msa")

    # Step 4: PDBQT
    if checkpoint.is_done(pair_id, role, "pdbqt"):
        logger.debug("Checkpoint: PDBQT {} already done", config.target_id)
    else:
        prepare_receptor_pdbqt(pdb_path, config.target_id, config.data_dir, config.docker_image)
        checkpoint.mark(pair_id, role, "pdbqt")


def prepare_pair(
    pair_id: str,
    pair_cfg: dict,
    *,
    data_dir: Path,
    msa_cache_dir: Path,
    project_root: Path,
    checkpoint: PrepCheckpoint,
    full: bool = False,
    docker_image: str = FREEDPP_DOCKER_IMAGE,
) -> None:
    """Prepare both targets in a pair.

    Args:
        pair_id: Pair identifier (e.g. ``"CDK2_ERK2"``).
        pair_cfg: Pair configuration dict from ``pairs.yaml``.
        data_dir: Base directory for target files.
        msa_cache_dir: Directory for cached MSA files.
        project_root: Project root directory.
        checkpoint: Checkpoint tracker.
        full: If True, run all preparation steps.
        docker_image: Docker image for PDBQT conversion.
    """
    for role in ("A", "B"):
        target = pair_cfg[f"target_{role.lower()}"]
        config = TargetPrepConfig(
            target_id=target["id"],
            pdb_id=target["pdb_id"],
            uniprot_id=target["uniprot"],
            pair_id=pair_id,
            role=role,
            data_dir=data_dir,
            msa_cache_dir=msa_cache_dir,
            project_root=project_root,
            docker_image=docker_image,
        )
        prepare_target(config, checkpoint, full=full)
