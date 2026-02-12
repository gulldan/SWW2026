"""Download and prepare protein target structures for docking experiments.

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
import shutil
import subprocess
import sys
import time
from pathlib import Path

import click
from loguru import logger
from tqdm import tqdm

from dualdocking.data.http_utils import get_with_retry, sha256_file

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "output" / "data" / "targets"
MSA_CACHE_DIR: Path = PROJECT_ROOT / "output" / "cache" / "boltz_msa"
CHECKPOINT_PATH: Path = PROJECT_ROOT / "output" / "cache" / "prepare_checkpoint.json"
MANIFEST_PATH: Path = DATA_DIR / "fetch_manifest.json"
DOCKER_BIN: str = shutil.which("docker") or "docker"

MSA_RETRY_ATTEMPTS: int = 3
MSA_RETRY_BACKOFF: int = 30  # seconds

FREEDPP_DOCKER_IMAGE: str = "dualdocking/freedpp:latest"


def _load_checkpoint() -> dict:
    """Load the target preparation checkpoint from disk.

    Returns:
        Dictionary mapping ``"<pair_id>/<role>/<step>"`` keys to completion
        booleans, or an empty dict if no checkpoint file exists.
    """
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text())
    return {}


def _save_checkpoint(checkpoint: dict) -> None:
    """Persist the preparation checkpoint to disk.

    Args:
        checkpoint: Current checkpoint state to save.
    """
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(checkpoint, indent=2))


def _step_done(checkpoint: dict, pair_id: str, role: str, step: str) -> bool:
    """Check whether a specific preparation step has been completed.

    Args:
        checkpoint: Current checkpoint dictionary.
        pair_id: Pair identifier (e.g. ``"CDK2_ERK2"``).
        role: Target role (``"A"`` or ``"B"``).
        step: Step name (``"pdb"``, ``"fasta"``, ``"msa"``, ``"pdbqt"``).

    Returns:
        True if the step is recorded as complete.
    """
    return checkpoint.get(f"{pair_id}/{role}/{step}") is True


def _mark_step(checkpoint: dict, pair_id: str, role: str, step: str) -> None:
    """Mark a preparation step as complete and persist the checkpoint.

    Args:
        checkpoint: Current checkpoint dictionary (mutated in place).
        pair_id: Pair identifier.
        role: Target role.
        step: Step name.
    """
    checkpoint[f"{pair_id}/{role}/{step}"] = True
    _save_checkpoint(checkpoint)


def _record_manifest(path: Path, *, url: str) -> None:
    """Record a downloaded file in the fetch manifest with SHA-256.

    Appends to ``MANIFEST_PATH`` (JSON dict keyed by relative path).

    Args:
        path: Downloaded file path.
        url: Source URL.
    """
    manifest: dict = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())

    rel = str(path.relative_to(PROJECT_ROOT))
    manifest[rel] = {
        "sha256": sha256_file(path),
        "url": url,
        "size_bytes": path.stat().st_size,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def download_pdb(pdb_id: str, output_path: Path) -> Path:
    """Download a PDB file from RCSB with retry.

    Args:
        pdb_id: Four-character PDB accession code.
        output_path: Local path to write the PDB file.

    Returns:
        The *output_path* after successful download.
    """
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    logger.info("Downloading PDB {} -> {}", url, output_path)
    resp = get_with_retry(url, timeout=30)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(resp.text)
    _record_manifest(output_path, url=url)
    return output_path


def fetch_sequence(uniprot_id: str, target_id: str) -> Path:
    """Download a FASTA sequence from UniProt with retry.

    Args:
        uniprot_id: UniProt accession (e.g. ``"P24941"``).
        target_id: Internal target identifier used for the filename.

    Returns:
        Path to the saved FASTA file (``data/targets/{target_id}.fasta``).
    """
    fasta_path = DATA_DIR / f"{target_id}.fasta"
    if fasta_path.exists():
        logger.info("FASTA already exists: {}", fasta_path)
        return fasta_path

    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    logger.info("Downloading FASTA {} -> {}", url, fasta_path)
    resp = get_with_retry(url, timeout=30)

    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    fasta_path.write_text(resp.text)
    _record_manifest(fasta_path, url=url)
    logger.info("FASTA saved: {} ({} bytes)", fasta_path, len(resp.text))
    return fasta_path


def _read_sequence(fasta_path: Path) -> str:
    """Read the amino acid sequence from a FASTA file, skipping headers.

    Args:
        fasta_path: Path to the FASTA file.

    Returns:
        Concatenated sequence string (no whitespace or header lines).
    """
    lines = fasta_path.read_text().strip().splitlines()
    return "".join(line.strip() for line in lines if not line.startswith(">"))


def _write_warmup_yaml(sequence: str, yaml_path: Path) -> None:
    """Generate a Boltz input YAML for MSA warmup (protein + benzene + affinity).

    Args:
        sequence: Protein amino acid sequence.
        yaml_path: Output path for the YAML file.
    """
    import yaml

    data = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": sequence}},
            {"ligand": {"id": "B", "smiles": "c1ccccc1"}},  # benzene as dummy
        ],
        "properties": [
            {"affinity": {"binder": "B"}},
        ],
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


def precompute_msa(target_id: str, sequence: str) -> Path | None:
    """Pre-compute MSA for a protein via Boltz + ColabFold API.

    Runs ``boltz predict`` with ``--use_msa_server`` and a dummy ligand.
    The MSA is cached in ``cache/boltz_msa/{target_id}.csv``.

    Args:
        target_id: Internal target identifier.
        sequence: Protein amino acid sequence.

    Returns:
        Path to the MSA CSV file, or None on failure.
    """
    msa_csv = MSA_CACHE_DIR / f"{target_id}.csv"
    if msa_csv.exists():
        logger.info("MSA already exists: {}", msa_csv)
        return msa_csv

    out_dir = MSA_CACHE_DIR / target_id
    out_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = out_dir / "msa_warmup.yaml"
    _write_warmup_yaml(sequence, yaml_path)

    cmd = [
        "uv",
        "run",
        "boltz",
        "predict",
        str(yaml_path),
        "--out_dir",
        str(out_dir),
        "--model",
        "boltz2",
        "--accelerator",
        "gpu",
        "--use_msa_server",
    ]

    for attempt in range(1, MSA_RETRY_ATTEMPTS + 1):
        logger.info("Pre-computing MSA for {} (attempt {}/{})", target_id, attempt, MSA_RETRY_ATTEMPTS)
        logger.debug("CMD: {}", " ".join(cmd))

        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(PROJECT_ROOT),
            check=False,
        )

        # MSA is generated before GPU inference.  Even if predict crashes
        # (e.g. missing cuequivariance_torch), the MSA .csv is already created.
        csv_files = list(out_dir.rglob("msa/*.csv"))
        if csv_files:
            logger.info("MSA .csv found after attempt {} (rc={})", attempt, result.returncode)
            if result.returncode != 0:
                logger.info("Boltz predict crashed after MSA stage — expected during warmup")
            break

        if result.returncode != 0:
            logger.warning(
                "Boltz MSA error (attempt {}/{}): rc={}\nstderr: {}",
                attempt,
                MSA_RETRY_ATTEMPTS,
                result.returncode,
                (result.stderr or "")[-1000:],
            )
        if attempt < MSA_RETRY_ATTEMPTS:
            logger.info("Retrying in {} sec...", MSA_RETRY_BACKOFF * attempt)
            time.sleep(MSA_RETRY_BACKOFF * attempt)
    else:
        logger.error("Failed to pre-compute MSA for {} after {} attempts", target_id, MSA_RETRY_ATTEMPTS)
        return None

    csv_files = list(out_dir.rglob("msa/*.csv"))
    if not csv_files:
        logger.error("MSA .csv file not found in {}", out_dir)
        return None

    # Copy the first found .csv -> cache/boltz_msa/{target_id}.csv
    src_csv = csv_files[0]
    shutil.copy2(src_csv, msa_csv)
    logger.info("MSA copied: {} -> {}", src_csv, msa_csv)
    return msa_csv


def prepare_receptor_pdbqt(pdb_path: Path, target_id: str) -> Path:
    """Convert a PDB file to PDBQT via Docker (RDKit + Meeko).

    Args:
        pdb_path: Path to the input PDB file.
        target_id: Target identifier (used for the output filename).

    Returns:
        Path to the generated PDBQT file.
    """
    pdbqt_path = DATA_DIR / f"{target_id}.pdbqt"
    if pdbqt_path.exists():
        logger.info("PDBQT already exists: {}", pdbqt_path)
        return pdbqt_path

    # Check Docker image exists
    inspect = subprocess.run(  # noqa: S603
        [DOCKER_BIN, "image", "inspect", FREEDPP_DOCKER_IMAGE],
        capture_output=True,
        check=False,
    )
    if inspect.returncode != 0:
        logger.warning(
            "Docker image {} not found. Build: docker build -f docker/freedpp/Dockerfile -t {} .",
            FREEDPP_DOCKER_IMAGE,
            FREEDPP_DOCKER_IMAGE,
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
            f"{DATA_DIR}:/output",
            FREEDPP_DOCKER_IMAGE,
            "prepare-receptor",
            f"/input/{pdb_path.name}",
            f"/output/{target_id}.pdbqt",
        ],
        check=True,
        timeout=300,
    )
    logger.info("PDBQT saved: {}", pdbqt_path)
    return pdbqt_path


def prepare_pair(pair_id: str, pair_cfg: dict, *, prepare_all: bool = False, checkpoint: dict | None = None) -> None:
    """Prepare input data for a single target pair.

    Downloads PDB files for both targets.  With *prepare_all*, also
    downloads FASTA, pre-computes MSA, and converts PDB to PDBQT.

    Args:
        pair_id: Pair identifier (e.g. ``"CDK2_ERK2"``).
        pair_cfg: Pair configuration dict from ``pairs.yaml``.
        prepare_all: If True, run all preparation steps (FASTA, MSA, PDBQT).
        checkpoint: Mutable checkpoint dict tracking completed steps.
    """
    if checkpoint is None:
        checkpoint = {}

    base = PROJECT_ROOT / "output" / "data" / "targets" / pair_id

    roles = ["A", "B"]
    for role in roles:
        target = pair_cfg[f"target_{role.lower()}"]
        pdb_id = target["pdb_id"]
        target_id = target["id"]
        uniprot_id = target["uniprot"]
        out_dir = base / role
        out_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: PDB
        if _step_done(checkpoint, pair_id, role, "pdb"):
            logger.debug("Checkpoint: PDB {} already done", target_id)
        else:
            pdb_path = out_dir / f"{pdb_id.lower()}.pdb"
            if pdb_path.exists():
                logger.info("Already exists: {}", pdb_path)
            else:
                download_pdb(pdb_id, pdb_path)
            _mark_step(checkpoint, pair_id, role, "pdb")

        pdb_path = out_dir / f"{pdb_id.lower()}.pdb"

        # Save metadata
        meta_path = out_dir / "meta.yaml"
        if not meta_path.exists():
            import yaml

            meta_path.write_text(yaml.dump(target, default_flow_style=False))

        if prepare_all:
            # Step 2: FASTA
            if _step_done(checkpoint, pair_id, role, "fasta"):
                logger.debug("Checkpoint: FASTA {} already done", target_id)
            else:
                fetch_sequence(uniprot_id, target_id)
                _mark_step(checkpoint, pair_id, role, "fasta")

            fasta_path = DATA_DIR / f"{target_id}.fasta"

            # Step 3: MSA pre-compute
            if _step_done(checkpoint, pair_id, role, "msa"):
                logger.debug("Checkpoint: MSA {} already done", target_id)
            else:
                sequence = _read_sequence(fasta_path)
                precompute_msa(target_id, sequence)
                _mark_step(checkpoint, pair_id, role, "msa")

            # Step 4: PDB -> PDBQT (for FREED++ / UniDock GPU)
            if _step_done(checkpoint, pair_id, role, "pdbqt"):
                logger.debug("Checkpoint: PDBQT {} already done", target_id)
            else:
                prepare_receptor_pdbqt(pdb_path, target_id)
                _mark_step(checkpoint, pair_id, role, "pdbqt")


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

    Args:
        config: Relative path to the pairs YAML config.
        pair: If set, process only this pair (must exist in config).
        prepare_all: Run all preparation steps (PDB + FASTA + MSA + PDBQT).
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

    checkpoint = _load_checkpoint()
    n_steps_per_pair = 4 if prepare_all else 1  # pdb, fasta, msa, pdbqt  vs  pdb only
    total_steps = len(pairs) * 2 * n_steps_per_pair  # 2 targets per pair

    # Count already-completed steps for correct initial progress
    done_steps = sum(
        1
        for pair_id in pairs
        for role in ("A", "B")
        for step in (["pdb", "fasta", "msa", "pdbqt"] if prepare_all else ["pdb"])
        if _step_done(checkpoint, pair_id, role, step)
    )

    with tqdm(total=total_steps, initial=done_steps, desc="Preparing targets", unit="step") as pbar:
        for pair_id, pair_cfg in pairs.items():
            logger.info("Preparing pair: {}", pair_id)
            steps_before = sum(1 for k, v in checkpoint.items() if v is True)
            prepare_pair(pair_id, pair_cfg, prepare_all=prepare_all, checkpoint=checkpoint)
            steps_after = sum(1 for k, v in checkpoint.items() if v is True)
            pbar.update(steps_after - steps_before)

    if prepare_all:
        logger.info("Full preparation complete. Scoring pipeline ready for OFFLINE operation.")
    logger.info("Done. Targets saved to data/targets/")


if __name__ == "__main__":
    main()
