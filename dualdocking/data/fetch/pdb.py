"""Download PDB structures from RCSB and FASTA sequences from UniProt.

Extracted from ``cli/s0_fetch/fetch_all.py`` and
``cli/s1_prepare/prepare_targets.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from loguru import logger

from dualdocking.data.http_utils import get_with_retry, sha256_file

_RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
_UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"


def download_pdb(pdb_id: str, output: Path) -> Path:
    """Download a PDB file from RCSB with retry.

    Args:
        pdb_id: Four-character PDB accession code.
        output: Local path to write the PDB file.

    Returns:
        The *output* path after successful download.
    """
    url = _RCSB_URL.format(pdb_id=pdb_id.upper())
    logger.info("Downloading PDB {} -> {}", url, output)
    resp = get_with_retry(url, timeout=30)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(resp.text)
    return output


def download_fasta(uniprot_id: str, output: Path) -> Path:
    """Download a FASTA sequence from UniProt with retry.

    Args:
        uniprot_id: UniProt accession (e.g. ``"P24941"``).
        output: Local path to write the FASTA file.

    Returns:
        Path to the saved FASTA file.
    """
    if output.exists():
        logger.info("FASTA already exists: {}", output)
        return output

    url = _UNIPROT_FASTA_URL.format(uniprot_id=uniprot_id)
    logger.info("Downloading FASTA {} -> {}", url, output)
    resp = get_with_retry(url, timeout=30)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(resp.text)
    logger.info("FASTA saved: {} ({} bytes)", output, len(resp.text))
    return output


def read_sequence(fasta_path: Path) -> str:
    """Read the amino acid sequence from a FASTA file, skipping headers.

    Args:
        fasta_path: Path to the FASTA file.

    Returns:
        Concatenated sequence string (no whitespace or header lines).
    """
    lines = fasta_path.read_text().strip().splitlines()
    return "".join(line.strip() for line in lines if not line.startswith(">"))


def record_manifest(
    path: Path,
    *,
    url: str,
    manifest_path: Path,
    project_root: Path,
) -> None:
    """Record a downloaded file in the fetch manifest with SHA-256.

    Args:
        path: Downloaded file path.
        url: Source URL.
        manifest_path: Path to the manifest JSON file.
        project_root: Project root for computing relative paths.
    """
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    rel = str(path.relative_to(project_root))
    manifest[rel] = {
        "sha256": sha256_file(path),
        "url": url,
        "size_bytes": path.stat().st_size,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
