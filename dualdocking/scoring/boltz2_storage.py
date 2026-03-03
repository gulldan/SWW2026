"""Storage and sequence/MSA helpers for Boltz-2 scoring."""

from __future__ import annotations

from pathlib import Path


def load_sequence(target_id: str, *, data_dir: Path, sequence_cache: dict[str, str]) -> str:
    """Load and cache target sequence from FASTA.

    Args:
        target_id: Target identifier used as FASTA stem.
        data_dir: Directory containing ``{target_id}.fasta`` files.
        sequence_cache: Mutable in-memory cache mapping target to sequence.

    Returns:
        Amino-acid sequence with FASTA headers removed.

    Raises:
        FileNotFoundError: FASTA file does not exist.
    """
    if target_id in sequence_cache:
        return sequence_cache[target_id]

    fasta_path = data_dir / f"{target_id}.fasta"
    if not fasta_path.exists():
        msg = f"FASTA not found: {fasta_path}. Run: uv run python cli/s1_prepare/prepare_targets.py --prepare-all"
        raise FileNotFoundError(msg)

    lines = fasta_path.read_text().strip().splitlines()
    sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
    sequence_cache[target_id] = sequence
    return sequence


def get_msa_path(target_id: str, *, msa_cache_dir: Path) -> Path | None:
    """Return precomputed MSA path for target if available.

    Args:
        target_id: Target identifier used as MSA stem.
        msa_cache_dir: Directory containing ``{target_id}.csv`` MSA files.

    Returns:
        Existing MSA CSV path, or ``None`` when missing.
    """
    msa_path = msa_cache_dir / f"{target_id}.csv"
    if msa_path.exists():
        return msa_path
    return None
