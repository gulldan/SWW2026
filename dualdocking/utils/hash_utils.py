"""Hashing helpers shared across data fetch, manifests, and reporting."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, *, chunk_size: int = 8192) -> str:
    """Compute a SHA-256 digest for a local file.

    Args:
        path: File path to hash.
        chunk_size: Read size in bytes for streaming hash computation.

    Returns:
        Hex-encoded SHA-256 digest.

    Raises:
        ValueError: If ``chunk_size`` is non-positive.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
