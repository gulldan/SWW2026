"""Fetch calibration SMILES from HuggingFace datasets or URLs.

Extracted from ``cli/s0_fetch/fetch_calibration_smiles.py``.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Iterable

from loguru import logger

from dualdocking.utils.hash_utils import sha256_file


def sha256_of_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return sha256_file(path)


def reservoir_sample(items: Iterable[str], k: int, rng: random.Random) -> list[str]:
    """Reservoir-sample up to *k* non-empty items from an iterable.

    Args:
        items: Stream of string items to sample from.
        k: Maximum number of items to retain.
        rng: Random number generator for reproducibility.

    Returns:
        List of at most *k* sampled items.
    """
    sample: list[str] = []
    for i, item in enumerate(items):
        if not item:
            continue
        if len(sample) < k:
            sample.append(item)
            continue
        j = rng.randint(0, i)
        if j < k:
            sample[j] = item
    return sample


def load_smiles_from_hf(dataset_name: str, split: str, smiles_column: str) -> Iterable[str]:
    """Stream SMILES strings from a HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g. ``"user/dataset"``).
        split: Dataset split to use (e.g. ``"train"``).
        smiles_column: Name of the column containing SMILES strings.

    Yields:
        Stripped, non-empty SMILES strings.

    Raises:
        ImportError: If the ``datasets`` package is not installed.
        KeyError: If *smiles_column* is not found in the dataset.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("Install datasets: uv pip install datasets") from exc

    ds = load_dataset(dataset_name, split=split, streaming=True)
    for row in ds:
        if smiles_column not in row:
            raise KeyError(f"Column '{smiles_column}' not found in dataset {dataset_name}")
        val = row[smiles_column]
        if val:
            yield str(val).strip()


def load_smiles_from_url(url: str, smiles_column: int) -> Iterable[str]:
    """Stream SMILES strings from a whitespace-delimited remote text file.

    Args:
        url: URL to fetch.
        smiles_column: Zero-based column index containing the SMILES.

    Yields:
        Stripped SMILES strings from the selected column.
    """
    import requests

    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            parts = raw.split()
            if len(parts) <= smiles_column:
                continue
            yield parts[smiles_column].strip()


def canonicalize_smiles(smiles: list[str]) -> list[str]:
    """Canonicalize a list of SMILES using RDKit.

    Args:
        smiles: Raw SMILES strings.

    Returns:
        List of canonical SMILES (invalid entries are silently dropped).
    """
    from dualdocking.data.chem_sanitize import canonicalize

    out: list[str] = []
    for smi in smiles:
        canon = canonicalize(smi)
        if canon:
            out.append(canon)
    return out


def fetch_calibration_smiles(
    sources: dict[str, dict],
    selected: str,
    *,
    output_dir: Path,
    output_path: Path,
    sample_n: int = 1000,
    seed: int = 137,
    download_all: bool = True,
    do_canonicalize: bool = False,
    force: bool = False,
) -> dict[str, dict]:
    """Fetch and sample calibration SMILES from configured sources.

    Args:
        sources: Dict of source name to source config (type, dataset/url, etc.).
        selected: Name of the selected dataset.
        output_dir: Directory for per-source .smi files.
        output_path: Canonical output path for the selected dataset.
        sample_n: Number of SMILES to reservoir-sample.
        seed: Random seed for reproducibility.
        download_all: If True, download all sources; if False, only selected.
        do_canonicalize: If True, canonicalize SMILES after sampling.
        force: If True, overwrite existing files.

    Returns:
        Manifest dict mapping source names to metadata.

    Raises:
        KeyError: If *selected* is not in *sources*.
    """
    if selected not in sources:
        raise KeyError(f"selected dataset '{selected}' not in sources")

    to_fetch = list(sources.keys()) if download_all else [selected]

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    selected_ok = False

    for name in to_fetch:
        src = sources[name]
        if src.get("enabled") is False:
            logger.info("Skip disabled source: {}", name)
            continue

        out_file = output_dir / f"{name}_n{sample_n}_seed{seed}.smi"
        try:
            if out_file.exists() and not force:
                logger.info("Exists, skip: {}", out_file)
            else:
                rng = random.Random(seed)  # noqa: S311
                if src["type"] == "huggingface":
                    smiles_iter = load_smiles_from_hf(src["dataset"], src.get("split", "train"), src["smiles_column"])
                elif src["type"] == "url":
                    url = src.get("url")
                    if not url:
                        raise ValueError(f"Source '{name}' requires url")
                    smiles_iter = load_smiles_from_url(url, int(src.get("smiles_column", 0)))
                else:
                    raise ValueError(f"Unknown source type: {src['type']}")

                sampled = reservoir_sample(smiles_iter, sample_n, rng)
                if do_canonicalize:
                    sampled = canonicalize_smiles(sampled)
                out_file.write_text("\n".join(sampled) + "\n")
                logger.info("Saved {} SMILES -> {}", len(sampled), out_file)

            manifest[name] = {
                "path": str(out_file),
                "sha256": sha256_of_file(out_file),
                "n": sample_n,
                "seed": seed,
                "source": src,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

            if name == selected:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(out_file.read_text())
                logger.info("Selected dataset -> {}", output_path)
                selected_ok = True
        except Exception as exc:
            logger.error("Failed to fetch {}: {}", name, exc)
            manifest[name] = {
                "error": str(exc),
                "source": src,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest -> {}", manifest_path)

    if not selected_ok:
        raise SystemExit(f"Selected dataset '{selected}' was not downloaded; check manifest for errors.")

    return manifest
