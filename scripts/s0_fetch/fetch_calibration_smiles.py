"""Fetch and prepare calibration SMILES sets based on configs/calibration.yaml.

Downloads SMILES from HuggingFace datasets or URLs, reservoir-samples to a
fixed size, optionally canonicalizes, and writes ``.smi`` files to the
calibration output directory.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Iterable

import click
import yaml
from loguru import logger

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _reservoir_sample(items: Iterable[str], k: int, rng: random.Random) -> list[str]:
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


def _load_smiles_from_hf(dataset_name: str, split: str, smiles_column: str) -> Iterable[str]:
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


def _load_smiles_from_url(url: str, smiles_column: int) -> Iterable[str]:
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


def _canonicalize(smiles: list[str]) -> list[str]:
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


@click.command()
@click.option("--config", default="configs/calibration.yaml")
@click.option("--only-selected", is_flag=True, default=False)
@click.option("--force", is_flag=True, default=False)
def main(config: str, only_selected: bool, force: bool) -> None:
    """Fetch calibration SMILES from configured sources.

    Reads ``configs/calibration.yaml``, downloads and reservoir-samples
    SMILES datasets, writes ``.smi`` files to the output directory, and
    copies the selected dataset to the canonical output path.

    Args:
        config: Relative path to the calibration YAML config.
        only_selected: If True, download only the selected dataset.
        force: If True, overwrite existing files.
    """
    cfg_path = PROJECT_ROOT / config
    cfg = yaml.safe_load(cfg_path.read_text())

    selected = cfg.get("selected")
    output_path = PROJECT_ROOT / cfg.get("output_path")
    sample_n = int(cfg.get("sample", {}).get("n", 1000))
    seed = int(cfg.get("sample", {}).get("seed", 137))
    download_all = bool(cfg.get("options", {}).get("download_all", True))
    do_canonicalize = bool(cfg.get("options", {}).get("canonicalize", False))

    sources = cfg.get("sources", {})
    if selected not in sources:
        raise KeyError(f"selected dataset '{selected}' not in sources")

    if only_selected:
        download_all = False

    to_fetch = [selected] if not download_all else list(sources.keys())

    out_dir = PROJECT_ROOT / "output" / "data" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    selected_ok = False

    for name in to_fetch:
        src = sources[name]
        if src.get("enabled") is False:
            logger.info("Skip disabled source: {}", name)
            continue

        out_file = out_dir / f"{name}_n{sample_n}_seed{seed}.smi"
        try:
            if out_file.exists() and not force:
                logger.info("Exists, skip: {}", out_file)
            else:
                rng = random.Random(seed)  # noqa: S311
                if src["type"] == "huggingface":
                    smiles_iter = _load_smiles_from_hf(src["dataset"], src.get("split", "train"), src["smiles_column"])
                elif src["type"] == "url":
                    url = src.get("url")
                    if not url:
                        raise ValueError(f"Source '{name}' requires url")
                    smiles_iter = _load_smiles_from_url(url, int(src.get("smiles_column", 0)))
                else:
                    raise ValueError(f"Unknown source type: {src['type']}")

                sampled = _reservoir_sample(smiles_iter, sample_n, rng)
                if do_canonicalize:
                    sampled = _canonicalize(sampled)
                out_file.write_text("\n".join(sampled) + "\n")
                logger.info("Saved {} SMILES -> {}", len(sampled), out_file)

            manifest[name] = {
                "path": str(out_file),
                "sha256": _sha256(out_file),
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

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest -> {}", manifest_path)

    if not selected_ok:
        raise SystemExit(f"Selected dataset '{selected}' was not downloaded; check manifest for errors.")


if __name__ == "__main__":
    main()
