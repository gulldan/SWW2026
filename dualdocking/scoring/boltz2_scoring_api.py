"""High-level single and batch scoring routines for Boltz-2."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
from tqdm import tqdm


class _LoggerLike(Protocol):
    """Minimal logger protocol for scoring orchestration."""

    def warning(self, message: str, *args: object) -> None:
        """Log warning message.

        Args:
            message: Message template/text.
            *args: Format arguments.
        """

    def info(self, message: str, *args: object) -> None:
        """Log info message.

        Args:
            message: Message template/text.
            *args: Format arguments.
        """

    def error(self, message: str, *args: object) -> None:
        """Log error message.

        Args:
            message: Message template/text.
            *args: Format arguments.
        """


class _RunPredictFn(Protocol):
    """Protocol for single-input Boltz prediction callback."""

    def __call__(
        self,
        yaml_path: Path,
        out_dir: Path,
        *,
        device: str | None,
    ) -> subprocess.CompletedProcess[str]:
        """Run prediction and return subprocess result.

        Args:
            yaml_path: Input YAML path.
            out_dir: Output directory.
            device: Optional CUDA device token.

        Returns:
            Completed subprocess result.
        """


class _WriteYamlFn(Protocol):
    """Protocol for Boltz YAML writer callback."""

    def __call__(
        self,
        sequence: str,
        smiles: str,
        path: Path,
        *,
        msa_path: Path | None,
    ) -> None:
        """Write input YAML.

        Args:
            sequence: Target protein sequence.
            smiles: Ligand SMILES.
            path: Output YAML path.
            msa_path: Optional precomputed MSA CSV path.
        """


class _CacheKeyFn(Protocol):
    """Protocol for cache-key builder callback."""

    def __call__(
        self,
        target_id: str,
        smi: str,
        *,
        ctx_hash: str,
        seq_hash: str,
        msa_hash: str,
    ) -> str:
        """Build deterministic cache key.

        Args:
            target_id: Target identifier.
            smi: SMILES string.
            ctx_hash: Context hash.
            seq_hash: Sequence hash.
            msa_hash: MSA hash.

        Returns:
            Cache key string.
        """


class _ScoreBatchInternalFn(Protocol):
    """Protocol for backend-specific batch scorer callback."""

    def __call__(
        self,
        target_id: str,
        smiles_list: list[str],
        sequence: str,
        msa_path: Path | None,
        *,
        device: str | None,
    ) -> list[float]:
        """Score one sub-batch.

        Args:
            target_id: Target identifier.
            smiles_list: SMILES list for the sub-batch.
            sequence: Target protein sequence.
            msa_path: Optional precomputed MSA CSV path.
            device: Optional CUDA device token.

        Returns:
            List of affinity scores aligned with ``smiles_list``.
        """


def score_single(  # noqa: PLR0913
    target_id: str,
    smiles: str,
    *,
    backend: str,
    max_ligand_atoms: int,
    boltz_timeout: int,
    validate_smiles: Callable[[str], bool],
    load_sequence: Callable[[str], str],
    neutralize_smiles: Callable[[str], str],
    nim_predict_affinity: Callable[[str, str], float],
    require_msa_path: Callable[[str], Path],
    write_boltz_yaml: _WriteYamlFn,
    run_boltz_predict: _RunPredictFn,
    parse_affinity: Callable[[Path, str], float],
    log: _LoggerLike,
) -> float:
    """Score one molecule against one target.

    Args:
        target_id: Target identifier.
        smiles: Input SMILES string.
        backend: Active backend (``"cli"`` or ``"nim"``).
        max_ligand_atoms: Max atom threshold used in warning message.
        boltz_timeout: Timeout used for CLI subprocess in seconds.
        validate_smiles: SMILES validation callback.
        load_sequence: FASTA loading callback.
        neutralize_smiles: SMILES neutralization callback.
        nim_predict_affinity: NIM prediction callback.
        require_msa_path: MSA lookup callback.
        write_boltz_yaml: YAML writer callback.
        run_boltz_predict: CLI predictor callback.
        parse_affinity: Affinity parser callback.
        log: Logger-like instance.

    Returns:
        Predicted affinity as log10(IC50 in uM), or ``NaN`` on failures.
    """
    if not validate_smiles(smiles):
        log.warning(f"Invalid SMILES or molecule exceeds {max_ligand_atoms} atoms: {smiles}")
        return float("nan")

    sequence = load_sequence(target_id)

    if backend == "nim":
        smiles_neutralized = neutralize_smiles(smiles)
        return nim_predict_affinity(sequence, smiles_neutralized)

    msa_path = require_msa_path(target_id)

    with tempfile.TemporaryDirectory(prefix=f"boltz_{target_id}_") as tmpdir:
        tmp_path = Path(tmpdir)
        yaml_stem = "input"
        yaml_path = tmp_path / f"{yaml_stem}.yaml"
        out_dir = tmp_path / "output"

        smiles_neutralized = neutralize_smiles(smiles)
        write_boltz_yaml(sequence, smiles_neutralized, yaml_path, msa_path=msa_path)

        try:
            result = run_boltz_predict(yaml_path, out_dir, device=None)
        except subprocess.TimeoutExpired:
            log.error(f"Boltz timeout ({boltz_timeout} sec) for SMILES: {smiles}")
            return float("nan")

        if result.returncode != 0:
            stderr = result.stderr or ""
            if "CUDA out of memory" in stderr:
                log.error(f"Boltz GPU OOM for SMILES: {smiles}")
            else:
                log.error(f"Boltz error (rc={result.returncode}) for SMILES: {smiles}\nstderr: {stderr[-2000:]}")
            return float("nan")

        try:
            affinity = parse_affinity(out_dir, yaml_stem)
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
            log.error(f"Failed to parse affinity for SMILES {smiles}: {exc}")
            return float("nan")

    return affinity


def score_batch(  # noqa: PLR0913
    target_id: str,
    smiles_list: list[str],
    *,
    device: str | None,
    use_cache: bool,
    batch_size: int,
    backend: str,
    load_sequence: Callable[[str], str],
    require_msa_path: Callable[[str], Path],
    get_msa_path: Callable[[str], Path | None],
    sequence_content_hash: Callable[[str], str],
    msa_content_hash: Callable[[Path | None], str],
    scoring_context_hash: Callable[[], str],
    load_cache: Callable[[str], dict[str, float]],
    save_cache: Callable[[str, dict[str, float]], None],
    cache_key: _CacheKeyFn,
    score_batch_internal: _ScoreBatchInternalFn,
    log: _LoggerLike,
) -> NDArray[np.float64]:
    """Score a SMILES batch with cache-aware sub-batching.

    Args:
        target_id: Target identifier.
        smiles_list: Input SMILES strings.
        device: Optional CUDA device token.
        use_cache: Whether to read/write disk cache.
        batch_size: Number of molecules per sub-batch.
        backend: Active backend (``"cli"`` or ``"nim"``).
        load_sequence: FASTA loading callback.
        require_msa_path: MSA lookup callback (raises if missing).
        get_msa_path: Optional MSA path lookup (returns ``None`` if absent).
        sequence_content_hash: Sequence hashing callback.
        msa_content_hash: MSA hashing callback.
        scoring_context_hash: Context-hash callback.
        load_cache: Per-target cache loading callback.
        save_cache: Per-target cache save callback.
        cache_key: Cache-key callback.
        score_batch_internal: Backend-specific scorer callback.
        log: Logger-like instance.

    Returns:
        Numpy array of scores aligned to ``smiles_list``.
    """
    sequence = load_sequence(target_id)
    msa_path = get_msa_path(target_id) if backend == "nim" else require_msa_path(target_id)

    seq_hash = sequence_content_hash(sequence)
    msa_hash = msa_content_hash(msa_path)
    ctx = scoring_context_hash()

    cache = load_cache(target_id) if use_cache else {}
    scores: list[float] = [float("nan")] * len(smiles_list)
    to_score: list[tuple[int, str]] = []

    for idx, smi in enumerate(smiles_list):
        key = cache_key(target_id, smi, ctx_hash=ctx, seq_hash=seq_hash, msa_hash=msa_hash)
        if key in cache:
            scores[idx] = cache[key]
        else:
            to_score.append((idx, smi))

    if not to_score:
        log.info(f"All {len(smiles_list)} molecules for target {target_id} found in cache")
        return np.array(scores, dtype=np.float64)

    log.info(f"Scoring {len(to_score)} new molecules for target {target_id} ({len(smiles_list)} total, batch_size={batch_size})")

    num_batches = (len(to_score) + batch_size - 1) // batch_size
    scoring_start = time.monotonic()
    scored_so_far = 0

    for batch_idx in tqdm(range(num_batches), desc=f"Boltz-2 [{target_id}]", unit="batch"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(to_score))
        batch = to_score[start:end]
        batch_smiles = [smi for _, smi in batch]

        batch_t0 = time.monotonic()
        batch_scores = score_batch_internal(target_id, batch_smiles, sequence, msa_path, device=device)
        batch_elapsed = time.monotonic() - batch_t0

        batch_ok = 0
        for i, (idx, smi) in enumerate(batch):
            score = batch_scores[i]
            scores[idx] = score
            if not np.isnan(score):
                key = cache_key(target_id, smi, ctx_hash=ctx, seq_hash=seq_hash, msa_hash=msa_hash)
                cache[key] = score
                batch_ok += 1

        scored_so_far += len(batch)
        total_elapsed = time.monotonic() - scoring_start
        mol_per_sec = scored_so_far / total_elapsed if total_elapsed > 0 else 0
        remaining_mol = len(to_score) - scored_so_far
        eta_min = (remaining_mol / mol_per_sec / 60) if mol_per_sec > 0 else 0
        log.info(
            f"[Boltz-2] [{target_id}] Batch {batch_idx + 1}/{num_batches} - "
            f"{batch_ok}/{len(batch)} scored, {batch_elapsed:.0f}s, {mol_per_sec:.2f} mol/s, ETA ~{eta_min:.0f}m",
        )

        if use_cache:
            save_cache(target_id, cache)

    return np.array(scores, dtype=np.float64)
