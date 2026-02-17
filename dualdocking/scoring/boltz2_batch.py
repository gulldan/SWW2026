"""Batch-scoring orchestration for Boltz-2 backends."""

from __future__ import annotations

import concurrent.futures
import subprocess
import tempfile
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Protocol, cast

import numpy as np


class _LoggerLike(Protocol):
    """Minimal logger protocol for batch helpers."""

    def debug(self, message: str, *args: object) -> None:
        """Log a debug message.

        Args:
            message: Log template or text.
            *args: Optional formatting arguments.
        """

    def info(self, message: str, *args: object) -> None:
        """Log an informational message.

        Args:
            message: Log template or text.
            *args: Optional formatting arguments.
        """

    def error(self, message: str, *args: object) -> None:
        """Log an error message.

        Args:
            message: Log template or text.
            *args: Optional formatting arguments.
        """


class _RunBatchPredictFn(Protocol):
    """Protocol for ``boltz predict <dir>`` runner."""

    def __call__(
        self,
        input_dir: Path,
        out_dir: Path,
        *,
        timeout: int,
        device: str | None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a batched Boltz predictor subprocess.

        Args:
            input_dir: Directory with YAML inputs.
            out_dir: Directory for outputs.
            timeout: Timeout in seconds.
            device: Optional CUDA device token.

        Returns:
            Completed subprocess result.
        """


class _WriteYamlFn(Protocol):
    """Protocol for Boltz YAML writer."""

    def __call__(
        self,
        sequence: str,
        smiles: str,
        path: Path,
        *,
        msa_path: Path | None,
    ) -> None:
        """Write a single Boltz YAML input file.

        Args:
            sequence: Target amino-acid sequence.
            smiles: Ligand SMILES.
            path: Destination YAML path.
            msa_path: Optional precomputed MSA CSV.
        """


class _ScoreSingleFn(Protocol):
    """Protocol for single-molecule fallback scorer."""

    def __call__(
        self,
        target_id: str,
        smiles: str,
        sequence: str,
        msa_path: Path | None,
        *,
        device: str | None,
    ) -> float:
        """Score one molecule with fallback path.

        Args:
            target_id: Target identifier.
            smiles: Neutralized SMILES.
            sequence: Target sequence.
            msa_path: Optional MSA path.
            device: Optional CUDA device token.

        Returns:
            Predicted affinity value or NaN on failure.
        """


class _CollectSmilesFn(Protocol):
    """Protocol for validated/neutralized SMILES collector."""

    def __call__(
        self,
        smiles_list: list[str],
        *,
        warning_template: str,
    ) -> list[tuple[int, str]]:
        """Validate and neutralize input SMILES strings.

        Args:
            smiles_list: Raw input SMILES.
            warning_template: Warning template for invalid inputs.

        Returns:
            List of ``(index, neutralized_smiles)`` tuples.
        """


class _PredictBatchAsyncFn(Protocol):
    """Protocol for async NIM batch predictor."""

    def __call__(
        self,
        sequence: str,
        smiles_indices: list[tuple[int, str]],
        *,
        nim_urls: list[str] | None,
    ) -> Coroutine[object, object, list[tuple[int, float]]]:
        """Run async NIM batch prediction.

        Args:
            sequence: Target amino-acid sequence.
            smiles_indices: ``(index, neutralized_smiles)`` tuples.
            nim_urls: Explicit NIM URL pool.

        Returns:
            Coroutine that resolves to ``(index, score)`` tuples.
        """


def score_batch_nim(
    smiles_list: list[str],
    sequence: str,
    *,
    collect_valid_smiles: _CollectSmilesFn,
    resolve_nim_urls: Callable[[], list[str]],
    predict_batch_async: _PredictBatchAsyncFn,
    nim_concurrency: int,
    log: _LoggerLike,
) -> list[float]:
    """Score molecules via NIM backend with async round-robin requests.

    Args:
        smiles_list: Raw input SMILES in original order.
        sequence: Target sequence.
        collect_valid_smiles: Validation/neutralization callback.
        resolve_nim_urls: Active NIM URL resolver.
        predict_batch_async: Async batch predictor function.
        nim_concurrency: Per-instance concurrency cap.
        log: Logger-like instance.

    Returns:
        Score list aligned with ``smiles_list``. Invalid/failed entries are NaN.
    """
    import asyncio

    scores = [float("nan")] * len(smiles_list)
    valid_pairs = collect_valid_smiles(
        smiles_list,
        warning_template="Invalid SMILES or too large (>{} atoms): {}",
    )
    if not valid_pairs:
        return scores

    nim_urls = resolve_nim_urls()
    log.debug(f"NIM async batch: {len(valid_pairs)} molecules, {len(nim_urls)} NIM URL(s), concurrency={nim_concurrency}")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    results: list[tuple[int, float]]
    if loop is not None and loop.is_running():
        coro = predict_batch_async(sequence, valid_pairs, nim_urls=nim_urls)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            results = cast(
                "list[tuple[int, float]]",
                pool.submit(asyncio.run, coro).result(),
            )
    else:
        results = asyncio.run(predict_batch_async(sequence, valid_pairs, nim_urls=nim_urls))

    for idx, score in results:
        scores[idx] = score
    return scores


def score_batch_cli(  # noqa: PLR0913
    target_id: str,
    smiles_list: list[str],
    sequence: str,
    msa_path: Path | None,
    *,
    device: str | None,
    batch_timeout: int,
    collect_valid_smiles: _CollectSmilesFn,
    write_boltz_yaml: _WriteYamlFn,
    run_boltz_predict_dir: _RunBatchPredictFn,
    parse_affinity_file: Callable[[Path], float],
    score_single_internal: _ScoreSingleFn,
    log: _LoggerLike,
) -> list[float]:
    """Score molecules via CLI backend with partial recovery and fallback.

    Args:
        target_id: Target identifier.
        smiles_list: Raw input SMILES in original order.
        sequence: Target sequence.
        msa_path: Optional MSA path.
        device: Optional CUDA device token.
        batch_timeout: Timeout in seconds for one batched CLI run.
        collect_valid_smiles: Validation/neutralization callback.
        write_boltz_yaml: YAML writer callback.
        run_boltz_predict_dir: Batched CLI runner callback.
        parse_affinity_file: Affinity JSON parser callback.
        score_single_internal: Single-molecule fallback scorer callback.
        log: Logger-like instance.

    Returns:
        Score list aligned with ``smiles_list``. Invalid/failed entries are NaN.
    """
    scores = [float("nan")] * len(smiles_list)
    valid_pairs = collect_valid_smiles(
        smiles_list,
        warning_template="Invalid SMILES or molecule exceeds {} atoms: {}",
    )
    if not valid_pairs:
        return scores
    neutralized_by_idx = dict(valid_pairs)

    batch_failed = False
    with tempfile.TemporaryDirectory(prefix=f"boltz_batch_{target_id}_") as tmpdir:
        tmp_path = Path(tmpdir)
        input_dir = tmp_path / "inputs"
        out_dir = tmp_path / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        stem_to_idx: dict[str, int] = {}
        for idx, neutralized in valid_pairs:
            stem = f"mol_{idx:04d}"
            write_boltz_yaml(sequence, neutralized, input_dir / f"{stem}.yaml", msa_path=msa_path)
            stem_to_idx[stem] = idx

        try:
            result = run_boltz_predict_dir(input_dir, out_dir, timeout=batch_timeout, device=device)
        except subprocess.TimeoutExpired:
            log.error(f"Boltz batch timeout ({batch_timeout} sec) for {len(valid_pairs)} molecules, parsing partial results")
            batch_failed = True
        else:
            if result.returncode != 0:
                stderr = result.stderr or ""
                if "CUDA out of memory" in stderr:
                    log.error(f"Boltz batch GPU OOM for {len(valid_pairs)} molecules")
                else:
                    log.error(
                        f"Boltz batch error (rc={result.returncode}) for {len(valid_pairs)} molecules\nstderr: {stderr[-2000:]}",
                    )
                batch_failed = True

        results_dir = out_dir / f"boltz_results_{input_dir.name}"
        parsed_ok = 0
        for stem, idx in stem_to_idx.items():
            affinity_path = results_dir / "predictions" / stem / f"affinity_{stem}.json"
            try:
                scores[idx] = parse_affinity_file(affinity_path)
                parsed_ok += 1
            except FileNotFoundError:
                continue
            except (KeyError, ValueError):  # ValueError covers malformed numeric fields.
                log.debug(f"Failed to parse affinity for {stem}")

        if batch_failed:
            log.info(f"Batch partial recovery: {parsed_ok}/{len(valid_pairs)} molecules scored from failed batch")

    if batch_failed:
        failed_indices = [idx for idx, _ in valid_pairs if np.isnan(scores[idx])]
        if failed_indices:
            log.info(f"Batch fallback: retrying {len(failed_indices)} NaN molecules individually for {target_id}")
            for idx in failed_indices:
                scores[idx] = score_single_internal(
                    target_id,
                    neutralized_by_idx[idx],
                    sequence,
                    msa_path,
                    device=device,
                )

    return scores
