"""NIM backend request routines for Boltz-2 scoring."""

from __future__ import annotations

import asyncio
import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

import aiohttp


class _LoggerLike(Protocol):
    """Minimal logger protocol for backend routines."""

    def warning(self, message: str, *args: object) -> None:
        """Log warning message.

        Args:
            message: Log message template/text.
            *args: Optional formatting arguments.
        """

    def error(self, message: str, *args: object) -> None:
        """Log error message.

        Args:
            message: Log message template/text.
            *args: Optional formatting arguments.
        """


def predict_affinity_sync(
    sequence: str,
    smiles: str,
    *,
    nim_urls: list[str],
    predict_endpoint: str,
    request_timeout: float,
    max_retries: int,
    payload_builder: Callable[[str, str, str | None], dict[str, Any]],
    parse_affinity: Callable[[dict[str, Any], str], float],
    log: _LoggerLike,
    msa_content: str | None = None,
) -> float:
    """Score one molecule via synchronous NIM HTTP requests.

    Args:
        sequence: Protein sequence.
        smiles: Ligand SMILES.
        nim_urls: Active NIM URL pool.
        predict_endpoint: NIM prediction endpoint path.
        request_timeout: Request timeout in seconds.
        max_retries: Number of retry attempts after the initial request.
        payload_builder: Function building request payload.
        parse_affinity: Function parsing response affinity.
        log: Logger-like instance.
        msa_content: Optional raw CSV content from ColabFold MSA file.

    Returns:
        Predicted affinity score, or NaN on failure.
    """
    n_urls = len(nim_urls)
    data = json.dumps(payload_builder(sequence, smiles, msa_content)).encode()

    start_idx = random.randrange(n_urls) if n_urls > 1 else 0  # noqa: S311
    url_order = [(start_idx + i) % n_urls for i in range(max(n_urls, max_retries + 1))]

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        idx = url_order[attempt % len(url_order)]
        url = f"{nim_urls[idx]}{predict_endpoint}"
        req = urllib.request.Request(  # noqa: S310
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:  # noqa: S310
                result = json.loads(resp.read().decode())
            return parse_affinity(result, smiles)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:500] if exc.fp else ""
            if 400 <= exc.code < 500:
                log.error(f"NIM HTTP {exc.code} for SMILES {smiles}: {body}")
                return float("nan")
            last_error = exc
            log.warning(
                f"NIM HTTP {exc.code} for SMILES {smiles} (attempt {attempt + 1}/{max_retries + 1}, url={nim_urls[idx]}): {body}",
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            log.warning(
                f"NIM request failed for SMILES {smiles} (attempt {attempt + 1}/{max_retries + 1}, url={nim_urls[idx]}): {exc}",
            )

        if attempt < max_retries:
            time.sleep(1.0 * (attempt + 1))

    log.error(f"NIM exhausted {max_retries + 1} retries for SMILES {smiles}: {last_error}")
    return float("nan")


async def predict_batch_async(
    sequence: str,
    smiles_indices: list[tuple[int, str]],
    *,
    nim_urls: list[str],
    predict_endpoint: str,
    request_timeout: float,
    concurrency: int,
    max_retries: int,
    payload_builder: Callable[[str, str, str | None], dict[str, Any]],
    parse_affinity: Callable[[dict[str, Any], str], float],
    log: _LoggerLike,
    msa_content: str | None = None,
) -> list[tuple[int, float]]:
    """Score a batch through async round-robin NIM requests.

    Args:
        sequence: Protein sequence.
        smiles_indices: ``(index, neutralized_smiles)`` tuples.
        nim_urls: Active NIM URL pool.
        predict_endpoint: NIM prediction endpoint path.
        request_timeout: Request timeout in seconds.
        concurrency: Per-instance concurrency limit.
        max_retries: Number of retry attempts after initial request.
        payload_builder: Function building request payload.
        parse_affinity: Function parsing response affinity.
        log: Logger-like instance.
        msa_content: Optional raw CSV content from ColabFold MSA file.

    Returns:
        List of ``(index, affinity)`` aligned to submitted indices.
    """
    n_urls = len(nim_urls)
    urls = [f"{url}{predict_endpoint}" for url in nim_urls]
    semaphores = [asyncio.Semaphore(concurrency) for _ in range(n_urls)]
    client_timeout = aiohttp.ClientTimeout(total=request_timeout)

    async def _predict_one(
        session: aiohttp.ClientSession,
        idx: int,
        smiles: str,
        url_idx: int,
    ) -> tuple[int, float]:
        """Score one molecule with URL-rotating retries.

        Args:
            session: Shared aiohttp session.
            idx: Original index in caller batch.
            smiles: Neutralized SMILES.
            url_idx: Preferred initial URL index.

        Returns:
            Tuple of ``(idx, score)``.
        """
        payload = payload_builder(sequence, smiles, msa_content)
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            current_idx = url_idx if attempt == 0 else (url_idx + attempt) % n_urls
            async with semaphores[current_idx]:
                try:
                    async with session.post(urls[current_idx], json=payload) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            return idx, parse_affinity(result, smiles)

                        body = (await resp.text())[:500]
                        if resp.status >= 400 and resp.status < 500 and "GPU available" not in body:
                            log.error(f"NIM HTTP {resp.status} for SMILES {smiles}: {body}")
                            return idx, float("nan")

                        last_error = Exception(f"HTTP {resp.status}: {body}")
                        log.warning(
                            f"NIM HTTP {resp.status} for SMILES {smiles} "
                            f"(attempt {attempt + 1}/{max_retries + 1}, url={nim_urls[current_idx]}): {body[:100]}",
                        )
                except (TimeoutError, aiohttp.ClientError, OSError) as exc:
                    last_error = exc
                    log.warning(
                        f"NIM async request failed for SMILES {smiles} "
                        f"(attempt {attempt + 1}/{max_retries + 1}, url={nim_urls[current_idx]}): {exc}",
                    )

            if attempt < max_retries:
                await asyncio.sleep(1.0 * (attempt + 1))

        log.error(f"NIM exhausted {max_retries + 1} retries for SMILES {smiles}: {last_error}")
        return idx, float("nan")

    connector = aiohttp.TCPConnector(limit=concurrency * n_urls, limit_per_host=concurrency)
    async with aiohttp.ClientSession(timeout=client_timeout, connector=connector) as session:
        tasks: list[Awaitable[tuple[int, float]]] = []
        for i, (idx, smiles) in enumerate(smiles_indices):
            tasks.append(_predict_one(session, idx, smiles, i % n_urls))
        return list(await asyncio.gather(*tasks))
