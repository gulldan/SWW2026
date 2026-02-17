"""Shared HTTP download helpers with retry, streaming, and SHA-256 checksums.

Used by Stage 0 fetch scripts and ``chembl_validation`` to provide
consistent retry / backoff behaviour across all network calls.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import requests
from loguru import logger

from dualdocking.utils.hash_utils import sha256_file as _sha256_file

_DEFAULT_RETRY_ATTEMPTS: int = 3
_DEFAULT_RETRY_BACKOFF: int = 5  # seconds (multiplied by attempt number)
_DEFAULT_TIMEOUT: int = 60


def get_with_retry(
    url: str,
    *,
    attempts: int = _DEFAULT_RETRY_ATTEMPTS,
    backoff: int = _DEFAULT_RETRY_BACKOFF,
    timeout: int = _DEFAULT_TIMEOUT,
) -> requests.Response:
    """HTTP GET with retry and linear backoff.

    Args:
        url: Fully-qualified URL to fetch.
        attempts: Maximum number of attempts before raising.
        backoff: Base backoff in seconds (multiplied by attempt number).
        timeout: HTTP request timeout in seconds.

    Returns:
        Successful :class:`requests.Response`.

    Raises:
        requests.RequestException: If all retry attempts fail.
    """
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.RequestException, requests.HTTPError) as exc:
            if attempt == attempts:
                raise
            logger.warning("HTTP GET attempt {}/{} failed for {}: {}", attempt, attempts, url, exc)
            time.sleep(backoff * attempt)
    raise RuntimeError("Unreachable")  # pragma: no cover


def download_file(
    url: str,
    dest: Path,
    *,
    timeout: int = _DEFAULT_TIMEOUT,
    attempts: int = _DEFAULT_RETRY_ATTEMPTS,
    backoff: int = _DEFAULT_RETRY_BACKOFF,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Stream-download a URL to a local file with retry and SHA-256.

    Args:
        url: Remote URL to GET.
        dest: Local destination path (parent directories created automatically).
        timeout: HTTP request timeout in seconds.
        attempts: Maximum number of download attempts.
        backoff: Base backoff in seconds between retries.
        chunk_size: Streaming chunk size in bytes.

    Returns:
        Hex-encoded SHA-256 digest of the downloaded file.

    Raises:
        requests.RequestException: If all retry attempts fail.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, attempts + 1):
        try:
            logger.info("GET {} -> {} (attempt {}/{})", url, dest, attempt, attempts)
            if tmp_path.exists():
                tmp_path.unlink()

            digest = hashlib.sha256()
            with requests.get(url, timeout=timeout, stream=True) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            handle.write(chunk)
                            digest.update(chunk)

            tmp_path.replace(dest)
            sha = digest.hexdigest()
            logger.info("Saved {:.1f} KiB (SHA-256: {})", dest.stat().st_size / 1024, sha[:16])
            return sha
        except (requests.RequestException, OSError) as exc:
            if attempt == attempts:
                raise
            logger.warning("Download attempt {}/{} failed: {}", attempt, attempts, exc)
            time.sleep(backoff * attempt)

    raise RuntimeError("Unreachable")  # pragma: no cover


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return _sha256_file(path)
