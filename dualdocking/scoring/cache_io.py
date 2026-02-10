"""Process-safe JSON cache I/O with atomic writes and file locking.

Addresses three concurrency hazards:

1. **Cross-process races** -- ``fcntl.flock`` serialises concurrent writers
   launched by ``run_full_experiment.py --parallel N``.
2. **Corrupt writes** -- temp-file + ``os.replace`` is atomic on POSIX; a
   crash mid-write leaves the previous valid file intact.
3. **Lost updates** -- ``save_cache`` performs read-merge-write under the
   lock so entries from other processes are preserved.

Both ``load_cache`` and ``save_cache`` use a **fail-closed** strategy:
if the file lock cannot be acquired within the timeout, the operation is
aborted (load returns ``{}``, save skips the write).
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from loguru import logger

# In-process lock avoids redundant flock syscalls when multiple threads
# within the *same* process contend (e.g. ThreadPoolExecutor in tests).
_thread_lock = threading.Lock()

# Defaults — tunable via function kwargs.
LOCK_TIMEOUT: float = 30.0
_RETRY_INITIAL: float = 0.05
_RETRY_CAP: float = 1.0


def _acquire_flock(fd: int, timeout: float) -> bool:
    """Acquire an exclusive ``flock`` with exponential back-off.

    Args:
        fd: Open file descriptor to lock.
        timeout: Maximum seconds to wait before giving up.

    Returns:
        ``True`` if the lock was acquired, ``False`` on timeout.
    """
    deadline = time.monotonic() + timeout
    delay = _RETRY_INITIAL
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (OSError, BlockingIOError):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, _RETRY_CAP)


def _parse_dict(data: object) -> dict[str, float]:
    """Coerce *data* to ``dict[str, float]``, filtering out NaN values.

    Args:
        data: Parsed JSON value (expected to be a dict).

    Returns:
        Dict with string keys and finite float values; empty if *data*
        is not a dict.
    """
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, (int, float)) and not np.isnan(v)}


def _lock_path(cache_path: Path) -> Path:
    """Return the dedicated ``.lock`` sidecar path for *cache_path*.

    Args:
        cache_path: Path to the JSON cache file.

    Returns:
        Adjacent path with ``.lock`` suffix (avoids interference with
        ``os.replace``).
    """
    return cache_path.with_suffix(".lock")


def load_cache(path: Path, *, timeout: float = LOCK_TIMEOUT) -> dict[str, float]:
    """Load a JSON cache file under an exclusive file lock.

    Returns an empty dict when the file is missing, empty, corrupted, or
    when the lock cannot be acquired within *timeout* (fail-closed).
    NaN values are filtered out during parsing.

    Args:
        path: Path to the JSON cache file.
        timeout: Maximum seconds to wait for the file lock.

    Returns:
        Mapping of cache keys to float scores, or ``{}`` on any failure.
    """
    with _thread_lock:
        if not path.exists():
            return {}

        lock_file = _lock_path(path)
        lock_file.touch(exist_ok=True)
        lock_fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if not _acquire_flock(lock_fd, timeout):
                logger.warning("Lock timeout ({:.1f}s) reading cache: {}", timeout, path)
                return {}

            raw = path.read_text()
            if not raw.strip():
                return {}
            data = json.loads(raw)
            return _parse_dict(data)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupted cache file {}: {}", path, exc)
            return {}
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)


def save_cache(path: Path, cache: dict[str, float], *, timeout: float = LOCK_TIMEOUT) -> None:
    """Atomically merge *cache* into the on-disk JSON file.

    Algorithm (SAGA pattern):
      1. Acquire flock on ``<path>.lock``.
      2. Read current disk contents (if any).
      3. Merge: ``disk | cache`` (new entries win).
      4. Write merged data to a temp file in the same directory.
      5. ``os.fsync`` + ``os.replace`` (atomic on POSIX).
      6. Release flock.

    If the lock cannot be acquired within *timeout*, the write is
    **skipped** (fail-closed).  Compensating action: unlink temp on
    failure before rename.

    Args:
        path: Destination JSON cache file.
        cache: New entries to merge (keys are cache hashes, values are
            float scores).
        timeout: Maximum seconds to wait for the file lock.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    lock_file = _lock_path(path)
    lock_file.touch(exist_ok=True)
    lock_fd: int | None = None
    tmp_path: str | None = None

    with _thread_lock:
        try:
            lock_fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT, 0o644)
            if not _acquire_flock(lock_fd, timeout):
                logger.warning("Lock timeout ({:.1f}s) writing cache: {} — write skipped", timeout, path)
                return

            # --- Read current disk state (merge) ---
            disk: dict[str, float] = {}
            if path.exists():
                try:
                    raw = path.read_text()
                    if raw.strip():
                        disk_data = json.loads(raw)
                        disk = _parse_dict(disk_data)
                except (json.JSONDecodeError, OSError):
                    pass  # corrupted — start fresh

            # Merge: existing disk entries + new entries (new win on conflict)
            merged = {**disk, **_parse_dict(cache)}
            payload = json.dumps(merged, indent=2, sort_keys=True).encode()

            # --- Atomic write: temp -> fsync -> replace ---
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent),
                prefix=f".{path.stem}.",
                suffix=".tmp",
            )
            os.write(fd, payload)
            os.fsync(fd)
            os.close(fd)

            Path(tmp_path).replace(path)
            tmp_path = None  # rename succeeded — nothing to clean up

        except OSError as exc:
            logger.error("Failed to save cache {}: {}", path, exc)
        finally:
            # Compensating action: clean up temp on failure
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    Path(tmp_path).unlink()
            if lock_fd is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
