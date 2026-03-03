"""Atomic I/O primitives with lock retry used by cache and manifest modules."""

from __future__ import annotations

import contextlib
import fcntl
import os
import tempfile
import time
from pathlib import Path


def acquire_flock(
    fd: int,
    timeout: float,
    *,
    retry_initial: float = 0.05,
    retry_cap: float = 1.0,
) -> bool:
    """Acquire an exclusive non-blocking flock with bounded retry.

    Args:
        fd: Open file descriptor to lock.
        timeout: Maximum seconds to wait.
        retry_initial: Initial sleep interval in seconds.
        retry_cap: Maximum sleep interval in seconds.

    Returns:
        ``True`` when lock is acquired, ``False`` on timeout.
    """
    deadline = time.monotonic() + timeout
    delay = max(retry_initial, 0.0)
    cap = max(retry_cap, delay)
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (OSError, BlockingIOError):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            sleep_for = min(delay, remaining) if delay > 0 else remaining
            time.sleep(sleep_for)
            delay = min(delay * 2 if delay > 0 else retry_initial, cap)


def write_bytes_atomic(
    path: Path,
    payload: bytes,
    *,
    prefix: str | None = None,
    suffix: str = ".tmp",
) -> None:
    """Atomically write bytes via temp file + fsync + replace.

    Args:
        path: Destination file path.
        payload: Raw bytes to persist.
        prefix: Optional temp-file prefix; defaults to ``.{path.stem}.``.
        suffix: Temp-file suffix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None
    fd: int | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=prefix or f".{path.stem}.",
            suffix=suffix,
        )
        os.write(fd, payload)
        os.fsync(fd)
        os.close(fd)
        fd = None
        Path(tmp_path).replace(path)
        tmp_path = None
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink()
