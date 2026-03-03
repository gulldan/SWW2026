"""Shared NIM endpoint health-check helpers."""

from __future__ import annotations

import urllib.request


def nim_url_ready(
    url: str,
    *,
    health_endpoint: str = "/v1/health/ready",
    timeout: float = 2.0,
) -> bool:
    """Return whether a NIM URL responds as ready.

    Args:
        url: NIM base URL (e.g. ``"http://localhost:8000"``).
        health_endpoint: Health endpoint path.
        timeout: Request timeout in seconds.

    Returns:
        ``True`` when response indicates readiness.
    """
    health_url = f"{url.rstrip('/')}{health_endpoint}"
    try:
        req = urllib.request.Request(health_url)  # noqa: S310
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read().lower()
        if b"false" in body:
            return False
        return b"true" in body or b"ready" in body or len(body) == 0
    except Exception:
        return False
