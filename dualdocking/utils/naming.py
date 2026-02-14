"""Canonical naming helpers for config-driven engine/backend aliases."""

from __future__ import annotations

from collections.abc import Mapping


def canonicalize_name(name: str, alias_map: Mapping[str, str] | None = None) -> str:
    """Normalize a user-provided name via alnum-folding + alias map.

    Args:
        name: Raw name from config/CLI (e.g. ``"Boltz-2"``).
        alias_map: Mapping from normalized token to canonical name.

    Returns:
        Canonical mapped name when alias exists; otherwise stripped original name.
    """
    normalized = "".join(ch for ch in name.lower() if ch.isalnum())
    if alias_map and normalized in alias_map:
        return alias_map[normalized]
    return name.strip()
