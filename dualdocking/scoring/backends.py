"""Scoring backend registry and adapters.

This module decouples pipeline scripts from a hard dependency on Boltz-2 by
providing a minimal backend interface and config-driven backend resolution.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

from dualdocking.utils.naming import canonicalize_name


class ScoringBackend(Protocol):
    """Minimal interface required by scoring cli."""

    def score_batch(
        self,
        target_id: str,
        smiles_list: list[str],
        *,
        use_cache: bool = True,
        device: str | None = None,
    ) -> Any:
        """Score a list of molecules for a target."""

    def get_scoring_context_info(self) -> dict[str, Any]:
        """Return scorer context metadata."""

    def prune_stale_cache(self) -> int:
        """Prune stale cache entries and return removed file count."""


@dataclass(frozen=True)
class FunctionScoringBackend:
    """Adapter from plain functions to :class:`ScoringBackend`."""

    score_batch_fn: Any
    context_info_fn: Any
    prune_cache_fn: Any

    def score_batch(
        self,
        target_id: str,
        smiles_list: list[str],
        *,
        use_cache: bool = True,
        device: str | None = None,
    ) -> Any:
        """Dispatch to wrapped ``score_batch`` function."""
        return self.score_batch_fn(target_id, smiles_list, use_cache=use_cache, device=device)

    def get_scoring_context_info(self) -> dict[str, Any]:
        """Dispatch to wrapped metadata function."""
        return cast("dict[str, Any]", self.context_info_fn())

    def prune_stale_cache(self) -> int:
        """Dispatch to wrapped cache-pruning function."""
        return int(self.prune_cache_fn())


def _as_mapping(value: object) -> Mapping[str, Any]:
    """Return *value* as mapping or empty mapping when invalid."""
    if isinstance(value, Mapping):
        return cast("Mapping[str, Any]", value)
    return {}


def _canonical_engine_name(name: str) -> str:
    """Normalize scorer engine names to canonical built-in keys."""
    return canonicalize_name(name, {"boltz": "boltz2", "boltz2": "boltz2"})


def _resolve_nim_urls(scorer_cfg: Mapping[str, Any]) -> list[str]:
    """Extract NIM URLs from scorer config with sane defaults."""
    nim_cfg = _as_mapping(scorer_cfg.get("nim"))
    raw_urls = nim_cfg.get("urls")
    urls: list[str] = []
    if isinstance(raw_urls, list):
        urls.extend(item.strip() for item in raw_urls if isinstance(item, str) and item.strip())
    if not urls:
        urls = ["http://localhost:8000"]
    return urls


def _configure_boltz2_env(scorer_cfg: Mapping[str, Any]) -> None:
    """Project scorer config into Boltz-2 environment variables."""
    backend = str(scorer_cfg.get("backend", os.getenv("BOLTZ_BACKEND", "cli"))).lower()
    os.environ["BOLTZ_BACKEND"] = backend
    if backend == "nim":
        urls = _resolve_nim_urls(scorer_cfg)
        os.environ["BOLTZ_NIM_URLS"] = ",".join(urls)
        os.environ["BOLTZ_NIM_URL"] = urls[0]


def _set_module_attr(module: Any, attr: str, value: Any) -> None:
    """Set an attribute on a dynamically imported module."""
    setattr(module, attr, value)


def _load_boltz2_backend(scorer_cfg: Mapping[str, Any]) -> ScoringBackend:
    """Load built-in Boltz-2 backend with runtime configuration."""
    _configure_boltz2_env(scorer_cfg)
    module = importlib.import_module("dualdocking.scoring.boltz2")

    # Keep already-imported module in sync with runtime config.
    backend = os.environ.get("BOLTZ_BACKEND", "cli").lower()
    _set_module_attr(module, "_BOLTZ_BACKEND", backend)
    if backend == "nim":
        urls = [u.strip() for u in os.environ.get("BOLTZ_NIM_URLS", os.environ.get("BOLTZ_NIM_URL", "")).split(",") if u.strip()]
        if urls:
            _set_module_attr(module, "_BOLTZ_NIM_URLS", urls)
            _set_module_attr(module, "_BOLTZ_NIM_URL", urls[0])
        if hasattr(module, "_NIM_ACTIVE_URLS"):
            _set_module_attr(module, "_NIM_ACTIVE_URLS", None)
        if hasattr(module, "_NIM_ACTIVE_URLS_KEY"):
            _set_module_attr(module, "_NIM_ACTIVE_URLS_KEY", None)
    if hasattr(module, "_CONTEXT_HASH"):
        _set_module_attr(module, "_CONTEXT_HASH", None)

    return FunctionScoringBackend(
        score_batch_fn=module.score_batch,
        context_info_fn=module.get_scoring_context_info,
        prune_cache_fn=module.prune_stale_cache,
    )


def _looks_like_backend(obj: object) -> bool:
    """Return whether object provides required scoring backend attributes."""
    return all(callable(getattr(obj, attr, None)) for attr in ("score_batch", "get_scoring_context_info", "prune_stale_cache"))


def _load_external_backend(spec: str) -> ScoringBackend:
    """Load external scoring backend from ``module`` or ``module:attr`` spec."""
    module_path, _, attr = spec.partition(":")
    module = importlib.import_module(module_path)

    candidate: object
    if attr:
        candidate = getattr(module, attr, None)
        if candidate is None:
            msg = f"Scoring backend '{spec}' attribute '{attr}' not found"
            raise ValueError(msg)
    else:
        candidate = module

    if callable(candidate) and not _looks_like_backend(candidate):
        candidate = candidate()

    if not _looks_like_backend(candidate):
        msg = f"Scoring backend '{spec}' must provide callables 'score_batch', 'get_scoring_context_info', 'prune_stale_cache'"
        raise TypeError(msg)

    return cast("ScoringBackend", candidate)


def list_builtin_scoring_engines() -> list[str]:
    """List built-in scoring engines."""
    return ["boltz2"]


def resolve_scoring_backend(scorer_cfg: Mapping[str, Any] | None) -> tuple[str, ScoringBackend]:
    """Resolve scoring backend from ``experiment.yaml`` scorer section.

    Resolution order:
    1. ``scorer.engine``
    2. ``scorer.name`` (legacy)
    3. built-in ``boltz2``

    Non built-in values are treated as external backend specs:
    ``package.module`` (module implements required functions) or
    ``package.module:factory_or_object``.
    """
    cfg = scorer_cfg or {}
    raw_spec = cfg.get("engine", cfg.get("name", "boltz2"))
    spec = str(raw_spec).strip() if raw_spec is not None else "boltz2"
    canonical = _canonical_engine_name(spec)

    if canonical == "boltz2":
        return "boltz2", _load_boltz2_backend(cfg)
    return spec, _load_external_backend(spec)
