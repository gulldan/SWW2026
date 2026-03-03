"""Dual-target reward patch for FREED++.

This patch does three things:
1) replaces ligand preparation backend (OpenBabel) with RDKit+Meeko,
2) applies z-score based rewards via shared reward-mode registry,
3) uses UniDock GPU batch docking (``--gpu_batch``) instead of per-ligand CPU calls.

Box parameters for the second receptor are read from ``/workspace/dual_config.json``::

    {"receptor_b": "/workspace/receptor_b.pdbqt", "center_b": "12.3,4.5,6.7", "size_b": "20,20,20"}

The implementation is split across helper modules:
- ``dual_reward_patch_deps.py``: compatibility imports,
- ``dual_reward_patch_state.py``: runtime state, context loading, metadata,
- ``dual_reward_patch_gpu.py``: GPU batch precompute,
- ``dual_reward_patch_hooks.py``: DockingVina monkey patches.

This facade delegates attribute lookups to the submodules via ``__getattr__``,
so tests and tooling can access internal symbols without an explicit re-export
list.  Only the public entry points are defined here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import dual_reward_patch_gpu as _gpu
import dual_reward_patch_hooks as _hooks
import dual_reward_patch_state as _state

if TYPE_CHECKING:
    from types import ModuleType

_SUBMODULES: tuple[ModuleType, ...] = (_state, _gpu, _hooks)


def _write_reward_metadata(*, path: Path | None = None) -> None:
    """Write reward metadata to JSON.

    Args:
        path: Explicit output path.  When ``None`` (default / production),
              uses the ``REWARD_METADATA_PATH`` environment variable.
    """
    _state._write_reward_metadata(path=path)  # noqa: SLF001


def _apply_dual_patch() -> None:
    """Monkey-patch DockingVina for dual-target reward calculation."""
    _hooks._apply_dual_patch()  # noqa: SLF001


def __getattr__(name: str) -> object:
    """Lazy attribute dispatch to submodules.

    Allows tests and tooling to access internal symbols
    (``_capped_zscore``, ``_call_stats``, etc.) without maintaining an
    explicit re-export list in the facade.
    """
    for submod in _SUBMODULES:
        try:
            return getattr(submod, name)
        except AttributeError:
            continue
    raise AttributeError(f"module 'dual_reward_patch' has no attribute {name!r}")


# Auto-apply on import
_apply_dual_patch()
