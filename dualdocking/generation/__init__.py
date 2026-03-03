"""Pluggable molecule generation backends.

Two built-in backends are available:

FREED++ (``freedpp``)
    Docker-based fragment RL generator (Python 3.7 / PyTorch 1.12 / DGL 0.9).
    UniDock GPU docking provides the RL reward.

REINVENT4 (``reinvent4``)
    SMILES-based RL generator running natively in Python 3.12.
    Uses DAP learning strategy with UniDock docking for RL reward.

The active backend is selected via ``generator.name`` in ``experiment.yaml``.
Additional backends can be registered via :func:`backends.register_generation_backend`
or loaded dynamically from ``generator.backend`` module specs.

Subpackages
-----------
backends
    Backend registry, resolution, and built-in adapter functions.
freedpp/
    Docker subprocess wrapper for FREED++ training and molecule extraction.
    core, docker, io, model submodules.
reinvent4/
    Native REINVENT4 training and molecule extraction.
    core, config, scoring submodules.
"""
