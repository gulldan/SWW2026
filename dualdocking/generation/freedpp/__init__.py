"""FREED++ Docker-based fragment RL generator.

Re-exports public API from submodules for backward compatibility.
"""

from dualdocking.generation.freedpp.core import (
    CACHE_DIR,
    DATA_DIR,
    GenerationConfig,
    _build_freedpp_args,
    _cache_key,
    _ensure_docker_image,
    _parse_output_smiles,
    _prepare_receptor,
    _resolve_pdb_path,
    _run_freedpp,
    generate,
    parse_training_log,
)

__all__ = [
    "CACHE_DIR",
    "DATA_DIR",
    "GenerationConfig",
    "_build_freedpp_args",
    "_cache_key",
    "_ensure_docker_image",
    "_parse_output_smiles",
    "_prepare_receptor",
    "_resolve_pdb_path",
    "_run_freedpp",
    "generate",
    "parse_training_log",
]
