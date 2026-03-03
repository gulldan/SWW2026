"""Shared low-level utilities used across multiple DualDocking subpackages."""

from dualdocking.utils.hardware import detect_primary_gpu_vram_gb
from dualdocking.utils.hash_utils import sha256_file
from dualdocking.utils.io_atomic import acquire_flock, write_bytes_atomic
from dualdocking.utils.naming import canonicalize_name

__all__ = [
    "acquire_flock",
    "canonicalize_name",
    "detect_primary_gpu_vram_gb",
    "sha256_file",
    "write_bytes_atomic",
]
