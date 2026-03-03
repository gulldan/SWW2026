"""Hardware detection helpers shared by scoring and generation modules."""

from __future__ import annotations

import shutil
import subprocess


def detect_primary_gpu_vram_gb(*, nvidia_smi_timeout: int = 10) -> float | None:
    """Detect VRAM (GiB) of GPU 0.

    Detection order:
    1. ``torch.cuda.get_device_properties(0)``
    2. ``nvidia-smi --query-gpu=memory.total``

    Args:
        nvidia_smi_timeout: Timeout in seconds for the ``nvidia-smi`` call.

    Returns:
        VRAM in GiB, or ``None`` if detection fails.
    """
    vram_gb: float | None = None

    try:
        import torch

        if torch.cuda.is_available():
            vram_gb = float(torch.cuda.get_device_properties(0).total_mem) / (1024**3)
    except Exception:
        vram_gb = None

    if vram_gb is not None:
        return vram_gb

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    try:
        result = subprocess.run(  # noqa: S603
            [
                nvidia_smi,
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=nvidia_smi_timeout,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        first_line = result.stdout.strip().splitlines()[0]
        return float(first_line) / 1024.0
    except Exception:
        return None
