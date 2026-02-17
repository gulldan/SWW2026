"""Wrapper: conditionally imports dual_reward_patch before FREED++ main.

When running ``python /opt/run_with_patch.py``, Python sets sys.path[0]
to the script's directory (/opt), NOT the CWD (/opt/freed).  We must
add /opt/freed explicitly so that ``import ffreed`` works — both for
the dual_reward_patch and for runpy.run_module below.
"""

import os
import runpy
import sys
import warnings
from contextlib import suppress
from multiprocessing import set_start_method

# ffreed lives in /opt/freed — must be importable before anything else
sys.path.insert(0, "/opt/freed")

# DGL should not auto-select backend at runtime; enforce PyTorch before imports.
os.environ.setdefault("DGLBACKEND", "pytorch")

# DGL wheels can emit a noisy SyntaxWarning from a docstring escape.
# Keep logs focused on actionable runtime diagnostics.
warnings.filterwarnings(
    "ignore",
    message=r"invalid escape sequence '\\P'",
    category=SyntaxWarning,
)

# Some dependencies may create multiprocessing pools implicitly.
# Force `spawn` to avoid unsafe fork() in a multi-threaded process.
if os.environ.get("FREEDPP_FORCE_SPAWN", "1") != "0":
    with suppress(RuntimeError):
        set_start_method("spawn", force=True)

if os.environ.get("REWARD_MODE"):
    # /opt holds dual_reward_patch.py
    if "/opt" not in sys.path:
        sys.path.insert(0, "/opt")
    import dual_reward_patch  # noqa: F401 — applies monkey-patch on import

runpy.run_module("ffreed.main", run_name="__main__")
