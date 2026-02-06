"""Wrapper: conditionally imports dual_reward_patch before FREED++ main.

When running ``python /opt/run_with_patch.py``, Python sets sys.path[0]
to the script's directory (/opt), NOT the CWD (/opt/freed).  We must
add /opt/freed explicitly so that ``import ffreed`` works — both for
the dual_reward_patch and for runpy.run_module below.
"""

import os
import sys

# ffreed lives in /opt/freed — must be importable before anything else
sys.path.insert(0, "/opt/freed")

if os.environ.get("REWARD_MODE"):
    # /opt holds dual_reward_patch.py
    if "/opt" not in sys.path:
        sys.path.insert(0, "/opt")
    import dual_reward_patch  # noqa: F401 — applies monkey-patch on import

import runpy

runpy.run_module("ffreed.main", run_name="__main__")
