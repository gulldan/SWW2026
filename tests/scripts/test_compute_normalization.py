"""Tests for cli/s2_calibrate/compute_normalization.py."""

from __future__ import annotations

import cli.s2_calibrate.compute_normalization as mod


def test_docking_version_fallback_is_unknown() -> None:
    """Version fallback should not execute external binaries."""
    assert mod._docking_version("unidock") == "unknown"
