"""Shared fixtures for test suite."""

import pytest


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reduce bootstrap iterations for fast tests (10K/5K -> 100)."""
    import scripts.s5_analyze.compute_metrics as cm
    import scripts.s5_analyze.proxy_correlation as pfc
    import scripts.s5_analyze.threshold_sensitivity as ts

    monkeypatch.setattr(cm, "_N_BOOTSTRAP", 100)
    monkeypatch.setattr(ts, "_N_BOOTSTRAP", 100)
    monkeypatch.setattr(pfc, "_N_BOOTSTRAP", 100)
