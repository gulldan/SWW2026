"""Shared fixtures for test suite."""

import pytest


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reduce bootstrap iterations for fast tests (10K/5K -> 100)."""
    import scripts.analysis.compute_metrics as cm
    import scripts.analysis.proxy_final_correlation as pfc
    import scripts.analysis.threshold_sensitivity as ts

    monkeypatch.setattr(cm, "_N_BOOTSTRAP", 100)
    monkeypatch.setattr(ts, "_N_BOOTSTRAP", 100)
    monkeypatch.setattr(pfc, "_N_BOOTSTRAP", 100)
