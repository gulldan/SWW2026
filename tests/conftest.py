"""Shared fixtures for test suite."""

import pytest


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reduce bootstrap iterations for fast tests (10K/5K -> 100)."""
    import dualdocking.analysis.metrics.computation as comp
    import dualdocking.analysis.metrics.correlation as corr
    import dualdocking.analysis.metrics.sensitivity as sens

    for mod in (comp, sens, corr):
        monkeypatch.setattr(mod, "_N_BOOTSTRAP", 100)
