"""Tests for dual_reward_patch — invalid docking exclusion (ENG-01 / KVA-255).

The patch file lives in docker/freedpp/ and auto-applies on import (requires
ffreed on sys.path).  We add its directory to sys.path and mock ffreed so
that the standalone functions (_capped_zscore, _write_reward_metadata, etc.)
can be tested without Docker.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PATCH_DIR = str(Path(__file__).resolve().parent.parent.parent / "docker" / "freedpp")

_NORMALIZATION_YAML = "targets:\n  CDK2:\n    mean: -7.5\n    std: 1.0\n  ERK2:\n    mean: -6.0\n    std: 0.5\n"


class _ScoresWithTimeout(list):
    """List subclass that sets timeout side-channel on the Nth pop."""

    def __init__(self, scores, mod, timeout_at):
        super().__init__(scores)
        self._mod = mod
        self._timeout_at = timeout_at
        self._pop_count = 0

    def pop(self, index=0):
        val = super().pop(index)
        self._pop_count += 1
        if self._pop_count == self._timeout_at:
            self._mod._set_last_docking_timed_out(True)
        return val


def _make_fake_docking_vina():
    """Return a FakeDockingVina class whose __call__ pops from instance._scores."""

    class FakeDockingVina:
        def __init__(self):
            self.config = {
                "error_val": 99.9,
                "receptor": "/workspace/receptor.pdbqt",
                "box_center": [0.0, 0.0, 0.0],
                "box_size": [20.0, 20.0, 20.0],
            }
            self._scores: list[float] = []

        def __call__(self, *args, **kwargs):
            return self._scores.pop(0)

        @staticmethod
        def parse_output(stdout: str, error_val: float) -> float:
            return error_val

        @staticmethod
        def docking(**kwargs: float) -> float:
            return kwargs.get("error_val", 99.9)

    return FakeDockingVina


def _install_ffreed_mock(monkeypatch, vina_class):
    """Register a fake ffreed package hierarchy using *vina_class*."""
    ffreed_pkg = types.ModuleType("ffreed")
    ffreed_env = types.ModuleType("ffreed.env")
    ffreed_docking = types.ModuleType("ffreed.env.docking")
    ffreed_docking.DockingVina = vina_class
    ffreed_pkg.env = ffreed_env
    ffreed_env.docking = ffreed_docking
    for name, mod in [("ffreed", ffreed_pkg), ("ffreed.env", ffreed_env), ("ffreed.env.docking", ffreed_docking)]:
        monkeypatch.setitem(sys.modules, name, mod)


def _setup_env(monkeypatch, tmp_path, *, reward_mode="single_a"):
    """Write normalization.yaml, set common env vars, return norm path."""
    norm_path = tmp_path / "normalization.yaml"
    norm_path.write_text(_NORMALIZATION_YAML)

    monkeypatch.setenv("REWARD_MODE", reward_mode)
    monkeypatch.setenv("TARGET_ID_A", "CDK2")
    monkeypatch.setenv("TARGET_ID_B", "ERK2")
    monkeypatch.setenv("NORMALIZATION_PATH", str(norm_path))
    monkeypatch.setenv("REWARD_METADATA_PATH", str(tmp_path / "reward_metadata.json"))
    monkeypatch.setenv("INVALID_Z_PENALTY", "4.0")

    if reward_mode in {"double", "linear_equal", "biased_a", "selective_a"}:
        dual_cfg_path = tmp_path / "dual_config.json"
        dual_cfg_path.write_text(
            json.dumps(
                {
                    "receptor_b": "/workspace/receptor_b.pdbqt",
                    "center_b": "1.0,2.0,3.0",
                    "size_b": "20.0,20.0,20.0",
                }
            )
        )
        monkeypatch.setenv("DUAL_RECEPTOR_B", "/workspace/receptor_b.pdbqt")
        monkeypatch.setenv("DUAL_CONFIG", str(dual_cfg_path))

    return norm_path


def _fresh_import(monkeypatch):
    """Remove cached module and import fresh."""
    if PATCH_DIR not in sys.path:
        sys.path.insert(0, PATCH_DIR)
    monkeypatch.delitem(sys.modules, "dual_reward_patch", raising=False)
    return importlib.import_module("dual_reward_patch")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _patch_module(tmp_path, monkeypatch):
    """Import dual_reward_patch with MagicMock DockingVina (for unit tests)."""
    from unittest.mock import MagicMock

    mock_vina = MagicMock(name="DockingVina")
    mock_vina.__call__ = MagicMock(return_value=-7.0)
    _install_ffreed_mock(monkeypatch, mock_vina)
    _setup_env(monkeypatch, tmp_path, reward_mode="single_a")

    mod = _fresh_import(monkeypatch)
    yield mod

    if PATCH_DIR in sys.path:
        sys.path.remove(PATCH_DIR)
    sys.modules.pop("dual_reward_patch", None)


@pytest.fixture
def _integration_single_a(tmp_path, monkeypatch):
    """Import in single_a mode with FakeDockingVina for integration tests."""
    FakeVina = _make_fake_docking_vina()
    _install_ffreed_mock(monkeypatch, FakeVina)
    _setup_env(monkeypatch, tmp_path, reward_mode="single_a")
    mod = _fresh_import(monkeypatch)
    instance = FakeVina()
    yield mod, instance

    if PATCH_DIR in sys.path:
        sys.path.remove(PATCH_DIR)
    sys.modules.pop("dual_reward_patch", None)


@pytest.fixture
def _integration_double(tmp_path, monkeypatch):
    """Import in double mode with FakeDockingVina for integration tests."""
    FakeVina = _make_fake_docking_vina()
    _install_ffreed_mock(monkeypatch, FakeVina)
    _setup_env(monkeypatch, tmp_path, reward_mode="double")
    mod = _fresh_import(monkeypatch)
    instance = FakeVina()
    yield mod, instance

    if PATCH_DIR in sys.path:
        sys.path.remove(PATCH_DIR)
    sys.modules.pop("dual_reward_patch", None)


@pytest.fixture
def _integration_single_b(tmp_path, monkeypatch):
    """Import in single_b mode with FakeDockingVina for integration tests."""
    FakeVina = _make_fake_docking_vina()
    _install_ffreed_mock(monkeypatch, FakeVina)
    _setup_env(monkeypatch, tmp_path, reward_mode="single_b")
    mod = _fresh_import(monkeypatch)
    instance = FakeVina()
    yield mod, instance

    if PATCH_DIR in sys.path:
        sys.path.remove(PATCH_DIR)
    sys.modules.pop("dual_reward_patch", None)


# ---------------------------------------------------------------------------
# _capped_zscore — standalone function tests
# ---------------------------------------------------------------------------


class TestCappedZscore:
    """Test _capped_zscore logic without full module import."""

    @pytest.fixture(autouse=True)
    def _import(self, _patch_module):
        self.fn = _patch_module._capped_zscore

    def test_normal_score(self):
        """Normal docking score gets standard z-score."""
        z, invalid = self.fn(score=-7.5, error_val=99.9, mean=-7.5, std=1.0, penalty_cap=4.0)
        assert z == pytest.approx(0.0)
        assert invalid is False

    def test_normal_negative_score(self):
        z, invalid = self.fn(score=-8.5, error_val=99.9, mean=-7.5, std=1.0, penalty_cap=4.0)
        assert z == pytest.approx(-1.0)
        assert invalid is False

    def test_error_val_returns_cap(self):
        """Error-val score returns capped penalty, not raw z-score."""
        z, invalid = self.fn(score=99.9, error_val=99.9, mean=-7.5, std=1.0, penalty_cap=4.0)
        assert z == 4.0
        assert invalid is True

    def test_raw_zscore_of_error_would_be_huge(self):
        """Verifies the problem: raw z of error_val would be ~107.4."""
        raw_z = (99.9 - (-7.5)) / 1.0
        assert raw_z == pytest.approx(107.4)
        # But capped_zscore returns 4.0 instead
        z, _ = self.fn(score=99.9, error_val=99.9, mean=-7.5, std=1.0, penalty_cap=4.0)
        assert z == 4.0

    def test_custom_penalty_cap(self):
        z, invalid = self.fn(score=99.9, error_val=99.9, mean=-7.5, std=1.0, penalty_cap=6.0)
        assert z == 6.0
        assert invalid is True

    def test_std_zero_raises(self):
        with pytest.raises(ZeroDivisionError, match="std=0"):
            self.fn(score=-5.0, error_val=99.9, mean=-7.5, std=0.0, penalty_cap=4.0)

    def test_std_zero_not_checked_for_error_val(self):
        """When score == error_val, std=0 is never reached."""
        z, invalid = self.fn(score=99.9, error_val=99.9, mean=-7.5, std=0.0, penalty_cap=4.0)
        assert z == 4.0
        assert invalid is True

    def test_extreme_negative_z_clamped(self):
        """Very good binding score gets z clamped to -MAX_Z_ABS."""
        # raw z = (-100.0 - (-7.5)) / 1.0 = -92.5
        z, invalid = self.fn(score=-100.0, error_val=99.9, mean=-7.5, std=1.0, penalty_cap=4.0)
        assert z == pytest.approx(-10.0)
        assert invalid is False

    def test_extreme_positive_z_clamped(self):
        """Very bad score gets z clamped to +MAX_Z_ABS."""
        # raw z = (50.0 - (-7.5)) / 1.0 = 57.5
        z, invalid = self.fn(score=50.0, error_val=99.9, mean=-7.5, std=1.0, penalty_cap=4.0)
        assert z == pytest.approx(10.0)
        assert invalid is False

    def test_z_at_boundary_not_clamped(self):
        """Score producing exactly z=-10 is not altered."""
        # score = mean + z*std = -7.5 + (-10.0)*1.0 = -17.5
        z, invalid = self.fn(score=-17.5, error_val=99.9, mean=-7.5, std=1.0, penalty_cap=4.0)
        assert z == pytest.approx(-10.0)
        assert invalid is False

    def test_within_bounds_not_clamped(self):
        """Moderate z-scores pass through unchanged."""
        # raw z = (-12.5 - (-7.5)) / 1.0 = -5.0
        z, invalid = self.fn(score=-12.5, error_val=99.9, mean=-7.5, std=1.0, penalty_cap=4.0)
        assert z == pytest.approx(-5.0)
        assert invalid is False


# ---------------------------------------------------------------------------
# _call_stats counter tracking (unit)
# ---------------------------------------------------------------------------


class TestCallStats:
    @pytest.fixture(autouse=True)
    def _import(self, _patch_module):
        self.mod = _patch_module

    def test_initial_stats_are_zero(self):
        stats = self.mod._call_stats
        assert stats["total"] == 0
        assert stats["invalid_a"] == 0
        assert stats["invalid_b"] == 0
        assert stats["timeout_a"] == 0
        assert stats["timeout_b"] == 0


# ---------------------------------------------------------------------------
# _write_reward_metadata
# ---------------------------------------------------------------------------


class TestWriteRewardMetadata:
    @pytest.fixture(autouse=True)
    def _import(self, _patch_module, tmp_path):
        self.mod = _patch_module
        self.tmp_path = tmp_path

    def test_no_write_when_zero_calls(self):
        self.mod._call_stats.update({"total": 0, "invalid_a": 0, "invalid_b": 0})
        self.mod._write_reward_metadata()
        assert not self.mod._METADATA_PATH.exists()

    def test_writes_correct_json(self):
        self.mod._call_stats.update(
            {
                "total": 100,
                "invalid_a": 5,
                "invalid_b": 3,
                "timeout_a": 2,
                "timeout_b": 1,
            }
        )
        self.mod._write_reward_metadata()
        data = json.loads(self.mod._METADATA_PATH.read_text())
        assert data["total_calls"] == 100
        assert data["invalid_count_a"] == 5
        assert data["invalid_count_b"] == 3
        assert data["invalid_reward_fraction_a"] == pytest.approx(0.05)
        assert data["invalid_reward_fraction_b"] == pytest.approx(0.03)
        assert data["timeout_count_a"] == 2
        assert data["timeout_count_b"] == 1
        assert data["timeout_fraction_a"] == pytest.approx(0.02)
        assert data["timeout_fraction_b"] == pytest.approx(0.01)
        assert data["invalid_z_penalty"] == 4.0

    def test_fractions_are_correct(self):
        self.mod._call_stats.update(
            {
                "total": 200,
                "invalid_a": 0,
                "invalid_b": 100,
                "timeout_a": 0,
                "timeout_b": 50,
            }
        )
        self.mod._write_reward_metadata()
        data = json.loads(self.mod._METADATA_PATH.read_text())
        assert data["invalid_reward_fraction_a"] == 0.0
        assert data["invalid_reward_fraction_b"] == 0.5
        assert data["timeout_fraction_a"] == 0.0
        assert data["timeout_fraction_b"] == pytest.approx(0.25)

    def test_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "reward_metadata.json"
        self.mod._METADATA_PATH = nested
        self.mod._call_stats.update({"total": 10, "invalid_a": 1, "invalid_b": 0})
        self.mod._write_reward_metadata()
        assert nested.exists()
        data = json.loads(nested.read_text())
        assert data["total_calls"] == 10


# ---------------------------------------------------------------------------
# INVALID_Z_PENALTY validation
# ---------------------------------------------------------------------------


class TestPenaltyValidation:
    def test_negative_penalty_rejected(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        mock_vina = MagicMock(name="DockingVina")
        mock_vina.__call__ = MagicMock(return_value=-7.0)
        _install_ffreed_mock(monkeypatch, mock_vina)
        _setup_env(monkeypatch, tmp_path, reward_mode="single_a")
        monkeypatch.setenv("INVALID_Z_PENALTY", "-4.0")

        with pytest.raises(ValueError, match="positive finite"):
            _fresh_import(monkeypatch)

        sys.modules.pop("dual_reward_patch", None)
        if PATCH_DIR in sys.path:
            sys.path.remove(PATCH_DIR)

    def test_zero_penalty_rejected(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        mock_vina = MagicMock(name="DockingVina")
        mock_vina.__call__ = MagicMock(return_value=-7.0)
        _install_ffreed_mock(monkeypatch, mock_vina)
        _setup_env(monkeypatch, tmp_path, reward_mode="single_a")
        monkeypatch.setenv("INVALID_Z_PENALTY", "0")

        with pytest.raises(ValueError, match="positive finite"):
            _fresh_import(monkeypatch)

        sys.modules.pop("dual_reward_patch", None)
        if PATCH_DIR in sys.path:
            sys.path.remove(PATCH_DIR)

    def test_inf_penalty_rejected(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        mock_vina = MagicMock(name="DockingVina")
        mock_vina.__call__ = MagicMock(return_value=-7.0)
        _install_ffreed_mock(monkeypatch, mock_vina)
        _setup_env(monkeypatch, tmp_path, reward_mode="single_a")
        monkeypatch.setenv("INVALID_Z_PENALTY", "inf")

        with pytest.raises(ValueError, match="positive finite"):
            _fresh_import(monkeypatch)

        sys.modules.pop("dual_reward_patch", None)
        if PATCH_DIR in sys.path:
            sys.path.remove(PATCH_DIR)


# ---------------------------------------------------------------------------
# Integration: _dual_call through real patched __call__ (single_a)
# ---------------------------------------------------------------------------


class TestDualCallSingleA:
    """Exercise _dual_call via the patched DockingVina.__call__ in single_a mode."""

    @pytest.fixture(autouse=True)
    def _setup(self, _integration_single_a):
        self.mod, self.vina = _integration_single_a

    def test_normal_score_increments_total(self):
        self.vina._scores = [-7.5]
        self.vina()  # triggers _dual_call
        assert self.mod._call_stats["total"] == 1
        assert self.mod._call_stats["invalid_a"] == 0

    def test_error_score_increments_invalid_a(self):
        self.vina._scores = [99.9]  # error_val
        self.vina()
        assert self.mod._call_stats["total"] == 1
        assert self.mod._call_stats["invalid_a"] == 1

    def test_normal_reward_value(self):
        """score=-7.5, mean=-7.5, std=1.0 → z=0 → reward=0."""
        self.vina._scores = [-7.5]
        reward = self.vina()
        assert reward == pytest.approx(0.0)

    def test_error_reward_value_is_capped(self):
        """error_val=99.9 → z=4.0 → reward=4.0 (not raw z)."""
        self.vina._scores = [99.9]
        reward = self.vina()
        assert reward == pytest.approx(4.0)

    def test_mixed_calls_track_correctly(self):
        """3 calls: 2 normal + 1 error → total=3, invalid_a=1."""
        for score in [-8.0, 99.9, -6.5]:
            self.vina._scores = [score]
            self.vina()
        assert self.mod._call_stats["total"] == 3
        assert self.mod._call_stats["invalid_a"] == 1
        assert self.mod._call_stats["invalid_b"] == 0

    def test_metadata_reflects_dual_call_counters(self, tmp_path):
        """After calls through _dual_call, metadata must have correct fractions."""
        for score in [-8.0, 99.9, -6.5, 99.9]:
            self.vina._scores = [score]
            self.vina()
        self.mod._write_reward_metadata()
        data = json.loads(self.mod._METADATA_PATH.read_text())
        assert data["total_calls"] == 4
        assert data["invalid_count_a"] == 2
        assert data["invalid_reward_fraction_a"] == pytest.approx(0.5)

    def test_timeout_flag_increments_timeout_a(self):
        """Simulate timeout via side-channel setter."""
        # Normal error (not timeout) — only invalid_a incremented
        self.vina._scores = [99.9]
        self.vina()
        assert self.mod._call_stats["invalid_a"] == 1
        assert self.mod._call_stats["timeout_a"] == 0

        # Timeout error — both invalid_a and timeout_a incremented
        self.vina._scores = [99.9]
        self.mod._set_last_docking_timed_out(True)
        self.vina()
        assert self.mod._call_stats["invalid_a"] == 2
        assert self.mod._call_stats["timeout_a"] == 1

    def test_timeout_flag_reset_after_read(self):
        """Flag must be consumed (reset to False) after _dual_call reads it."""
        self.vina._scores = [99.9]
        self.mod._set_last_docking_timed_out(True)
        self.vina()
        assert self.mod._get_last_docking_timed_out() is False

    def test_primary_target_in_metadata_single_a(self):
        """Metadata reports primary_target=CDK2 and reward_mode=single_a."""
        self.vina._scores = [-7.5]
        self.vina()
        self.mod._write_reward_metadata()
        data = json.loads(self.mod._METADATA_PATH.read_text())
        assert data["primary_target"] == "CDK2"
        assert data["reward_mode"] == "single_a"


# ---------------------------------------------------------------------------
# Integration: _dual_call in double mode
# ---------------------------------------------------------------------------


class TestDualCallDouble:
    """Exercise _dual_call in double mode (scores both targets)."""

    @pytest.fixture(autouse=True)
    def _setup(self, _integration_double):
        self.mod, self.vina = _integration_double

    def test_both_normal(self):
        """Both targets normal → no invalids."""
        self.vina._scores = [-7.5, -6.0]  # A then B
        reward = self.vina()
        # z_a = (-7.5 - (-7.5))/1.0 = 0.0, z_b = (-6.0 - (-6.0))/0.5 = 0.0
        assert reward == pytest.approx(0.0)  # max(0, 0) = 0
        assert self.mod._call_stats["total"] == 1
        assert self.mod._call_stats["invalid_a"] == 0
        assert self.mod._call_stats["invalid_b"] == 0

    def test_a_error_b_normal(self):
        """Target A error, B normal → invalid_a incremented."""
        self.vina._scores = [99.9, -6.0]
        reward = self.vina()
        # z_a = 4.0 (capped), z_b = 0.0 → max(4.0, 0.0) = 4.0
        assert reward == pytest.approx(4.0)
        assert self.mod._call_stats["invalid_a"] == 1
        assert self.mod._call_stats["invalid_b"] == 0

    def test_both_error(self):
        """Both targets error → both invalids incremented."""
        self.vina._scores = [99.9, 99.9]
        reward = self.vina()
        # z_a = z_b = 4.0 → max(4.0, 4.0) = 4.0
        assert reward == pytest.approx(4.0)
        assert self.mod._call_stats["invalid_a"] == 1
        assert self.mod._call_stats["invalid_b"] == 1

    def test_b_error_a_normal(self):
        """Target A normal, B error → only invalid_b incremented."""
        self.vina._scores = [-7.5, 99.9]
        reward = self.vina()
        # z_a = 0.0, z_b = 4.0 (capped) → max(0.0, 4.0) = 4.0
        assert reward == pytest.approx(4.0)
        assert self.mod._call_stats["invalid_a"] == 0
        assert self.mod._call_stats["invalid_b"] == 1

    def test_timeout_b_tracked_separately(self):
        """Timeout on B → both invalid_b and timeout_b incremented."""
        # In double mode _original_call is called twice (A then B).
        # _dual_call consumes timeout side-channel after each call.
        # Use a list subclass that sets the flag on the 2nd pop (= B's call).
        self.vina._scores = _ScoresWithTimeout([-7.5, 99.9], self.mod, timeout_at=2)
        self.vina()

        assert self.mod._call_stats["timeout_a"] == 0
        assert self.mod._call_stats["timeout_b"] == 1
        assert self.mod._call_stats["invalid_b"] == 1


# ---------------------------------------------------------------------------
# Integration: _dual_call in single_b mode
# ---------------------------------------------------------------------------


class TestDualCallSingleB:
    """Exercise _dual_call in single_b mode (B is primary receptor)."""

    @pytest.fixture(autouse=True)
    def _setup(self, _integration_single_b):
        self.mod, self.vina = _integration_single_b

    def test_normal_b_score(self):
        """Normal B score → no invalids, correct reward."""
        self.vina._scores = [-6.0]  # Only B (primary)
        reward = self.vina()
        # z_b = (-6.0 - (-6.0))/0.5 = 0.0 → reward = 0.0
        assert reward == pytest.approx(0.0)
        assert self.mod._call_stats["invalid_b"] == 0

    def test_error_b_score(self):
        """Error B score → invalid_b incremented, capped reward."""
        self.vina._scores = [99.9]  # Only B (primary, error)
        reward = self.vina()
        assert reward == pytest.approx(4.0)
        assert self.mod._call_stats["invalid_b"] == 1

    def test_single_b_does_not_score_a(self):
        """single_b only scores B — A is never called."""
        self.vina._scores = [-6.0]
        self.vina()
        assert self.mod._call_stats["total"] == 1
        assert self.mod._call_stats["invalid_a"] == 0

    def test_single_b_timeout_tracked(self):
        """Timeout on B is tracked correctly with 1 score."""
        self.vina._scores = _ScoresWithTimeout([99.9], self.mod, timeout_at=1)
        self.vina()
        assert self.mod._call_stats["timeout_b"] == 1
        assert self.mod._call_stats["invalid_b"] == 1
        assert self.mod._call_stats["timeout_a"] == 0

    def test_primary_target_in_metadata_single_b(self):
        """Metadata reports primary_target=ERK2 and reward_mode=single_b."""
        self.vina._scores = [-6.0]
        self.vina()
        self.mod._write_reward_metadata()
        data = json.loads(self.mod._METADATA_PATH.read_text())
        assert data["primary_target"] == "ERK2"
        assert data["reward_mode"] == "single_b"


# ---------------------------------------------------------------------------
# Integration: reward value bounds (standalone)
# ---------------------------------------------------------------------------


class TestRewardBounds:
    """Verify that the capped penalty produces bounded rewards."""

    @pytest.fixture(autouse=True)
    def _import(self, _patch_module):
        self.mod = _patch_module

    def test_normal_reward_is_small(self):
        z, _ = self.mod._capped_zscore(score=-8.0, error_val=99.9, mean=-7.5, std=1.0, penalty_cap=4.0)
        reward = -z
        assert -5.0 <= reward <= 5.0

    def test_error_reward_is_capped(self):
        z, _ = self.mod._capped_zscore(score=99.9, error_val=99.9, mean=-7.5, std=1.0, penalty_cap=4.0)
        reward = -z
        assert reward == -4.0

    def test_error_reward_does_not_produce_extreme_z(self):
        """The whole point: error_val must NOT produce |z| > 10."""
        z, _ = self.mod._capped_zscore(score=99.9, error_val=99.9, mean=-7.5, std=0.705, penalty_cap=4.0)
        assert abs(z) <= 10

    def test_no_nan_in_reward(self):
        z, _ = self.mod._capped_zscore(score=-7.5, error_val=99.9, mean=-7.5, std=1.0, penalty_cap=4.0)
        assert not math.isnan(z)


# ---------------------------------------------------------------------------
# run_valid in metadata (ENG-03 / KVA-257)
# ---------------------------------------------------------------------------


class TestRunValidInMetadata:
    """Verify _write_reward_metadata sets run_valid based on QG env thresholds."""

    @pytest.fixture(autouse=True)
    def _import(self, _patch_module, tmp_path, monkeypatch):
        self.mod = _patch_module
        self.tmp_path = tmp_path

    def test_run_valid_in_metadata_true(self, monkeypatch):
        """All fractions below default thresholds → run_valid=True."""
        monkeypatch.setenv("QG_MAX_INVALID_FRACTION", "0.10")
        monkeypatch.setenv("QG_MAX_TIMEOUT_FRACTION", "0.05")
        self.mod._call_stats.update({"total": 100, "invalid_a": 5, "invalid_b": 3, "timeout_a": 2, "timeout_b": 1})
        self.mod._write_reward_metadata()
        data = json.loads(self.mod._METADATA_PATH.read_text())
        assert data["run_valid"] is True

    def test_run_valid_in_metadata_false(self, monkeypatch):
        """Invalid fraction exceeds threshold → run_valid=False."""
        monkeypatch.setenv("QG_MAX_INVALID_FRACTION", "0.10")
        monkeypatch.setenv("QG_MAX_TIMEOUT_FRACTION", "0.05")
        self.mod._call_stats.update({"total": 100, "invalid_a": 15, "invalid_b": 3, "timeout_a": 2, "timeout_b": 1})
        self.mod._write_reward_metadata()
        data = json.loads(self.mod._METADATA_PATH.read_text())
        assert data["run_valid"] is False


# ---------------------------------------------------------------------------
# Score validation in _parse_result_from_pdbqt
# ---------------------------------------------------------------------------


class TestParseResultValidation:
    """Verify that _parse_result_from_pdbqt rejects corrupt scores."""

    @pytest.fixture(autouse=True)
    def _import(self, _patch_module, tmp_path):
        self.mod = _patch_module
        self.tmp_path = tmp_path

    def test_normal_score_accepted(self):
        p = self.tmp_path / "normal.pdbqt"
        p.write_text("REMARK VINA RESULT:    -7.500      0.000      0.000\n")
        assert self.mod._parse_result_from_pdbqt(p) == pytest.approx(-7.5)

    def test_positive_in_range_accepted(self):
        p = self.tmp_path / "positive.pdbqt"
        p.write_text("REMARK VINA RESULT:     5.000      0.000      0.000\n")
        assert self.mod._parse_result_from_pdbqt(p) == pytest.approx(5.0)

    def test_extreme_negative_rejected(self):
        p = self.tmp_path / "corrupt_neg.pdbqt"
        p.write_text("REMARK VINA RESULT:    -336.073      0.000      0.000\n")
        assert self.mod._parse_result_from_pdbqt(p) is None

    def test_extreme_positive_rejected(self):
        p = self.tmp_path / "corrupt_pos.pdbqt"
        p.write_text("REMARK VINA RESULT:    10725114.000      0.000      0.000\n")
        assert self.mod._parse_result_from_pdbqt(p) is None

    def test_boundary_min_accepted(self):
        p = self.tmp_path / "boundary_min.pdbqt"
        p.write_text("REMARK VINA RESULT:    -20.000      0.000      0.000\n")
        assert self.mod._parse_result_from_pdbqt(p) == pytest.approx(-20.0)

    def test_boundary_max_accepted(self):
        p = self.tmp_path / "boundary_max.pdbqt"
        p.write_text("REMARK VINA RESULT:     10.000      0.000      0.000\n")
        assert self.mod._parse_result_from_pdbqt(p) == pytest.approx(10.0)

    def test_just_outside_min_rejected(self):
        p = self.tmp_path / "outside_min.pdbqt"
        p.write_text("REMARK VINA RESULT:    -20.001      0.000      0.000\n")
        assert self.mod._parse_result_from_pdbqt(p) is None

    def test_missing_file_returns_none(self):
        assert self.mod._parse_result_from_pdbqt(self.tmp_path / "nonexistent.pdbqt") is None

    def test_no_result_line_returns_none(self):
        p = self.tmp_path / "no_result.pdbqt"
        p.write_text("ATOM      1  C1  UNL     1       0.000   0.000   0.000\n")
        assert self.mod._parse_result_from_pdbqt(p) is None


# ---------------------------------------------------------------------------
# Integration: _dual_call in new weight modes (linear_equal, biased_a, selective_a)
# ---------------------------------------------------------------------------


def _make_new_mode_fixture(mode_name):
    """Factory: create a fixture for a new dual-target weight mode."""

    @pytest.fixture
    def _fixture(tmp_path, monkeypatch):
        FakeVina = _make_fake_docking_vina()
        _install_ffreed_mock(monkeypatch, FakeVina)
        _setup_env(monkeypatch, tmp_path, reward_mode=mode_name)
        mod = _fresh_import(monkeypatch)
        instance = FakeVina()
        yield mod, instance

        if PATCH_DIR in sys.path:
            sys.path.remove(PATCH_DIR)
        sys.modules.pop("dual_reward_patch", None)

    return _fixture


_integration_linear_equal = _make_new_mode_fixture("linear_equal")
_integration_biased_a = _make_new_mode_fixture("biased_a")
_integration_selective_a = _make_new_mode_fixture("selective_a")


class TestDualCallLinearEqual:
    """Exercise _dual_call in linear_equal mode: z_a + z_b."""

    @pytest.fixture(autouse=True)
    def _setup(self, _integration_linear_equal):
        self.mod, self.vina = _integration_linear_equal

    def test_reward_value(self):
        # score_a=-6.5 → z_a=(−6.5−(−7.5))/1.0=1.0
        # score_b=-5.0 → z_b=(−5.0−(−6.0))/0.5=2.0
        # linear_equal: z_a + z_b = 3.0
        self.vina._scores = [-6.5, -5.0]
        reward = self.vina()
        assert reward == pytest.approx(3.0)

    def test_both_mean_returns_zero(self):
        # z_a=0, z_b=0 → 0+0=0
        self.vina._scores = [-7.5, -6.0]
        reward = self.vina()
        assert reward == pytest.approx(0.0)

    def test_stats_tracked(self):
        self.vina._scores = [-6.5, -5.0]
        self.vina()
        assert self.mod._call_stats["total"] == 1
        assert self.mod._call_stats["invalid_a"] == 0
        assert self.mod._call_stats["invalid_b"] == 0


class TestDualCallBiasedA:
    """Exercise _dual_call in biased_a mode: z_a + 0.3*z_b."""

    @pytest.fixture(autouse=True)
    def _setup(self, _integration_biased_a):
        self.mod, self.vina = _integration_biased_a

    def test_reward_value(self):
        # z_a=1.0, z_b=2.0 → 1.0 + 0.3*2.0 = 1.6
        self.vina._scores = [-6.5, -5.0]
        reward = self.vina()
        assert reward == pytest.approx(1.6)

    def test_both_mean_returns_zero(self):
        self.vina._scores = [-7.5, -6.0]
        reward = self.vina()
        assert reward == pytest.approx(0.0)

    def test_stats_tracked(self):
        self.vina._scores = [-6.5, -5.0]
        self.vina()
        assert self.mod._call_stats["total"] == 1
        assert self.mod._call_stats["invalid_a"] == 0
        assert self.mod._call_stats["invalid_b"] == 0


class TestDualCallSelectiveA:
    """Exercise _dual_call in selective_a mode: z_a - z_b."""

    @pytest.fixture(autouse=True)
    def _setup(self, _integration_selective_a):
        self.mod, self.vina = _integration_selective_a

    def test_reward_value(self):
        # z_a=1.0, z_b=2.0 → 1.0 - 2.0 = -1.0
        self.vina._scores = [-6.5, -5.0]
        reward = self.vina()
        assert reward == pytest.approx(-1.0)

    def test_both_mean_returns_zero(self):
        self.vina._scores = [-7.5, -6.0]
        reward = self.vina()
        assert reward == pytest.approx(0.0)

    def test_stats_tracked(self):
        self.vina._scores = [-6.5, -5.0]
        self.vina()
        assert self.mod._call_stats["total"] == 1
        assert self.mod._call_stats["invalid_a"] == 0
        assert self.mod._call_stats["invalid_b"] == 0
