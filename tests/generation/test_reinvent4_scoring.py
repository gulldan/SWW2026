"""Tests for REINVENT4 scoring component."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from collections.abc import Callable

from dualdocking.generation.reinvent4.scoring import (
    INVALID_Z_PENALTY,
    MAX_Z_ABS,
    SCORE_MAX,
    SCORE_MIN,
    DockingTarget,
    DualDockingComponent,
    Parameters,
    RewardStats,
    _reward_stats,
    capped_zscore,
    parse_vina_score,
    score_batch,
)


# --- capped_zscore ---


class TestCappedZscore:
    def test_normal(self) -> None:
        """Normal z-score within bounds."""
        assert capped_zscore(-8.0, -7.0, 1.0) == pytest.approx(-1.0)

    def test_clamped_positive(self) -> None:
        """Extreme positive value clamped to +MAX_Z_ABS."""
        z = capped_zscore(100.0, 0.0, 1.0)
        assert z == pytest.approx(MAX_Z_ABS)

    def test_clamped_negative(self) -> None:
        """Extreme negative value clamped to -MAX_Z_ABS."""
        z = capped_zscore(-100.0, 0.0, 1.0)
        assert z == pytest.approx(-MAX_Z_ABS)

    def test_zero_std(self) -> None:
        """Zero std returns 0."""
        assert capped_zscore(-7.0, -7.0, 0.0) == 0.0


# --- parse_vina_score ---


class TestParseVinaScore:
    def test_valid_score(self, tmp_path: Path) -> None:
        """Parse valid docking score from PDBQT."""
        pdbqt = tmp_path / "out.pdbqt"
        pdbqt.write_text(
            "REMARK VINA RESULT:   -8.5      0.000      0.000\n"
            "MODEL 1\nATOM ...\nENDMDL\n"
        )
        assert parse_vina_score(pdbqt) == pytest.approx(-8.5)

    def test_out_of_range(self, tmp_path: Path) -> None:
        """Score outside [SCORE_MIN, SCORE_MAX] returns None."""
        pdbqt = tmp_path / "out.pdbqt"
        pdbqt.write_text("REMARK VINA RESULT:   -50.0      0.000      0.000\n")
        assert parse_vina_score(pdbqt) is None

    def test_missing_file(self, tmp_path: Path) -> None:
        """Missing file returns None."""
        assert parse_vina_score(tmp_path / "nonexistent.pdbqt") is None

    def test_malformed(self, tmp_path: Path) -> None:
        """Malformed PDBQT returns None."""
        pdbqt = tmp_path / "out.pdbqt"
        pdbqt.write_text("REMARK not a result line\n")
        assert parse_vina_score(pdbqt) is None


# --- score_batch ---


def _make_gpu_batch_mock(score_value: float | None) -> Callable:
    """Return a mock for ``dock_gpu_batch`` that assigns *score_value* to all ligands."""

    def _mock(
        ligand_pdbqts: list[Path], *_args: object, **_kwargs: object,
    ) -> dict[Path, float | None]:
        return {lig: score_value for lig in ligand_pdbqts}

    return _mock


def _make_gpu_batch_mock_ab(score_a: float | None, score_b: float | None) -> Callable:
    """Return a mock for ``dock_gpu_batch`` that returns *score_a* on first call, *score_b* on second."""
    call_count = {"n": 0}

    def _mock(
        ligand_pdbqts: list[Path], *_args: object, **_kwargs: object,
    ) -> dict[Path, float | None]:
        val = score_a if call_count["n"] == 0 else score_b
        call_count["n"] += 1
        return {lig: val for lig in ligand_pdbqts}

    return _mock


class TestScoreBatch:
    @pytest.fixture(autouse=True)
    def _reset_stats(self) -> None:
        """Reset module-level reward stats before each test."""
        _reward_stats.reset()

    @pytest.fixture()
    def target_a(self, tmp_path: Path) -> DockingTarget:
        receptor = tmp_path / "receptor_a.pdbqt"
        receptor.write_text("ATOM mock receptor\n")
        return DockingTarget(
            pdbqt_path=receptor,
            center="1.0,2.0,3.0",
            size="20.0,20.0,20.0",
            norm_mean=-7.0,
            norm_std=1.0,
        )

    @pytest.fixture()
    def target_b(self, tmp_path: Path) -> DockingTarget:
        receptor = tmp_path / "receptor_b.pdbqt"
        receptor.write_text("ATOM mock receptor\n")
        return DockingTarget(
            pdbqt_path=receptor,
            center="4.0,5.0,6.0",
            size="20.0,20.0,20.0",
            norm_mean=-8.0,
            norm_std=0.8,
        )

    def test_all_failures_get_penalty(self, target_a: DockingTarget) -> None:
        """When all SMILES fail conversion, rewards are -INVALID_Z_PENALTY."""
        with patch("dualdocking.generation.reinvent4.scoring.smiles_to_pdbqt", return_value=False):
            rewards = score_batch(["bad1", "bad2"], target_a, None, "single_a")
        np.testing.assert_allclose(rewards, [-INVALID_Z_PENALTY, -INVALID_Z_PENALTY])

    def test_single_a_mode(self, target_a: DockingTarget) -> None:
        """single_a mode returns negated z_a (REINVENT4: higher = better)."""
        with (
            patch("dualdocking.generation.reinvent4.scoring.smiles_to_pdbqt", return_value=True),
            patch("dualdocking.generation.reinvent4.scoring.dock_gpu_batch", side_effect=_make_gpu_batch_mock(-8.0)),
        ):
            rewards = score_batch(["CCO"], target_a, None, "single_a")
        # z = (-8.0 - (-7.0)) / 1.0 = -1.0; negated → +1.0
        assert rewards[0] == pytest.approx(1.0)

    def test_single_b_mode(self, target_a: DockingTarget, target_b: DockingTarget) -> None:
        """single_b mode docks both targets; reward uses z_b only."""
        with (
            patch("dualdocking.generation.reinvent4.scoring.smiles_to_pdbqt", return_value=True),
            patch("dualdocking.generation.reinvent4.scoring.dock_gpu_batch",
                  side_effect=_make_gpu_batch_mock_ab(-8.0, -8.8)),
        ):
            rewards = score_batch(["CCO"], target_a, target_b, "single_b")
        # z_b = (-8.8 - (-8.0)) / 0.8 = -1.0; negated → +1.0
        assert rewards[0] == pytest.approx(1.0)

    def test_double_mode(self, target_a: DockingTarget, target_b: DockingTarget) -> None:
        """double mode returns negated max(z_a, z_b)."""
        with (
            patch("dualdocking.generation.reinvent4.scoring.smiles_to_pdbqt", return_value=True),
            patch("dualdocking.generation.reinvent4.scoring.dock_gpu_batch",
                  side_effect=_make_gpu_batch_mock_ab(-8.0, -8.8)),
        ):
            rewards = score_batch(["CCO"], target_a, target_b, "double")
        # z_a = (-8.0 - (-7.0)) / 1.0 = -1.0
        # z_b = (-8.8 - (-8.0)) / 0.8 = -1.0
        # max(-1.0, -1.0) = -1.0; negated → +1.0
        assert rewards[0] == pytest.approx(1.0)

    def test_docking_failure_returns_penalty(self, target_a: DockingTarget) -> None:
        """Docking failure (None score) returns penalty."""
        with (
            patch("dualdocking.generation.reinvent4.scoring.smiles_to_pdbqt", return_value=True),
            patch("dualdocking.generation.reinvent4.scoring.dock_gpu_batch", side_effect=_make_gpu_batch_mock(None)),
        ):
            rewards = score_batch(["CCO"], target_a, None, "single_a")
        assert rewards[0] == pytest.approx(-INVALID_Z_PENALTY)

    def test_selective_a_mode(self, target_a: DockingTarget, target_b: DockingTarget) -> None:
        """selective_a mode returns negated (z_a - z_b)."""
        with (
            patch("dualdocking.generation.reinvent4.scoring.smiles_to_pdbqt", return_value=True),
            patch("dualdocking.generation.reinvent4.scoring.dock_gpu_batch",
                  side_effect=_make_gpu_batch_mock_ab(-9.0, -8.0)),
        ):
            rewards = score_batch(["CCO"], target_a, target_b, "selective_a")
        # z_a = (-9.0 - (-7.0)) / 1.0 = -2.0
        # z_b = (-8.0 - (-8.0)) / 0.8 = 0.0
        # selective_a: z_a - z_b = -2.0; negated → +2.0
        assert rewards[0] == pytest.approx(2.0)

    def test_linear_equal_mode(self, target_a: DockingTarget, target_b: DockingTarget) -> None:
        """linear_equal mode returns negated (z_a + z_b)."""
        with (
            patch("dualdocking.generation.reinvent4.scoring.smiles_to_pdbqt", return_value=True),
            patch("dualdocking.generation.reinvent4.scoring.dock_gpu_batch",
                  side_effect=_make_gpu_batch_mock_ab(-8.0, -8.8)),
        ):
            rewards = score_batch(["CCO"], target_a, target_b, "linear_equal")
        # z_a = -1.0, z_b = -1.0 -> -1.0 + -1.0 = -2.0; negated → +2.0
        assert rewards[0] == pytest.approx(2.0)

    def test_biased_a_mode(self, target_a: DockingTarget, target_b: DockingTarget) -> None:
        """biased_a mode returns negated (z_a + 0.3*z_b)."""
        with (
            patch("dualdocking.generation.reinvent4.scoring.smiles_to_pdbqt", return_value=True),
            patch("dualdocking.generation.reinvent4.scoring.dock_gpu_batch",
                  side_effect=_make_gpu_batch_mock_ab(-8.0, -8.8)),
        ):
            rewards = score_batch(["CCO"], target_a, target_b, "biased_a")
        # z_a = -1.0, z_b = -1.0 -> -1.0 + 0.3 * -1.0 = -1.3; negated → +1.3
        assert rewards[0] == pytest.approx(1.3)


# --- DualDockingComponent ---


class TestDualDockingComponent:
    def test_init_single_target(self, tmp_path: Path) -> None:
        """Component initializes with single target."""
        receptor = tmp_path / "receptor.pdbqt"
        receptor.write_text("ATOM\n")
        params = Parameters(
            receptor_a=[str(receptor)],
            center_a=["1,2,3"],
            size_a=["20,20,20"],
            norm_mean_a=[-7.0],
            norm_std_a=[1.0],
            reward_mode=["single_a"],
        )
        comp = DualDockingComponent(params)
        assert comp.reward_mode == "single_a"
        assert comp.target_a.pdbqt_path == receptor
        assert comp.target_b is None

    def test_init_dual_target(self, tmp_path: Path) -> None:
        """Component initializes with both targets."""
        receptor_a = tmp_path / "receptor_a.pdbqt"
        receptor_b = tmp_path / "receptor_b.pdbqt"
        receptor_a.write_text("ATOM\n")
        receptor_b.write_text("ATOM\n")
        params = Parameters(
            receptor_a=[str(receptor_a)],
            center_a=["1,2,3"],
            size_a=["20,20,20"],
            norm_mean_a=[-7.0],
            norm_std_a=[1.0],
            receptor_b=[str(receptor_b)],
            center_b=["4,5,6"],
            size_b=["20,20,20"],
            norm_mean_b=[-8.0],
            norm_std_b=[0.8],
            reward_mode=["double"],
        )
        comp = DualDockingComponent(params)
        assert comp.reward_mode == "double"
        assert comp.target_b is not None

    def test_init_missing_receptor_raises(self) -> None:
        """Missing receptor_a raises ValueError."""
        params = Parameters()
        with pytest.raises(ValueError, match="requires receptor_a"):
            DualDockingComponent(params)

    def test_call_delegates_to_score_batch(self, tmp_path: Path) -> None:
        """__call__ delegates to score_batch."""
        receptor = tmp_path / "receptor.pdbqt"
        receptor.write_text("ATOM\n")
        params = Parameters(
            receptor_a=[str(receptor)],
            center_a=["1,2,3"],
            size_a=["20,20,20"],
            norm_mean_a=[-7.0],
            norm_std_a=[1.0],
            reward_mode=["single_a"],
        )
        comp = DualDockingComponent(params)

        fake_rewards = np.array([1.0, 2.0])
        with patch("dualdocking.generation.reinvent4.scoring.score_batch", return_value=fake_rewards) as mock:
            result = comp(["CCO", "CCC"])
        mock.assert_called_once()
        # __call__ wraps in ComponentResults for REINVENT4 v4.7.15
        np.testing.assert_array_equal(result.scores[0], fake_rewards)


# --- Plugin protocol ---


class TestPluginProtocol:
    def test_parameters_has_tag(self) -> None:
        """Parameters class has __parameters attribute for REINVENT4 discovery."""
        # REINVENT4's add_tag("__parameters") sets cls.__parameters = "True"
        assert getattr(Parameters, "__parameters", None) is not None

    def test_component_has_tag(self) -> None:
        """DualDockingComponent has __component attribute for REINVENT4 discovery."""
        assert getattr(DualDockingComponent, "__component", None) is not None


# --- RewardStats ---


class TestRewardStats:
    def test_initial_state(self) -> None:
        """Fresh instance has zero counters."""
        stats = RewardStats()
        assert stats.total == 0
        assert stats.invalid_a == 0
        assert stats.invalid_b == 0

    def test_reset(self) -> None:
        """reset() zeros all counters."""
        stats = RewardStats(total=10, invalid_a=3, invalid_b=2)
        stats.reset()
        assert stats.total == 0
        assert stats.invalid_a == 0
        assert stats.invalid_b == 0

    def test_record_valid_single(self) -> None:
        """Single-target success increments total only."""
        stats = RewardStats()
        stats.record(valid_a=True)
        assert stats.total == 1
        assert stats.invalid_a == 0
        assert stats.invalid_b == 0

    def test_record_invalid_single(self) -> None:
        """Single-target failure increments total and invalid_a."""
        stats = RewardStats()
        stats.record(valid_a=False)
        assert stats.total == 1
        assert stats.invalid_a == 1
        assert stats.invalid_b == 0

    def test_record_valid_dual(self) -> None:
        """Dual-target success."""
        stats = RewardStats()
        stats.record(valid_a=True, valid_b=True)
        assert stats.total == 1
        assert stats.invalid_a == 0
        assert stats.invalid_b == 0

    def test_record_invalid_b_only(self) -> None:
        """Target-B failure with target-A success."""
        stats = RewardStats()
        stats.record(valid_a=True, valid_b=False)
        assert stats.total == 1
        assert stats.invalid_a == 0
        assert stats.invalid_b == 1

    def test_record_both_invalid(self) -> None:
        """Both targets fail (e.g. SMILES conversion failed)."""
        stats = RewardStats()
        stats.record(valid_a=False, valid_b=False)
        assert stats.total == 1
        assert stats.invalid_a == 1
        assert stats.invalid_b == 1

    def test_record_none_b_ignored(self) -> None:
        """valid_b=None does not count toward invalid_b."""
        stats = RewardStats()
        stats.record(valid_a=False, valid_b=None)
        assert stats.invalid_b == 0

    def test_to_metadata_schema(self) -> None:
        """to_metadata() returns all keys expected by validate_reward_quality()."""
        stats = RewardStats(total=100, invalid_a=5, invalid_b=3)
        meta = stats.to_metadata(reward_mode="double", primary_target="both")
        expected_keys = {
            "total_calls", "invalid_count_a", "invalid_count_b",
            "invalid_reward_fraction_a", "invalid_reward_fraction_b",
            "timeout_count_a", "timeout_count_b",
            "timeout_fraction_a", "timeout_fraction_b",
            "invalid_z_penalty", "max_z_abs",
            "primary_target", "reward_mode", "reward_params", "run_valid",
        }
        assert set(meta.keys()) == expected_keys

    def test_to_metadata_fractions(self) -> None:
        """Fractions are computed correctly."""
        stats = RewardStats(total=200, invalid_a=20, invalid_b=10)
        meta = stats.to_metadata(reward_mode="single_a", primary_target="CDK2")
        assert meta["invalid_reward_fraction_a"] == pytest.approx(0.1)
        assert meta["invalid_reward_fraction_b"] == pytest.approx(0.05)
        assert meta["total_calls"] == 200

    def test_to_metadata_run_valid_true(self) -> None:
        """run_valid=True when both fractions <= 0.10."""
        stats = RewardStats(total=100, invalid_a=10, invalid_b=10)
        meta = stats.to_metadata(reward_mode="double", primary_target="both")
        assert meta["run_valid"] is True

    def test_to_metadata_run_valid_false(self) -> None:
        """run_valid=False when a fraction > 0.10."""
        stats = RewardStats(total=100, invalid_a=11, invalid_b=0)
        meta = stats.to_metadata(reward_mode="single_a", primary_target="CDK2")
        assert meta["run_valid"] is False

    def test_to_metadata_zero_total(self) -> None:
        """Zero total does not cause division by zero."""
        stats = RewardStats()
        meta = stats.to_metadata(reward_mode="single_a", primary_target="CDK2")
        assert meta["total_calls"] == 0
        assert meta["invalid_reward_fraction_a"] == 0.0

    def test_write_metadata(self, tmp_path: Path) -> None:
        """write_metadata() creates a valid JSON file."""
        import json

        stats = RewardStats(total=50, invalid_a=2, invalid_b=1)
        out = tmp_path / "ws" / "reward_metadata.json"
        stats.write_metadata(out, reward_mode="double", primary_target="both")
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["total_calls"] == 50
        assert data["reward_mode"] == "double"
        assert data["primary_target"] == "both"


# --- score_batch stats integration ---


class TestScoreBatchStats:
    """Verify that score_batch() updates the module-level _reward_stats."""

    @pytest.fixture(autouse=True)
    def _reset_stats(self) -> None:
        _reward_stats.reset()

    @pytest.fixture()
    def target_a(self, tmp_path: Path) -> DockingTarget:
        receptor = tmp_path / "receptor_a.pdbqt"
        receptor.write_text("ATOM mock receptor\n")
        return DockingTarget(
            pdbqt_path=receptor,
            center="1.0,2.0,3.0",
            size="20.0,20.0,20.0",
            norm_mean=-7.0,
            norm_std=1.0,
        )

    @pytest.fixture()
    def target_b(self, tmp_path: Path) -> DockingTarget:
        receptor = tmp_path / "receptor_b.pdbqt"
        receptor.write_text("ATOM mock receptor\n")
        return DockingTarget(
            pdbqt_path=receptor,
            center="4.0,5.0,6.0",
            size="20.0,20.0,20.0",
            norm_mean=-8.0,
            norm_std=0.8,
        )

    def test_all_failures_counted(self, target_a: DockingTarget) -> None:
        """SMILES conversion failures are tracked."""
        with patch("dualdocking.generation.reinvent4.scoring.smiles_to_pdbqt", return_value=False):
            score_batch(["bad1", "bad2"], target_a, None, "single_a")
        assert _reward_stats.total == 2
        assert _reward_stats.invalid_a == 2
        assert _reward_stats.invalid_b == 0  # single mode → valid_b=None

    def test_success_counted(self, target_a: DockingTarget) -> None:
        """Successful docking is tracked."""
        with (
            patch("dualdocking.generation.reinvent4.scoring.smiles_to_pdbqt", return_value=True),
            patch("dualdocking.generation.reinvent4.scoring.dock_gpu_batch", side_effect=_make_gpu_batch_mock(-8.0)),
        ):
            score_batch(["CCO"], target_a, None, "single_a")
        assert _reward_stats.total == 1
        assert _reward_stats.invalid_a == 0

    def test_dock_a_failure_counted(self, target_a: DockingTarget) -> None:
        """Docking failure on target A is tracked."""
        with (
            patch("dualdocking.generation.reinvent4.scoring.smiles_to_pdbqt", return_value=True),
            patch("dualdocking.generation.reinvent4.scoring.dock_gpu_batch", side_effect=_make_gpu_batch_mock(None)),
        ):
            score_batch(["CCO"], target_a, None, "single_a")
        assert _reward_stats.total == 1
        assert _reward_stats.invalid_a == 1

    def test_dock_b_failure_counted(self, target_a: DockingTarget, target_b: DockingTarget) -> None:
        """Docking failure on target B is tracked (target A succeeds)."""
        with (
            patch("dualdocking.generation.reinvent4.scoring.smiles_to_pdbqt", return_value=True),
            patch("dualdocking.generation.reinvent4.scoring.dock_gpu_batch",
                  side_effect=_make_gpu_batch_mock_ab(-8.0, None)),
        ):
            score_batch(["CCO"], target_a, target_b, "double")
        assert _reward_stats.total == 1
        assert _reward_stats.invalid_a == 0
        assert _reward_stats.invalid_b == 1

    def test_dual_success_counted(self, target_a: DockingTarget, target_b: DockingTarget) -> None:
        """Both targets succeed in dual mode."""
        with (
            patch("dualdocking.generation.reinvent4.scoring.smiles_to_pdbqt", return_value=True),
            patch("dualdocking.generation.reinvent4.scoring.dock_gpu_batch",
                  side_effect=_make_gpu_batch_mock_ab(-8.0, -8.8)),
        ):
            score_batch(["CCO"], target_a, target_b, "double")
        assert _reward_stats.total == 1
        assert _reward_stats.invalid_a == 0
        assert _reward_stats.invalid_b == 0
