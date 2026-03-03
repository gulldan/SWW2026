"""Tests for power analysis (dualdocking.analysis.calibration.power)."""

from __future__ import annotations

import json

import pytest

from dualdocking.analysis.calibration.power import (
    cliffs_delta_to_cohens_d,
    mde_paired_ttest,
    power_at_effect,
    power_curve,
    required_seeds,
)


class TestMDEPairedTTest:
    """Tests for minimum detectable effect size."""

    def test_mde_decreases_with_more_seeds(self) -> None:
        d5 = mde_paired_ttest(5)
        d20 = mde_paired_ttest(20)
        d50 = mde_paired_ttest(50)
        assert d5 > d20 > d50

    def test_mde_m5_is_very_large(self) -> None:
        """With m=5, MDE should be > 1.0 (very large effect)."""
        d = mde_paired_ttest(5, alpha=0.05, power=0.80)
        assert d > 1.0

    def test_one_sided_lower_than_two_sided(self) -> None:
        d_two = mde_paired_ttest(10, alternative="two-sided")
        d_one = mde_paired_ttest(10, alternative="one-sided")
        assert d_one < d_two

    def test_lower_alpha_increases_mde(self) -> None:
        d_05 = mde_paired_ttest(10, alpha=0.05)
        d_01 = mde_paired_ttest(10, alpha=0.01)
        assert d_01 > d_05


class TestRequiredSeeds:
    """Tests for required seed count calculation."""

    def test_large_effect_needs_few_seeds(self) -> None:
        m = required_seeds(0.8, alpha=0.05, power=0.80)
        assert 10 <= m <= 20

    def test_small_effect_needs_many_seeds(self) -> None:
        m = required_seeds(0.2, alpha=0.05, power=0.80)
        assert m > 100

    def test_medium_effect(self) -> None:
        m = required_seeds(0.5, alpha=0.05, power=0.80)
        assert 25 <= m <= 40


class TestPowerAtEffect:
    """Tests for achieved power calculation."""

    def test_power_increases_with_effect_size(self) -> None:
        p_small = power_at_effect(20, 0.2)
        p_large = power_at_effect(20, 0.8)
        assert p_large > p_small

    def test_power_increases_with_seeds(self) -> None:
        p5 = power_at_effect(5, 0.5)
        p50 = power_at_effect(50, 0.5)
        assert p50 > p5

    def test_power_at_mde_is_target_power(self) -> None:
        """Power at MDE should be approximately the target power."""
        d = mde_paired_ttest(20, alpha=0.05, power=0.80)
        p = power_at_effect(20, d, alpha=0.05)
        assert abs(p - 0.80) < 0.01

    def test_power_m5_d05_is_low(self) -> None:
        """With m=5 and d=0.5, power should be well below 80%."""
        p = power_at_effect(5, 0.5)
        assert p < 0.30


class TestPowerCurve:
    """Tests for power curve generation."""

    def test_returns_correct_range(self) -> None:
        curve = power_curve(0.5, m_range=(3, 10))
        assert len(curve) == 8
        assert curve[0]["m"] == 3
        assert curve[-1]["m"] == 10

    def test_power_monotonically_increases(self) -> None:
        curve = power_curve(0.5, m_range=(3, 50))
        powers = [c["power"] for c in curve]
        for i in range(1, len(powers)):
            assert powers[i] >= powers[i - 1]


class TestCliffsDeltaConversion:
    """Tests for Cliff's delta to Cohen's d conversion."""

    def test_zero_delta_gives_zero_d(self) -> None:
        assert cliffs_delta_to_cohens_d(0.0) == pytest.approx(0.0)

    def test_positive_delta_gives_positive_d(self) -> None:
        assert cliffs_delta_to_cohens_d(0.33) > 0

    def test_medium_delta_reasonable(self) -> None:
        """Cliff's delta 0.33 should map to roughly d ~ 0.35."""
        d = cliffs_delta_to_cohens_d(0.33)
        assert 0.3 < d < 0.4

    def test_near_one_saturates(self) -> None:
        d = cliffs_delta_to_cohens_d(0.99)
        assert d > 5.0


class TestMainCLI:
    """Tests for the CLI entry point."""

    def test_main_writes_json(self, tmp_path) -> None:
        from click.testing import CliRunner

        from cli.s2_calibrate.power_analysis import main

        out_rel = "test_power.json"
        # Patch PROJECT_ROOT so output goes to tmp_path
        import cli.s2_calibrate.power_analysis as pa_mod

        original_root = pa_mod.PROJECT_ROOT
        pa_mod.PROJECT_ROOT = tmp_path
        try:
            runner = CliRunner()
            result = runner.invoke(main, ["--output", out_rel, "--seeds", "5"])
            assert result.exit_code == 0, result.output
        finally:
            pa_mod.PROJECT_ROOT = original_root

        out_path = tmp_path / out_rel
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "minimum_detectable_effect" in data
        assert "power_at_standard_effects" in data
        assert "recommendations" in data
        assert data["parameters"]["m_seeds"] == 5
