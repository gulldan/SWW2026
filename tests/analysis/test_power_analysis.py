"""Tests for power_analysis module."""


from dualdocking.analysis.power_analysis import (
    detectable_effect,
    sensitivity_table,
    simulate_power_hierarchical,
)


class TestSimulatePowerHierarchical:
    def test_high_effect_high_power(self):
        # Large effect size -> power should be high
        pwr = simulate_power_hierarchical(
            n_seeds=5,
            n_molecules=200,
            effect_size=0.8,
            n_simulations=200,
            n_bootstrap=199,
            seed=42,
        )
        assert pwr > 0.5

    def test_zero_effect_low_power(self):
        # No effect -> power should be near alpha
        pwr = simulate_power_hierarchical(
            n_seeds=5,
            n_molecules=200,
            effect_size=0.0,
            n_simulations=200,
            n_bootstrap=199,
            seed=42,
        )
        assert pwr < 0.20  # should be near 0.05

    def test_more_seeds_more_power(self):
        pwr5 = simulate_power_hierarchical(
            n_seeds=5,
            n_molecules=200,
            effect_size=0.4,
            n_simulations=200,
            n_bootstrap=199,
            seed=42,
        )
        pwr15 = simulate_power_hierarchical(
            n_seeds=15,
            n_molecules=200,
            effect_size=0.4,
            n_simulations=200,
            n_bootstrap=199,
            seed=42,
        )
        assert pwr15 >= pwr5

    def test_returns_float_in_unit_interval(self):
        pwr = simulate_power_hierarchical(
            n_seeds=5,
            n_molecules=100,
            effect_size=0.5,
            n_simulations=50,
            n_bootstrap=99,
        )
        assert isinstance(pwr, float)
        assert 0.0 <= pwr <= 1.0


class TestDetectableEffect:
    def test_returns_positive(self):
        d = detectable_effect(5, n_simulations=100, n_bootstrap=99)
        assert d > 0

    def test_more_seeds_smaller_effect(self):
        d5 = detectable_effect(5, n_simulations=100, n_bootstrap=99)
        d15 = detectable_effect(15, n_simulations=100, n_bootstrap=99)
        assert d15 <= d5


class TestSensitivityTable:
    def test_returns_correct_length(self):
        table = sensitivity_table(
            seed_range=[3, 5, 10],
            n_simulations=50,
            n_bootstrap=49,
        )
        assert len(table) == 3

    def test_keys_present(self):
        table = sensitivity_table(
            seed_range=[5],
            n_simulations=50,
            n_bootstrap=49,
        )
        assert "n_seeds" in table[0]
        assert "min_cliffs_delta" in table[0]

    def test_decreasing_effect(self):
        table = sensitivity_table(
            seed_range=[3, 5, 10, 15],
            n_simulations=100,
            n_bootstrap=99,
        )
        effects = [row["min_cliffs_delta"] for row in table]
        for i in range(len(effects) - 1):
            assert effects[i] >= effects[i + 1]
