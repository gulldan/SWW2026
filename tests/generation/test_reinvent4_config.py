"""Tests for REINVENT4 TOML configuration generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from dualdocking.generation.reinvent4.config import Reinvent4TOMLConfig, generate_toml, write_toml
from dualdocking.generation.reinvent4.scoring import DockingTarget


@pytest.fixture()
def target_a(tmp_path: Path) -> DockingTarget:
    receptor = tmp_path / "receptor_a.pdbqt"
    receptor.write_text("ATOM mock\n")
    return DockingTarget(
        pdbqt_path=receptor,
        center="1.5,2.5,3.5",
        size="20.0,20.0,20.0",
        norm_mean=-7.42,
        norm_std=0.705,
    )


@pytest.fixture()
def target_b(tmp_path: Path) -> DockingTarget:
    receptor = tmp_path / "receptor_b.pdbqt"
    receptor.write_text("ATOM mock\n")
    return DockingTarget(
        pdbqt_path=receptor,
        center="4.0,5.0,6.0",
        size="22.0,22.0,22.0",
        norm_mean=-8.16,
        norm_std=0.876,
    )


class TestGenerateToml:
    def test_single_target(self, target_a: DockingTarget) -> None:
        """Single-target config contains correct parameters."""
        cfg = Reinvent4TOMLConfig(target_a=target_a, reward_mode="single_a", n_steps=100)
        toml = generate_toml(cfg)

        assert 'run_type = "staged_learning"' in toml
        assert "max_steps = 100" in toml
        assert 'reward_mode = "single_a"' in toml
        assert str(target_a.pdbqt_path) in toml
        assert "norm_mean_a = -7.42" in toml
        assert "norm_std_a = 0.705" in toml

    def test_dual_target(self, target_a: DockingTarget, target_b: DockingTarget) -> None:
        """Dual-target config includes target_b parameters."""
        cfg = Reinvent4TOMLConfig(target_a=target_a, target_b=target_b, reward_mode="double")
        toml = generate_toml(cfg)

        assert str(target_b.pdbqt_path) in toml
        assert "norm_mean_b = -8.16" in toml
        assert "norm_std_b = 0.876" in toml

    def test_no_target_a_raises(self) -> None:
        """Missing target_a raises ValueError."""
        cfg = Reinvent4TOMLConfig()
        with pytest.raises(ValueError, match="target_a is required"):
            generate_toml(cfg)

    def test_dap_algorithm(self, target_a: DockingTarget) -> None:
        """DAP algorithm is set correctly."""
        cfg = Reinvent4TOMLConfig(target_a=target_a, algorithm="dap")
        toml = generate_toml(cfg)
        assert 'type = "dap"' in toml

    def test_diversity_filter_enabled(self, target_a: DockingTarget) -> None:
        """Diversity filter section present when enabled."""
        cfg = Reinvent4TOMLConfig(target_a=target_a, use_diversity_filter=True)
        toml = generate_toml(cfg)
        assert 'type = "IdenticalMurckoScaffold"' in toml

    def test_diversity_filter_disabled(self, target_a: DockingTarget) -> None:
        """Diversity filter section omitted when disabled."""
        cfg = Reinvent4TOMLConfig(target_a=target_a, use_diversity_filter=False)
        toml = generate_toml(cfg)
        assert "[diversity_filter]" not in toml

    def test_batch_size_in_output(self, target_a: DockingTarget) -> None:
        """Batch size appears in parameters section."""
        cfg = Reinvent4TOMLConfig(target_a=target_a, batch_size=256)
        toml = generate_toml(cfg)
        assert "batch_size = 256" in toml

    def test_seed_in_output(self, target_a: DockingTarget) -> None:
        """Seed appears in config."""
        cfg = Reinvent4TOMLConfig(target_a=target_a, seed=137)
        toml = generate_toml(cfg)
        assert "seed = 137" in toml

    def test_vina_program(self, target_a: DockingTarget) -> None:
        """Vina program passed to scoring component."""
        cfg = Reinvent4TOMLConfig(target_a=target_a, vina_program="unidock")
        toml = generate_toml(cfg)
        assert 'vina_program = "unidock"' in toml

    def test_scoring_type(self, target_a: DockingTarget) -> None:
        """Scoring section includes aggregation type."""
        cfg = Reinvent4TOMLConfig(target_a=target_a)
        toml = generate_toml(cfg)
        assert 'type = "custom_sum"' in toml

    def test_absent_target_b_has_empty_strings(self, target_a: DockingTarget) -> None:
        """When target_b is None, receptor_b is empty string."""
        cfg = Reinvent4TOMLConfig(target_a=target_a, target_b=None)
        toml = generate_toml(cfg)
        assert 'receptor_b = ""' in toml
        assert 'center_b = ""' in toml
        assert 'size_b = ""' in toml


class TestWriteToml:
    def test_writes_file(self, tmp_path: Path, target_a: DockingTarget) -> None:
        """write_toml creates a file on disk."""
        cfg = Reinvent4TOMLConfig(target_a=target_a)
        path = write_toml(cfg, tmp_path / "sub" / "config.toml")
        assert path.exists()
        content = path.read_text()
        assert "staged_learning" in content

    def test_creates_parent_dirs(self, tmp_path: Path, target_a: DockingTarget) -> None:
        """write_toml creates parent directories."""
        cfg = Reinvent4TOMLConfig(target_a=target_a)
        path = write_toml(cfg, tmp_path / "a" / "b" / "c" / "config.toml")
        assert path.exists()
