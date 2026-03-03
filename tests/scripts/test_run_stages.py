"""Tests for cli/pipeline/run_stages.py."""

from __future__ import annotations

import yaml
from click.testing import CliRunner

from cli.pipeline.run_stages import main


def _pairs_cfg():
    return {
        "pairs": {
            "pair1": {
                "target_a": {"id": "A", "pdb_id": "1aaa", "organism": "Homo sapiens"},
                "target_b": {"id": "B", "pdb_id": "2bbb", "organism": "Homo sapiens"},
                "set": "core",
            }
        }
    }


def _experiment_cfg():
    return {"modes": ["single_a", "single_b", "double"], "seeds": [42]}


def _setup_configs(tmp_path):
    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "configs" / "pairs.yaml").write_text(yaml.dump(_pairs_cfg()))
    (tmp_path / "configs" / "experiment.yaml").write_text(yaml.dump(_experiment_cfg()))


def _create_scored_outputs(tmp_path):
    import pandas as pd

    for mode in ("single_a", "single_b", "double"):
        run_dir = tmp_path / "output" / "results" / "pair1" / mode / "seed_42"
        run_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"smiles": ["CCO"], "affinity_a": [-1.0], "affinity_b": [-1.0]})
        df.to_parquet(run_dir / "scored_joined.parquet", index=False)


def _create_target_pdbs(tmp_path):
    targets = tmp_path / "output" / "data" / "targets" / "pair1"
    (targets / "A").mkdir(parents=True, exist_ok=True)
    (targets / "B").mkdir(parents=True, exist_ok=True)
    (targets / "A" / "1aaa.pdb").write_text("MODEL\nENDMDL\n")
    (targets / "B" / "2bbb.pdb").write_text("MODEL\nENDMDL\n")


def test_list_stages():
    runner = CliRunner()
    result = runner.invoke(main, ["--list-stages"])
    assert result.exit_code == 0
    assert "fetch" in result.output
    assert "calibrate" in result.output
    assert "generate" in result.output
    assert "score" in result.output
    assert "analyze" in result.output
    assert "report" in result.output
    assert "audit" in result.output


def test_no_stages_gives_error():
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 1


def test_unknown_stage_gives_error(tmp_path, monkeypatch):
    """Unknown stage name should fail with error message."""
    from dualdocking.pipeline import config as config_mod

    # Mock config loading to use tmp_path
    def _mock_from_config_dir(config_dir):
        from dualdocking.pipeline.config import PipelineConfig

        return PipelineConfig(
            project_root=tmp_path,
            pairs=_pairs_cfg()["pairs"],
            modes=["single_a"],
            seeds=[42],
        )

    monkeypatch.setattr(config_mod.PipelineConfig, "from_config_dir", staticmethod(_mock_from_config_dir))

    runner = CliRunner()
    result = runner.invoke(main, ["--stages", "nonexistent"])
    assert result.exit_code == 1
    assert "unknown stage" in result.output


def test_dry_run_analyze(tmp_path, monkeypatch):
    """Dry-run should work without actual files."""
    from dualdocking.pipeline import config as config_mod

    _setup_configs(tmp_path)

    # Mock config loading to use tmp_path
    def _mock_from_config_dir(config_dir):
        from dualdocking.pipeline.config import PipelineConfig

        return PipelineConfig(
            project_root=tmp_path,
            pairs=_pairs_cfg()["pairs"],
            modes=["single_a", "single_b", "double"],
            seeds=[42],
        )

    monkeypatch.setattr(config_mod.PipelineConfig, "from_config_dir", staticmethod(_mock_from_config_dir))

    # Create prereqs for analyze stage
    _create_scored_outputs(tmp_path)

    runner = CliRunner()
    result = runner.invoke(main, ["--stages", "analyze", "--dry-run"])
    assert result.exit_code == 0


def test_preflight_fails_on_missing_inputs(tmp_path, monkeypatch):
    """Preflight check should fail if required inputs are missing."""
    from dualdocking.pipeline import config as config_mod

    _setup_configs(tmp_path)

    # Mock config loading to use tmp_path
    def _mock_from_config_dir(config_dir):
        from dualdocking.pipeline.config import PipelineConfig

        return PipelineConfig(
            project_root=tmp_path,
            pairs=_pairs_cfg()["pairs"],
            modes=["single_a", "single_b", "double"],
            seeds=[42],
        )

    monkeypatch.setattr(config_mod.PipelineConfig, "from_config_dir", staticmethod(_mock_from_config_dir))

    # Don't create any scored files -> preflight should fail for analyze
    runner = CliRunner()
    result = runner.invoke(main, ["--stages", "analyze"])
    assert result.exit_code == 1


def test_skip_preflight(tmp_path, monkeypatch):
    """Skip-preflight flag should bypass input validation."""
    from dualdocking.pipeline import config as config_mod

    _setup_configs(tmp_path)

    # Mock config loading to use tmp_path
    def _mock_from_config_dir(config_dir):
        from dualdocking.pipeline.config import PipelineConfig

        return PipelineConfig(
            project_root=tmp_path,
            pairs=_pairs_cfg()["pairs"],
            modes=["single_a"],
            seeds=[42],
        )

    monkeypatch.setattr(config_mod.PipelineConfig, "from_config_dir", staticmethod(_mock_from_config_dir))

    runner = CliRunner()
    # analyze with --skip-preflight --dry-run should not fail preflight
    result = runner.invoke(main, ["--stages", "analyze", "--skip-preflight", "--dry-run"])
    assert result.exit_code == 0


def test_preflight_fails_md_when_target_pdbs_missing(tmp_path, monkeypatch):
    """MD preflight should fail if target PDB inputs are missing."""
    from dualdocking.pipeline import config as config_mod

    _setup_configs(tmp_path)
    _create_scored_outputs(tmp_path)

    def _mock_from_config_dir(config_dir):
        from dualdocking.pipeline.config import PipelineConfig

        return PipelineConfig(
            project_root=tmp_path,
            pairs=_pairs_cfg()["pairs"],
            modes=["single_a", "single_b", "double"],
            seeds=[42],
        )

    monkeypatch.setattr(config_mod.PipelineConfig, "from_config_dir", staticmethod(_mock_from_config_dir))

    runner = CliRunner()
    result = runner.invoke(main, ["--stages", "md"])
    assert result.exit_code == 1


def test_preflight_fails_gnina_when_binary_missing(tmp_path, monkeypatch):
    """GNINA preflight should fail when no GNINA binary is available."""
    from dualdocking.pipeline import config as config_mod
    from dualdocking.pipeline import checks as checks_mod

    _setup_configs(tmp_path)
    _create_scored_outputs(tmp_path)
    _create_target_pdbs(tmp_path)

    def _mock_from_config_dir(config_dir):
        from dualdocking.pipeline.config import PipelineConfig

        return PipelineConfig(
            project_root=tmp_path,
            pairs=_pairs_cfg()["pairs"],
            modes=["single_a", "single_b", "double"],
            seeds=[42],
        )

    monkeypatch.setattr(config_mod.PipelineConfig, "from_config_dir", staticmethod(_mock_from_config_dir))
    monkeypatch.setattr(checks_mod.shutil, "which", lambda _name: None)

    runner = CliRunner()
    result = runner.invoke(main, ["--stages", "gnina"])
    assert result.exit_code == 1
