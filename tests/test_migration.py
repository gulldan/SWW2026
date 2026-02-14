"""Tests for cli/migrate_csv_to_parquet.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from cli.migrate_csv_to_parquet import _find_csvs, _migrate_one, _should_exclude, main


def _make_csv(path: Path, n: int = 3) -> None:
    """Create a small test CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"smiles": [f"C{'C' * i}O" for i in range(n)], "score": [-1.0 * i for i in range(n)]})
    df.to_csv(path, index=False)


class TestShouldExclude:
    def test_excludes_boltz_msa(self):
        assert _should_exclude(Path("output/cache/boltz_msa/target.csv"))

    def test_excludes_train(self):
        assert _should_exclude(Path("output/results/pair/mode/seed_42/freedpp_workspace/output/run/mols/train_000.csv"))

    def test_excludes_alert_collections(self):
        assert _should_exclude(Path("docker/freedpp/data/alert_collections.csv"))

    def test_allows_scored_joined(self):
        assert not _should_exclude(Path("output/results/pair/mode/seed_42/scored_joined.csv"))

    def test_allows_calibration(self):
        assert not _should_exclude(Path("output/data/calibration/scores/CDK2.csv"))


class TestFindCsvs:
    def test_finds_scored_joined(self, tmp_path):
        csv = tmp_path / "output" / "results" / "pair1" / "double" / "seed_42" / "scored_joined.csv"
        _make_csv(csv)
        found = _find_csvs(tmp_path)
        assert csv in found

    def test_finds_calibration_scores(self, tmp_path):
        csv = tmp_path / "output" / "data" / "calibration" / "scores" / "CDK2.csv"
        _make_csv(csv)
        found = _find_csvs(tmp_path)
        assert csv in found

    def test_finds_enrichment_actives(self, tmp_path):
        csv = tmp_path / "output" / "data" / "enrichment" / "scores" / "CDK2_actives.csv"
        _make_csv(csv)
        found = _find_csvs(tmp_path)
        assert csv in found

    def test_finds_all_results(self, tmp_path):
        csv = tmp_path / "report" / "data" / "all_results.csv"
        _make_csv(csv)
        found = _find_csvs(tmp_path)
        assert csv in found

    def test_ignores_train_csvs(self, tmp_path):
        csv = (
            tmp_path
            / "output"
            / "results"
            / "pair1"
            / "double"
            / "seed_42"
            / "freedpp_workspace"
            / "output"
            / "run"
            / "mols"
            / "train_000.csv"
        )
        _make_csv(csv)
        found = _find_csvs(tmp_path)
        assert csv not in found

    # --- Server layout (flat, no output/ prefix) ---

    def test_finds_scored_joined_flat(self, tmp_path):
        csv = tmp_path / "results" / "CDK2_ERK2" / "single_a" / "seed_42" / "scored_joined.csv"
        _make_csv(csv)
        found = _find_csvs(tmp_path)
        assert csv in found

    def test_finds_generated_flat(self, tmp_path):
        csv = tmp_path / "results" / "CDK2_ERK2" / "single_a" / "seed_42" / "generated.csv"
        _make_csv(csv)
        found = _find_csvs(tmp_path)
        assert csv in found

    def test_finds_scores_flat(self, tmp_path):
        csv_a = tmp_path / "results" / "CDK2_ERK2" / "single_a" / "seed_42" / "scores_A.csv"
        csv_b = tmp_path / "results" / "CDK2_ERK2" / "single_a" / "seed_42" / "scores_B.csv"
        _make_csv(csv_a)
        _make_csv(csv_b)
        found = _find_csvs(tmp_path)
        assert csv_a in found
        assert csv_b in found

    def test_finds_calibration_flat(self, tmp_path):
        csv = tmp_path / "data" / "calibration" / "scores" / "CDK2.csv"
        _make_csv(csv)
        found = _find_csvs(tmp_path)
        assert csv in found

    def test_finds_enrichment_actives_flat(self, tmp_path):
        csv = tmp_path / "data" / "enrichment" / "scores" / "CDK2_actives.csv"
        _make_csv(csv)
        found = _find_csvs(tmp_path)
        assert csv in found

    def test_no_duplicates_when_both_layouts(self, tmp_path):
        """If output/ is a symlink to the flat dir, files should not appear twice."""
        flat = tmp_path / "results" / "pair1" / "double" / "seed_42" / "scored_joined.csv"
        _make_csv(flat)
        # Create output/results as a symlink to results/
        (tmp_path / "output").mkdir()
        (tmp_path / "output" / "results").symlink_to(tmp_path / "results")
        found = _find_csvs(tmp_path)
        # Should find exactly one (deduplicated by resolved path)
        scored = [p for p in found if p.name == "scored_joined.csv"]
        assert len(scored) == 1


class TestMigrateOne:
    def test_converts_csv_to_parquet(self, tmp_path):
        csv = tmp_path / "data.csv"
        _make_csv(csv)
        status = _migrate_one(csv)
        assert status == "converted"
        assert csv.with_suffix(".parquet").exists()
        df = pd.read_parquet(csv.with_suffix(".parquet"))
        assert len(df) == 3

    def test_skips_existing_parquet(self, tmp_path):
        csv = tmp_path / "data.csv"
        _make_csv(csv)
        # Pre-create matching parquet
        df = pd.read_csv(csv)
        df.to_parquet(csv.with_suffix(".parquet"), index=False)
        status = _migrate_one(csv)
        assert status == "skipped:exists"

    def test_force_overwrites(self, tmp_path):
        csv = tmp_path / "data.csv"
        _make_csv(csv)
        df = pd.read_csv(csv)
        df.to_parquet(csv.with_suffix(".parquet"), index=False)
        status = _migrate_one(csv, force=True)
        assert status == "converted"

    def test_dry_run(self, tmp_path):
        csv = tmp_path / "data.csv"
        _make_csv(csv)
        status = _migrate_one(csv, dry_run=True)
        assert status == "would_convert"
        assert not csv.with_suffix(".parquet").exists()

    def test_verify_success(self, tmp_path):
        csv = tmp_path / "data.csv"
        _make_csv(csv)
        status = _migrate_one(csv, verify=True)
        assert status == "converted"

    def test_remove_csv_creates_backup(self, tmp_path):
        csv = tmp_path / "data.csv"
        _make_csv(csv)
        status = _migrate_one(csv, remove_csv=True)
        assert status == "converted"
        assert not csv.exists()
        assert csv.with_suffix(".csv.bak").exists()

    def test_empty_csv_converts(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        csv.write_text("smiles,score\n")
        status = _migrate_one(csv)
        assert status == "converted"
        df = pd.read_parquet(csv.with_suffix(".parquet"))
        assert len(df) == 0


class TestMainCli:
    def test_dry_run(self, tmp_path):
        csv = tmp_path / "output" / "results" / "pair1" / "double" / "seed_42" / "scored_joined.csv"
        _make_csv(csv)
        runner = CliRunner()
        result = runner.invoke(main, ["--project-root", str(tmp_path), "--dry-run"])
        assert result.exit_code == 0

    def test_full_migration(self, tmp_path):
        csv = tmp_path / "output" / "results" / "pair1" / "double" / "seed_42" / "scored_joined.csv"
        _make_csv(csv)
        runner = CliRunner()
        result = runner.invoke(main, ["--project-root", str(tmp_path), "--verify"])
        assert result.exit_code == 0
        assert csv.with_suffix(".parquet").exists()

    def test_no_csvs_found(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["--project-root", str(tmp_path)])
        assert result.exit_code == 0
