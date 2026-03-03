"""Tests for dualdocking.run_manifest — lifecycle tracking for experiment runs."""

from __future__ import annotations

import json
from pathlib import Path

from dualdocking.reporting.run_manifest import (
    finalize_manifest,
    init_manifest,
    load_manifest,
    manifest_path,
    skip_task,
    update_task,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_configs(tmp_path: Path) -> Path:
    """Create minimal config files and return the config directory."""
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "pairs.yaml").write_text("pairs: {}")
    (cfg / "experiment.yaml").write_text("seeds: [42]")
    (cfg / "metrics.yaml").write_text("thresholds: {}")
    (cfg / "normalization.yaml").write_text("targets: {}")
    return cfg


TASKS = [("CDK2_ERK2", "single_a", 42), ("CDK2_ERK2", "double", 137)]


# ---------------------------------------------------------------------------
# manifest_path
# ---------------------------------------------------------------------------


def test_manifest_path():
    p = manifest_path(Path("/results"), "run01")
    assert p == Path("/results/run_manifest_run01.json")


# ---------------------------------------------------------------------------
# init_manifest
# ---------------------------------------------------------------------------


def test_init_creates_valid_json(tmp_path: Path):
    cfg = _make_configs(tmp_path)
    mf = tmp_path / "results" / "run_manifest_test.json"

    doc = init_manifest(mf, run_id="test", config_dir=cfg, tasks=TASKS)

    assert mf.exists()
    raw = json.loads(mf.read_text())
    assert raw["manifest_version"] == "1.0.0"
    assert raw["run_id"] == "test"
    assert raw["status"] == "running"
    assert raw["started_at"] is not None
    assert raw["completed_at"] is None
    assert len(raw["tasks"]) == 2
    assert raw["summary"]["total"] == 2
    assert raw["summary"]["completed"] == 0
    # Return value matches file contents
    assert doc["run_id"] == raw["run_id"]


def test_init_records_config_hashes(tmp_path: Path):
    cfg = _make_configs(tmp_path)
    mf = tmp_path / "manifest.json"

    doc = init_manifest(mf, run_id="r1", config_dir=cfg, tasks=TASKS)

    assert "pairs.yaml" in doc["config_hashes"]
    assert len(doc["config_hashes"]["pairs.yaml"]) == 64  # SHA-256 hex
    assert doc["config_hashes"]["pairs.yaml"] != ""


def test_init_missing_config_gives_empty_hash(tmp_path: Path):
    cfg = tmp_path / "empty_configs"
    cfg.mkdir()
    mf = tmp_path / "manifest.json"

    doc = init_manifest(mf, run_id="r1", config_dir=cfg, tasks=TASKS)

    assert doc["config_hashes"]["pairs.yaml"] == ""


def test_init_records_execution_params(tmp_path: Path):
    cfg = _make_configs(tmp_path)
    mf = tmp_path / "manifest.json"

    doc = init_manifest(mf, run_id="r1", config_dir=cfg, tasks=TASKS, parallel=4, resume=True)

    assert doc["execution"]["parallel"] == 4
    assert doc["execution"]["resume"] is True


def test_init_task_structure(tmp_path: Path):
    cfg = _make_configs(tmp_path)
    mf = tmp_path / "manifest.json"

    doc = init_manifest(mf, run_id="r1", config_dir=cfg, tasks=TASKS)

    task = doc["tasks"][0]
    assert task["task_id"] == "CDK2_ERK2/single_a/seed_42"
    assert task["pair"] == "CDK2_ERK2"
    assert task["mode"] == "single_a"
    assert task["seed"] == 42
    assert task["status"] == "pending"
    assert task["stages"] == {}


# ---------------------------------------------------------------------------
# update_task
# ---------------------------------------------------------------------------


def test_update_task_generate_running(tmp_path: Path):
    cfg = _make_configs(tmp_path)
    mf = tmp_path / "manifest.json"
    init_manifest(mf, run_id="r1", config_dir=cfg, tasks=TASKS)

    update_task(mf, task_id="CDK2_ERK2/single_a/seed_42", stage="generate", status="running")

    doc = load_manifest(mf)
    task = doc["tasks"][0]
    assert task["stages"]["generate"]["status"] == "running"
    assert task["status"] == "running"


def test_update_task_generate_completed_with_duration(tmp_path: Path):
    cfg = _make_configs(tmp_path)
    mf = tmp_path / "manifest.json"
    init_manifest(mf, run_id="r1", config_dir=cfg, tasks=TASKS)

    update_task(mf, task_id="CDK2_ERK2/single_a/seed_42", stage="generate", status="completed", duration_seconds=123.45)

    doc = load_manifest(mf)
    task = doc["tasks"][0]
    assert task["stages"]["generate"]["duration_seconds"] == 123.5  # rounded to 1 decimal


def test_update_task_full_lifecycle(tmp_path: Path):
    cfg = _make_configs(tmp_path)
    mf = tmp_path / "manifest.json"
    init_manifest(mf, run_id="r1", config_dir=cfg, tasks=TASKS)
    tid = "CDK2_ERK2/single_a/seed_42"

    update_task(mf, task_id=tid, stage="generate", status="completed", duration_seconds=100)
    update_task(mf, task_id=tid, stage="score", status="completed", duration_seconds=200)

    doc = load_manifest(mf)
    task = doc["tasks"][0]
    assert task["status"] == "completed"
    assert doc["summary"]["completed"] == 1


def test_update_task_failure_records_error(tmp_path: Path):
    cfg = _make_configs(tmp_path)
    mf = tmp_path / "manifest.json"
    init_manifest(mf, run_id="r1", config_dir=cfg, tasks=TASKS)
    tid = "CDK2_ERK2/single_a/seed_42"

    update_task(mf, task_id=tid, stage="generate", status="failed", error="Command failed (rc=1)")

    doc = load_manifest(mf)
    task = doc["tasks"][0]
    assert task["status"] == "failed"
    assert "Command failed" in task["stages"]["generate"]["error"]
    assert doc["summary"]["failed"] == 1


def test_update_task_score_failure_after_generate_success(tmp_path: Path):
    cfg = _make_configs(tmp_path)
    mf = tmp_path / "manifest.json"
    init_manifest(mf, run_id="r1", config_dir=cfg, tasks=TASKS)
    tid = "CDK2_ERK2/single_a/seed_42"

    update_task(mf, task_id=tid, stage="generate", status="completed", duration_seconds=100)
    update_task(mf, task_id=tid, stage="score", status="failed", error="OOM")

    doc = load_manifest(mf)
    task = doc["tasks"][0]
    assert task["status"] == "failed"


def test_update_nonexistent_task_is_noop(tmp_path: Path):
    cfg = _make_configs(tmp_path)
    mf = tmp_path / "manifest.json"
    init_manifest(mf, run_id="r1", config_dir=cfg, tasks=TASKS)

    # Should not raise
    update_task(mf, task_id="NONEXISTENT/single_a/seed_0", stage="generate", status="running")

    doc = load_manifest(mf)
    assert doc["summary"]["completed"] == 0  # no change


def test_update_on_missing_manifest_is_noop(tmp_path: Path):
    mf = tmp_path / "does_not_exist.json"
    # Should not raise
    update_task(mf, task_id="X/single_a/seed_42", stage="generate", status="running")


# ---------------------------------------------------------------------------
# skip_task
# ---------------------------------------------------------------------------


def test_skip_task(tmp_path: Path):
    cfg = _make_configs(tmp_path)
    mf = tmp_path / "manifest.json"
    init_manifest(mf, run_id="r1", config_dir=cfg, tasks=TASKS)

    skip_task(mf, task_id="CDK2_ERK2/single_a/seed_42")

    doc = load_manifest(mf)
    task = doc["tasks"][0]
    assert task["status"] == "skipped"
    assert task["stages"]["generate"]["status"] == "skipped"
    assert task["stages"]["score"]["status"] == "skipped"
    assert doc["summary"]["skipped"] == 1


def test_skip_all_tasks(tmp_path: Path):
    cfg = _make_configs(tmp_path)
    mf = tmp_path / "manifest.json"
    init_manifest(mf, run_id="r1", config_dir=cfg, tasks=TASKS)

    for task in TASKS:
        tid = f"{task[0]}/{task[1]}/seed_{task[2]}"
        skip_task(mf, task_id=tid)

    doc = load_manifest(mf)
    assert doc["summary"]["skipped"] == 2
    assert doc["summary"]["completed"] == 0


# ---------------------------------------------------------------------------
# finalize_manifest
# ---------------------------------------------------------------------------


def test_finalize_sets_completed(tmp_path: Path):
    cfg = _make_configs(tmp_path)
    mf = tmp_path / "manifest.json"
    init_manifest(mf, run_id="r1", config_dir=cfg, tasks=TASKS)

    finalize_manifest(mf, status="completed")

    doc = load_manifest(mf)
    assert doc["status"] == "completed"
    assert doc["completed_at"] is not None
    assert "total_duration_seconds" in doc


def test_finalize_sets_failed(tmp_path: Path):
    cfg = _make_configs(tmp_path)
    mf = tmp_path / "manifest.json"
    init_manifest(mf, run_id="r1", config_dir=cfg, tasks=TASKS)

    finalize_manifest(mf, status="failed")

    doc = load_manifest(mf)
    assert doc["status"] == "failed"


def test_finalize_missing_manifest_is_noop(tmp_path: Path):
    mf = tmp_path / "nonexistent.json"
    # Should not raise
    finalize_manifest(mf, status="completed")


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------


def test_load_roundtrip(tmp_path: Path):
    cfg = _make_configs(tmp_path)
    mf = tmp_path / "manifest.json"
    original = init_manifest(mf, run_id="r1", config_dir=cfg, tasks=TASKS)

    loaded = load_manifest(mf)
    assert loaded["run_id"] == original["run_id"]
    assert loaded["manifest_version"] == original["manifest_version"]
    assert len(loaded["tasks"]) == len(original["tasks"])


def test_load_missing_returns_empty(tmp_path: Path):
    assert load_manifest(tmp_path / "nope.json") == {}


def test_load_corrupted_returns_empty(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{broken json")
    assert load_manifest(bad) == {}


def test_load_empty_file_returns_empty(tmp_path: Path):
    empty = tmp_path / "empty.json"
    empty.write_text("")
    assert load_manifest(empty) == {}


# ---------------------------------------------------------------------------
# Full lifecycle integration
# ---------------------------------------------------------------------------


def test_full_lifecycle_two_tasks(tmp_path: Path):
    """Simulate a complete run with two tasks: one succeeds, one fails."""
    cfg = _make_configs(tmp_path)
    mf = tmp_path / "manifest.json"
    init_manifest(mf, run_id="integration", config_dir=cfg, tasks=TASKS)

    # Task 1: success
    tid1 = "CDK2_ERK2/single_a/seed_42"
    update_task(mf, task_id=tid1, stage="generate", status="running")
    update_task(mf, task_id=tid1, stage="generate", status="completed", duration_seconds=50)
    update_task(mf, task_id=tid1, stage="score", status="running")
    update_task(mf, task_id=tid1, stage="score", status="completed", duration_seconds=100)

    # Task 2: generate succeeds, score fails
    tid2 = "CDK2_ERK2/double/seed_137"
    update_task(mf, task_id=tid2, stage="generate", status="completed", duration_seconds=60)
    update_task(mf, task_id=tid2, stage="score", status="failed", error="timeout")

    finalize_manifest(mf, status="failed")

    doc = load_manifest(mf)
    assert doc["status"] == "failed"
    assert doc["summary"]["completed"] == 1
    assert doc["summary"]["failed"] == 1
    assert doc["completed_at"] is not None


def test_resume_lifecycle(tmp_path: Path):
    """Simulate resume: task 1 skipped (already done), task 2 runs normally."""
    cfg = _make_configs(tmp_path)
    mf = tmp_path / "manifest.json"
    init_manifest(mf, run_id="resume_test", config_dir=cfg, tasks=TASKS, resume=True)

    # Task 1: fully skipped
    skip_task(mf, task_id="CDK2_ERK2/single_a/seed_42")

    # Task 2: generate skipped (resume), score runs
    tid2 = "CDK2_ERK2/double/seed_137"
    update_task(mf, task_id=tid2, stage="generate", status="skipped")
    update_task(mf, task_id=tid2, stage="score", status="completed", duration_seconds=80)

    finalize_manifest(mf, status="completed")

    doc = load_manifest(mf)
    assert doc["status"] == "completed"
    assert doc["execution"]["resume"] is True
    assert doc["summary"]["skipped"] == 1
    # Task 2 has generate skipped + score completed → task status = completed
    task2 = doc["tasks"][1]
    assert task2["status"] == "completed"


# ---------------------------------------------------------------------------
# Atomicity: file is never partially written
# ---------------------------------------------------------------------------


def test_atomic_write_creates_parent_dirs(tmp_path: Path):
    cfg = _make_configs(tmp_path)
    mf = tmp_path / "deep" / "nested" / "manifest.json"

    init_manifest(mf, run_id="nested", config_dir=cfg, tasks=TASKS)

    assert mf.exists()
    doc = load_manifest(mf)
    assert doc["run_id"] == "nested"


def test_error_truncated_to_2000_chars(tmp_path: Path):
    cfg = _make_configs(tmp_path)
    mf = tmp_path / "manifest.json"
    init_manifest(mf, run_id="r1", config_dir=cfg, tasks=TASKS)

    long_error = "x" * 5000
    update_task(mf, task_id="CDK2_ERK2/single_a/seed_42", stage="generate", status="failed", error=long_error)

    doc = load_manifest(mf)
    assert len(doc["tasks"][0]["stages"]["generate"]["error"]) == 2000
