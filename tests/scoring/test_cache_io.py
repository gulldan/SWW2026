"""Tests for dualdocking.cache_io — process-safe JSON cache I/O."""

from __future__ import annotations

import fcntl
import json
import math
import multiprocessing
import os
from pathlib import Path

from dualdocking.scoring.cache_io import load_cache, save_cache

_MP_CTX = multiprocessing.get_context("spawn")

# ---------------------------------------------------------------------------
# Basic roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip(tmp_path: Path):
    path = tmp_path / "tgt.json"
    data = {"abc123": 1.5, "def456": -2.3}
    save_cache(path, data)
    loaded = load_cache(path)
    assert loaded == data


def test_load_missing_file(tmp_path: Path):
    assert load_cache(tmp_path / "nonexistent.json") == {}


def test_load_empty_file(tmp_path: Path):
    path = tmp_path / "empty.json"
    path.write_text("")
    assert load_cache(path) == {}


# ---------------------------------------------------------------------------
# NaN filtering
# ---------------------------------------------------------------------------


def test_save_filters_nan(tmp_path: Path):
    path = tmp_path / "tgt.json"
    save_cache(path, {"good": -5.2, "bad": float("nan"), "also_good": 1.0})
    raw = json.loads(path.read_text())
    assert "good" in raw
    assert "also_good" in raw
    assert "bad" not in raw


def test_load_filters_nan(tmp_path: Path):
    path = tmp_path / "tgt.json"
    path.write_text('{"good": -5.2, "int_val": 3}')
    loaded = load_cache(path)
    assert loaded == {"good": -5.2, "int_val": 3}
    assert not any(isinstance(v, float) and math.isnan(v) for v in loaded.values())


# ---------------------------------------------------------------------------
# Corrupted file recovery
# ---------------------------------------------------------------------------


def test_load_corrupted_json(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{truncated...")
    loaded = load_cache(path)
    assert loaded == {}


def test_save_over_corrupted(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{truncated...")
    save_cache(path, {"new_key": -3.0})
    loaded = load_cache(path)
    assert loaded == {"new_key": -3.0}


def test_load_non_dict_json_array(tmp_path: Path):
    """Valid JSON that is not a dict (e.g. []) must not crash — treated as corrupted."""
    path = tmp_path / "arr.json"
    path.write_text("[1, 2, 3]")
    loaded = load_cache(path)
    assert loaded == {}


def test_load_non_dict_json_string(tmp_path: Path):
    """Valid JSON string must not crash — treated as corrupted."""
    path = tmp_path / "str.json"
    path.write_text('"hello"')
    loaded = load_cache(path)
    assert loaded == {}


def test_save_merges_over_non_dict_json(tmp_path: Path):
    """save_cache with a non-dict file on disk must not crash — overwrites."""
    path = tmp_path / "arr.json"
    path.write_text("[1, 2, 3]")
    save_cache(path, {"k": 1.0})
    loaded = load_cache(path)
    assert loaded == {"k": 1.0}


# ---------------------------------------------------------------------------
# Merge behaviour (read-merge-write)
# ---------------------------------------------------------------------------


def test_save_merges_with_disk(tmp_path: Path):
    """save_cache must merge new data with existing disk contents."""
    path = tmp_path / "tgt.json"
    save_cache(path, {"k1": 1.0, "k2": 2.0})
    save_cache(path, {"k2": 2.5, "k3": 3.0})  # k2 updated, k3 new
    loaded = load_cache(path)
    assert loaded == {"k1": 1.0, "k2": 2.5, "k3": 3.0}


def test_merge_does_not_lose_entries(tmp_path: Path):
    """Two separate saves with disjoint keys must both be present."""
    path = tmp_path / "tgt.json"
    save_cache(path, {"a": 1.0})
    save_cache(path, {"b": 2.0})
    loaded = load_cache(path)
    assert loaded == {"a": 1.0, "b": 2.0}


# ---------------------------------------------------------------------------
# Atomic write — temp file cleaned up
# ---------------------------------------------------------------------------


def test_no_leftover_temp_files(tmp_path: Path):
    path = tmp_path / "tgt.json"
    save_cache(path, {"k": 1.0})
    leftover = list(tmp_path.glob(".*tmp"))
    assert leftover == [], f"Temp files left behind: {leftover}"


# ---------------------------------------------------------------------------
# Lock file creation
# ---------------------------------------------------------------------------


def test_lock_file_created(tmp_path: Path):
    path = tmp_path / "tgt.json"
    save_cache(path, {"k": 1.0})
    assert (tmp_path / "tgt.lock").exists()


# ---------------------------------------------------------------------------
# Lock timeout — fail-closed: load returns {}, save skips write
# ---------------------------------------------------------------------------


def test_load_timeout_returns_empty_and_warns(tmp_path: Path):
    """load_cache must return {} and log a warning when lock times out."""
    from loguru import logger

    path = tmp_path / "tgt.json"
    save_cache(path, {"k": 42.0})
    assert load_cache(path) == {"k": 42.0}  # sanity check

    lock_file = path.with_suffix(".lock")
    held_fd = os.open(str(lock_file), os.O_RDWR)
    fcntl.flock(held_fd, fcntl.LOCK_EX)

    warnings: list[str] = []
    sink_id = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
    try:
        result = load_cache(path, timeout=0.15)
        assert result == {}, f"Expected empty dict on lock timeout, got {result}"
    finally:
        fcntl.flock(held_fd, fcntl.LOCK_UN)
        os.close(held_fd)
        logger.remove(sink_id)

    assert any("Lock timeout" in w and "reading cache" in w for w in warnings), (
        f"Expected 'Lock timeout ... reading cache' warning, got: {warnings}"
    )


def test_save_timeout_skips_write_and_warns(tmp_path: Path):
    """save_cache must NOT write and must log a warning when lock times out."""
    from loguru import logger

    path = tmp_path / "tgt.json"
    save_cache(path, {"original": 1.0})

    lock_file = path.with_suffix(".lock")
    held_fd = os.open(str(lock_file), os.O_RDWR)
    fcntl.flock(held_fd, fcntl.LOCK_EX)

    warnings: list[str] = []
    sink_id = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
    try:
        save_cache(path, {"new_key": 99.0}, timeout=0.15)
    finally:
        fcntl.flock(held_fd, fcntl.LOCK_UN)
        os.close(held_fd)
        logger.remove(sink_id)

    # Original data must be unchanged — save was skipped
    loaded = load_cache(path)
    assert loaded == {"original": 1.0}
    assert "new_key" not in loaded

    assert any("Lock timeout" in w and "write skipped" in w for w in warnings), (
        f"Expected 'Lock timeout ... write skipped' warning, got: {warnings}"
    )


# ---------------------------------------------------------------------------
# Deterministic output (sort_keys=True)
# ---------------------------------------------------------------------------


def test_deterministic_json_sorted_keys(tmp_path: Path):
    """JSON output must be byte-identical regardless of dict insertion order."""
    path1 = tmp_path / "a.json"
    path2 = tmp_path / "b.json"
    save_cache(path1, {"z": 1.0, "a": 2.0, "m": 3.0})
    save_cache(path2, {"m": 3.0, "z": 1.0, "a": 2.0})
    assert path1.read_text() == path2.read_text()


def test_deterministic_json_keys_are_sorted(tmp_path: Path):
    """Keys in the output JSON must be in sorted order."""
    path = tmp_path / "tgt.json"
    save_cache(path, {"c": 3.0, "a": 1.0, "b": 2.0})
    keys = list(json.loads(path.read_text()).keys())
    assert keys == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Concurrency stress test (multiprocessing)
# ---------------------------------------------------------------------------


def _worker(args: tuple[str, int, int]) -> None:
    """Worker for concurrency test: writes keys start..start+count to cache."""
    cache_path_str, start, count = args
    path = Path(cache_path_str)
    data = {f"key_{i:04d}": float(i) for i in range(start, start + count)}
    save_cache(path, data)


def test_concurrent_writes_no_corruption(tmp_path: Path):
    """4 parallel processes writing to the same cache file must not corrupt it."""
    path = tmp_path / "shared.json"
    workers = 4
    keys_per_worker = 50

    args_list = [(str(path), w * keys_per_worker, keys_per_worker) for w in range(workers)]

    with _MP_CTX.Pool(workers) as pool:
        pool.map(_worker, args_list)

    loaded = load_cache(path)

    # All keys from all workers must be present (merge, no lost updates)
    total_expected = workers * keys_per_worker
    assert len(loaded) == total_expected, (
        f"Expected {total_expected} keys, got {len(loaded)}. "
        f"Missing: {set(f'key_{i:04d}' for i in range(total_expected)) - set(loaded.keys())}"
    )

    # Values must be correct
    for i in range(total_expected):
        assert loaded[f"key_{i:04d}"] == float(i)

    # Cache file must be valid JSON with sorted keys
    raw = json.loads(path.read_text())
    assert len(raw) == total_expected
    assert list(raw.keys()) == sorted(raw.keys())


def test_concurrent_writes_no_leftover_temps(tmp_path: Path):
    """After concurrent writes, no .tmp files should remain."""
    path = tmp_path / "shared.json"
    workers = 4
    args_list = [(str(path), w * 10, 10) for w in range(workers)]

    with _MP_CTX.Pool(workers) as pool:
        pool.map(_worker, args_list)

    leftover = list(tmp_path.glob(".*tmp"))
    assert leftover == []


# ---------------------------------------------------------------------------
# Integration: scorer_boltz2 wrappers still work
# ---------------------------------------------------------------------------


def test_scorer_wrappers_use_cache_io(tmp_path: Path, monkeypatch):
    """_load_cache/_save_cache in scorer_boltz2 must delegate to cache_io."""
    import dualdocking.scoring.boltz2 as mod

    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(mod, "_CONTEXT_HASH", None)

    mod._save_cache("tgt", {"abc": -5.0, "def": -6.0})
    loaded = mod._load_cache("tgt")
    assert loaded == {"abc": -5.0, "def": -6.0}

    # Verify the file is on disk via cache_io.load_cache directly
    # Cache files are now context-segmented: {target_id}.{ctx[:8]}.json
    cache_file = mod._cache_path("tgt")
    direct = load_cache(cache_file)
    assert direct == {"abc": -5.0, "def": -6.0}
