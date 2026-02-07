"""Tests for scorer_boltz2 pure-logic helpers."""

import json
import math

import pytest
import yaml

import dualdocking.scorer_boltz2 as mod
from dualdocking.scorer_boltz2 import (
    _cache_key,
    _cache_path,
    _load_cache,
    _load_sequence,
    _msa_content_hash,
    _parse_affinity,
    _save_cache,
    _scoring_context_hash,
    _sequence_content_hash,
    _validate_smiles,
    _write_boltz_yaml,
    get_scoring_context_info,
    prune_stale_cache,
)


@pytest.fixture(autouse=True)
def _reset_scorer_state(monkeypatch):
    """Reset module-level cached state between tests."""
    monkeypatch.setattr(mod, "_CONTEXT_HASH", None)
    monkeypatch.setattr(mod, "_ACCELERATOR", None)
    mod._sequence_cache.clear()
    yield
    mod._sequence_cache.clear()


# ---------------------------------------------------------------------------
# Stable context fixture for tests that need _load_cache / _save_cache
# (they call _cache_path → _scoring_context_hash under the hood)
# ---------------------------------------------------------------------------


@pytest.fixture
def _stable_context(monkeypatch):
    """Pin all context inputs so _scoring_context_hash is deterministic."""
    monkeypatch.setattr(mod, "_boltz_version", lambda: "1.0.0")
    monkeypatch.setattr(mod, "_docker_digest", lambda: "sha256:abc")
    monkeypatch.setenv("BOLTZ_ACCELERATOR", "cpu")


# ===================================================================
# Original tests (updated for v2 cache key)
# ===================================================================


def test_cache_key_deterministic():
    k1 = _cache_key("target1", "CCO", ctx_hash="c", seq_hash="s", msa_hash="m")
    k2 = _cache_key("target1", "CCO", ctx_hash="c", seq_hash="s", msa_hash="m")
    assert k1 == k2


def test_cache_key_different():
    k1 = _cache_key("target1", "CCO", ctx_hash="c", seq_hash="s", msa_hash="m")
    k2 = _cache_key("target1", "CCCO", ctx_hash="c", seq_hash="s", msa_hash="m")
    assert k1 != k2


def test_load_save_cache(tmp_path, monkeypatch, _stable_context):
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    cache = {"abc123": 1.5, "def456": -2.3}
    _save_cache("tgt", cache)
    loaded = _load_cache("tgt")
    assert loaded == cache


def test_load_cache_missing(tmp_path, monkeypatch, _stable_context):
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    assert _load_cache("nonexistent") == {}


def test_load_sequence(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    fasta = ">target1\nMKTAYIAKQR\nQISFVKSH\n"
    (tmp_path / "target1.fasta").write_text(fasta)
    seq = _load_sequence("target1")
    assert seq == "MKTAYIAKQRQISFVKSH"


def test_load_sequence_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    try:
        _load_sequence("missing_target")
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError:
        pass


def test_validate_smiles_valid():
    assert _validate_smiles("CCO") is True


def test_validate_smiles_invalid():
    assert _validate_smiles("not_a_smiles!!!") is False


def test_write_boltz_yaml(tmp_path):
    out = tmp_path / "input.yaml"
    _write_boltz_yaml("MKTAYIAKQR", "CCO", out)
    data = yaml.safe_load(out.read_text())
    assert data["version"] == 1
    assert len(data["sequences"]) == 2
    assert "protein" in data["sequences"][0]
    assert "ligand" in data["sequences"][1]
    assert data["sequences"][1]["ligand"]["smiles"] == "CCO"
    assert "affinity" in data["properties"][0]


def test_write_boltz_yaml_with_msa(tmp_path):
    msa = tmp_path / "msa.csv"
    msa.write_text("dummy")
    out = tmp_path / "input.yaml"
    _write_boltz_yaml("MKTAYIAKQR", "CCO", out, msa_path=msa)
    data = yaml.safe_load(out.read_text())
    assert "msa" in data["sequences"][0]["protein"]


def test_parse_affinity(tmp_path):
    stem = "input"
    pred_dir = tmp_path / f"boltz_results_{stem}" / "predictions" / stem
    pred_dir.mkdir(parents=True)
    payload = {"affinity_pred_value": -6.42}
    (pred_dir / f"affinity_{stem}.json").write_text(json.dumps(payload))
    val = _parse_affinity(tmp_path, stem)
    assert abs(val - (-6.42)) < 1e-6


def test_parse_affinity_missing(tmp_path):
    try:
        _parse_affinity(tmp_path, "nonexistent")
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError:
        pass


def test_save_cache_filters_nan(tmp_path, monkeypatch, _stable_context):
    """NaN values must not be persisted to disk."""
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    cache = {"good": -5.2, "bad": float("nan"), "also_good": 1.0}
    _save_cache("tgt", cache)
    # Find the written file (context-segmented name)
    files = list(tmp_path.glob("tgt.*.json"))
    assert len(files) == 1
    raw = json.loads(files[0].read_text())
    assert "good" in raw
    assert "also_good" in raw
    assert "bad" not in raw


def test_load_cache_filters_nan(tmp_path, monkeypatch, _stable_context):
    """NaN values in cache file are filtered out on load."""
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    cache = {"good": -5.2, "int_val": 3}
    _save_cache("tgt2", cache)
    loaded = _load_cache("tgt2")
    assert loaded == {"good": -5.2, "int_val": 3}
    assert not any(isinstance(v, float) and math.isnan(v) for v in loaded.values())


# ===================================================================
# NEW: Cache key v2 tests
# ===================================================================


class TestCacheKeyContextParams:
    """Test that each context component changes the cache key."""

    def test_cache_key_changes_with_ctx_hash(self):
        k1 = _cache_key("t", "CCO", ctx_hash="aaa", seq_hash="s", msa_hash="m")
        k2 = _cache_key("t", "CCO", ctx_hash="bbb", seq_hash="s", msa_hash="m")
        assert k1 != k2

    def test_cache_key_changes_with_seq_hash(self):
        k1 = _cache_key("t", "CCO", ctx_hash="c", seq_hash="aaa", msa_hash="m")
        k2 = _cache_key("t", "CCO", ctx_hash="c", seq_hash="bbb", msa_hash="m")
        assert k1 != k2

    def test_cache_key_changes_with_msa_hash(self):
        k1 = _cache_key("t", "CCO", ctx_hash="c", seq_hash="s", msa_hash="aaa")
        k2 = _cache_key("t", "CCO", ctx_hash="c", seq_hash="s", msa_hash="bbb")
        assert k1 != k2


class TestScoringContextHash:
    """Test global context hash computation."""

    def test_scoring_context_hash_deterministic(self, monkeypatch):
        monkeypatch.setattr(mod, "_boltz_version", lambda: "2.0")
        monkeypatch.setattr(mod, "_docker_digest", lambda: "d1")
        monkeypatch.setenv("BOLTZ_ACCELERATOR", "cpu")
        h1 = _scoring_context_hash()
        # Reset to force recomputation
        monkeypatch.setattr(mod, "_CONTEXT_HASH", None)
        h2 = _scoring_context_hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_scoring_context_hash_cached(self, monkeypatch):
        monkeypatch.setattr(mod, "_boltz_version", lambda: "2.0")
        monkeypatch.setattr(mod, "_docker_digest", lambda: "d1")
        monkeypatch.setenv("BOLTZ_ACCELERATOR", "cpu")
        h1 = _scoring_context_hash()
        # Mutate inputs — should still return cached value
        monkeypatch.setattr(mod, "_boltz_version", lambda: "99.0")
        h2 = _scoring_context_hash()
        assert h1 == h2

    def test_scoring_context_hash_changes_with_boltz_version(self, monkeypatch):
        monkeypatch.setattr(mod, "_docker_digest", lambda: "d1")
        monkeypatch.setenv("BOLTZ_ACCELERATOR", "cpu")

        monkeypatch.setattr(mod, "_boltz_version", lambda: "1.0")
        h1 = _scoring_context_hash()

        monkeypatch.setattr(mod, "_CONTEXT_HASH", None)
        monkeypatch.setattr(mod, "_boltz_version", lambda: "2.0")
        h2 = _scoring_context_hash()

        assert h1 != h2

    def test_scoring_context_hash_changes_with_cli_args(self, monkeypatch):
        monkeypatch.setattr(mod, "_boltz_version", lambda: "1.0")
        monkeypatch.setattr(mod, "_docker_digest", lambda: "d1")
        monkeypatch.setenv("BOLTZ_ACCELERATOR", "cpu")

        monkeypatch.setattr(mod, "_BOLTZ_CLI_ARGS_BASE", ["--model", "boltz2"])
        h1 = _scoring_context_hash()

        monkeypatch.setattr(mod, "_CONTEXT_HASH", None)
        monkeypatch.setattr(mod, "_BOLTZ_CLI_ARGS_BASE", ["--model", "boltz2", "--extra"])
        h2 = _scoring_context_hash()

        assert h1 != h2

    def test_scoring_context_hash_changes_with_accelerator(self, monkeypatch):
        monkeypatch.setattr(mod, "_boltz_version", lambda: "1.0")
        monkeypatch.setattr(mod, "_docker_digest", lambda: "d1")

        monkeypatch.setenv("BOLTZ_ACCELERATOR", "cpu")
        h1 = _scoring_context_hash()

        monkeypatch.setattr(mod, "_CONTEXT_HASH", None)
        monkeypatch.setenv("BOLTZ_ACCELERATOR", "gpu")
        h2 = _scoring_context_hash()

        assert h1 != h2

    def test_scoring_context_hash_changes_with_docker_digest(self, monkeypatch):
        monkeypatch.setattr(mod, "_boltz_version", lambda: "1.0")
        monkeypatch.setenv("BOLTZ_ACCELERATOR", "cpu")

        monkeypatch.setattr(mod, "_docker_digest", lambda: "digest_v1")
        h1 = _scoring_context_hash()

        monkeypatch.setattr(mod, "_CONTEXT_HASH", None)
        monkeypatch.setattr(mod, "_docker_digest", lambda: "digest_v2")
        h2 = _scoring_context_hash()

        assert h1 != h2


class TestContentHashes:
    """Test per-target content hash functions."""

    def test_msa_content_hash_no_file(self):
        assert _msa_content_hash(None) == "no_msa"

    def test_msa_content_hash_with_file(self, tmp_path):
        msa = tmp_path / "msa.csv"
        msa.write_bytes(b"header\nACGT\n")
        h = _msa_content_hash(msa)
        assert len(h) == 16
        # Same file → same hash
        assert _msa_content_hash(msa) == h

    def test_sequence_content_hash(self):
        h1 = _sequence_content_hash("MKTAYIAKQR")
        h2 = _sequence_content_hash("MKTAYIAKQR")
        h3 = _sequence_content_hash("DIFFERENT")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 16


class TestCachePath:
    """Test context-segmented cache file paths."""

    def test_cache_path_segmented(self, monkeypatch):
        monkeypatch.setattr(mod, "_boltz_version", lambda: "1.0")
        monkeypatch.setattr(mod, "_docker_digest", lambda: "d1")
        monkeypatch.setenv("BOLTZ_ACCELERATOR", "cpu")

        p = _cache_path("CDK2")
        ctx = _scoring_context_hash()
        assert p.name == f"CDK2.{ctx[:8]}.json"
        assert p.parent == mod.CACHE_DIR


class TestPruneStaleCache:
    """Test stale cache pruning."""

    def test_prune_stale_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(mod, "_boltz_version", lambda: "1.0")
        monkeypatch.setattr(mod, "_docker_digest", lambda: "d1")
        monkeypatch.setenv("BOLTZ_ACCELERATOR", "cpu")

        ctx = _scoring_context_hash()
        current_prefix = ctx[:8]

        # Create current-context file
        current_file = tmp_path / f"CDK2.{current_prefix}.json"
        current_file.write_text("{}")

        # Create old-context files
        old1 = tmp_path / "CDK2.deadbeef.json"
        old1.write_text("{}")
        old2 = tmp_path / "ERK2.12345678.json"
        old2.write_text("{}")
        # Legacy file (no context segment)
        legacy = tmp_path / "CDK2.json"
        legacy.write_text("{}")

        removed = prune_stale_cache()
        assert removed == 3
        assert current_file.exists()
        assert not old1.exists()
        assert not old2.exists()
        assert not legacy.exists()


class TestOldCacheNotReusedE2E:
    """Integration test: changing context invalidates cache."""

    def test_old_cache_not_reused_e2e(self, tmp_path, monkeypatch):
        # Setup paths
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        msa_dir = cache_dir / "boltz_msa"
        msa_dir.mkdir()

        monkeypatch.setattr(mod, "CACHE_DIR", cache_dir)
        monkeypatch.setattr(mod, "DATA_DIR", data_dir)
        monkeypatch.setattr(mod, "MSA_CACHE_DIR", msa_dir)

        # Create FASTA + MSA for target
        (data_dir / "tgt.fasta").write_text(">tgt\nMKTAY\n")
        (msa_dir / "tgt.csv").write_text("header\nACGT\n")

        # --- Phase 1: populate cache with context v1 ---
        monkeypatch.setattr(mod, "_boltz_version", lambda: "1.0")
        monkeypatch.setattr(mod, "_docker_digest", lambda: "d1")
        monkeypatch.setenv("BOLTZ_ACCELERATOR", "cpu")

        ctx1 = _scoring_context_hash()
        seq_hash = _sequence_content_hash("MKTAY")
        msa_hash = _msa_content_hash(msa_dir / "tgt.csv")
        old_key = _cache_key("tgt", "CCO", ctx_hash=ctx1, seq_hash=seq_hash, msa_hash=msa_hash)

        # Write an entry to the old context cache file
        old_cache_file = cache_dir / f"tgt.{ctx1[:8]}.json"
        old_cache_file.write_text(json.dumps({old_key: -5.0}))

        # --- Phase 2: change Boltz version → new context ---
        monkeypatch.setattr(mod, "_CONTEXT_HASH", None)
        monkeypatch.setattr(mod, "_boltz_version", lambda: "2.0")

        ctx2 = _scoring_context_hash()
        assert ctx1 != ctx2

        new_cache_file = cache_dir / f"tgt.{ctx2[:8]}.json"

        # Mock _score_batch_internal to return known scores
        def fake_score_batch_internal(target_id, smiles_list, sequence, msa_path):
            return [-7.5] * len(smiles_list)

        monkeypatch.setattr(mod, "_score_batch_internal", fake_score_batch_internal)

        # Call score_batch — should NOT find old cache, should call internal scorer
        scores = mod.score_batch("tgt", ["CCO"])
        assert scores[0] == pytest.approx(-7.5)

        # Old file untouched
        old_data = json.loads(old_cache_file.read_text())
        assert old_data == {old_key: -5.0}

        # New file created with new entry
        assert new_cache_file.exists()
        new_data = json.loads(new_cache_file.read_text())
        new_key = _cache_key("tgt", "CCO", ctx_hash=ctx2, seq_hash=seq_hash, msa_hash=msa_hash)
        assert new_key in new_data
        assert new_data[new_key] == pytest.approx(-7.5)


class TestGetScoringContextInfo:
    """Test the public metadata accessor."""

    def test_get_scoring_context_info(self, monkeypatch):
        monkeypatch.setattr(mod, "_boltz_version", lambda: "2.2.1")
        monkeypatch.setattr(mod, "_docker_digest", lambda: "sha256:abc123")
        monkeypatch.setenv("BOLTZ_ACCELERATOR", "cpu")

        info = get_scoring_context_info()
        assert info["boltz_version"] == "2.2.1"
        assert info["docker_digest"] == "sha256:abc123"
        assert info["accelerator"] == "cpu"
        assert "cache_context_hash" in info
        assert "cli_args_hash" in info
        assert len(info["cache_context_hash"]) == 16
        assert len(info["cli_args_hash"]) == 16
