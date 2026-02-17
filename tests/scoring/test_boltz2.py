"""Tests for scorer_boltz2 pure-logic helpers."""

import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest
import yaml

import dualdocking.scoring.boltz2 as mod
from dualdocking.scoring.boltz2 import (
    _cache_key,
    _cache_path,
    _load_cache,
    _load_sequence,
    _msa_content_hash,
    _nim_health_check,
    _nim_predict_affinity,
    _parse_affinity,
    _save_cache,
    _score_batch_internal_nim,
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
    monkeypatch.setattr(mod, "_BOLTZ_BACKEND", "cli")
    monkeypatch.setattr(mod, "_NIM_ACTIVE_URLS", None)
    monkeypatch.setattr(mod, "_NIM_ACTIVE_URLS_KEY", None)
    monkeypatch.setattr(mod, "_NIM_ACTIVE_URLS_TIME", 0.0)
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
    monkeypatch.setattr(mod, "SCORE_CACHE_DIR", tmp_path)
    cache = {"abc123": 1.5, "def456": -2.3}
    _save_cache("tgt", cache)
    loaded = _load_cache("tgt")
    assert loaded == cache


def test_load_cache_missing(tmp_path, monkeypatch, _stable_context):
    monkeypatch.setattr(mod, "SCORE_CACHE_DIR", tmp_path)
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


def test_parse_affinity_file(tmp_path):
    affinity_path = tmp_path / "affinity_test.json"
    affinity_path.write_text(json.dumps({"affinity_pred_value": -7.25}))
    assert mod._parse_affinity_file(affinity_path) == pytest.approx(-7.25)


def test_parse_affinity_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        mod._parse_affinity_file(tmp_path / "missing.json")


def test_require_msa_path_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "MSA_CACHE_DIR", tmp_path)
    expected = tmp_path / "tgt.csv"
    expected.write_text("msa")
    assert mod._require_msa_path("tgt") == expected


def test_require_msa_path_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "MSA_CACHE_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="Pre-computed MSA not found for tgt"):
        mod._require_msa_path("tgt")


def test_collect_valid_smiles_filters_and_neutralizes(monkeypatch):
    monkeypatch.setattr(mod, "_validate_smiles", lambda smi: smi != "invalid")
    monkeypatch.setattr(mod, "_neutralize_smiles", lambda smi: f"neutral:{smi}")
    pairs = mod._collect_valid_smiles(
        ["CCO", "invalid", "C[NH3+]"],
        warning_template="Invalid SMILES (>{} atoms): {}",
    )
    assert pairs == [(0, "neutral:CCO"), (2, "neutral:C[NH3+]")]


def test_nim_payload_has_expected_structure():
    payload = mod._nim_payload("MKTAY", "CCO")
    assert payload["polymers"] == [{"id": "A", "molecule_type": "protein", "sequence": "MKTAY"}]
    assert payload["ligands"] == [{"id": "B", "smiles": "CCO", "predict_affinity": True}]
    for key, value in mod._NIM_PREDICT_PARAMS.items():
        assert payload[key] == value


def test_save_cache_filters_nan(tmp_path, monkeypatch, _stable_context):
    """NaN values must not be persisted to disk."""
    monkeypatch.setattr(mod, "SCORE_CACHE_DIR", tmp_path)
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
    monkeypatch.setattr(mod, "SCORE_CACHE_DIR", tmp_path)
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
        assert p.parent == mod.SCORE_CACHE_DIR


class TestPruneStaleCache:
    """Test stale cache pruning."""

    def test_prune_stale_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "SCORE_CACHE_DIR", tmp_path)
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

        score_cache_dir = cache_dir / "boltz_scores"
        score_cache_dir.mkdir()
        monkeypatch.setattr(mod, "CACHE_DIR", cache_dir)
        monkeypatch.setattr(mod, "SCORE_CACHE_DIR", score_cache_dir)
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
        old_cache_file = score_cache_dir / f"tgt.{ctx1[:8]}.json"
        old_cache_file.write_text(json.dumps({old_key: -5.0}))

        # --- Phase 2: change Boltz version → new context ---
        monkeypatch.setattr(mod, "_CONTEXT_HASH", None)
        monkeypatch.setattr(mod, "_boltz_version", lambda: "2.0")

        ctx2 = _scoring_context_hash()
        assert ctx1 != ctx2

        new_cache_file = score_cache_dir / f"tgt.{ctx2[:8]}.json"

        # Mock _score_batch_internal to return known scores
        def fake_score_batch_internal(target_id, smiles_list, sequence, msa_path, *, device=None):
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


class TestBoltzEnvDevice:
    """Test _boltz_env device parameter for multi-GPU support."""

    def test_device_none_no_env(self, monkeypatch):
        """device=None + no env var → CUDA_VISIBLE_DEVICES not set."""
        monkeypatch.delenv("BOLTZ_CUDA_DEVICE", raising=False)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        env = mod._boltz_env(device=None)
        assert "CUDA_VISIBLE_DEVICES" not in env

    def test_device_none_with_env(self, monkeypatch):
        """device=None + BOLTZ_CUDA_DEVICE=1 → falls back to env."""
        monkeypatch.setenv("BOLTZ_CUDA_DEVICE", "1")
        env = mod._boltz_env(device=None)
        assert env["CUDA_VISIBLE_DEVICES"] == "1"

    def test_device_explicit_overrides_env(self, monkeypatch):
        """Explicit device="0" overrides BOLTZ_CUDA_DEVICE=1."""
        monkeypatch.setenv("BOLTZ_CUDA_DEVICE", "1")
        env = mod._boltz_env(device="0")
        assert env["CUDA_VISIBLE_DEVICES"] == "0"

    def test_device_explicit_no_env(self, monkeypatch):
        """Explicit device="1" without env var."""
        monkeypatch.delenv("BOLTZ_CUDA_DEVICE", raising=False)
        env = mod._boltz_env(device="1")
        assert env["CUDA_VISIBLE_DEVICES"] == "1"


class TestGetScoringContextInfo:
    """Test the public metadata accessor."""

    def test_get_scoring_context_info(self, monkeypatch):
        monkeypatch.setattr(mod, "_boltz_version", lambda: "2.2.1")
        monkeypatch.setattr(mod, "_docker_digest", lambda: "sha256:abc123")
        monkeypatch.setenv("BOLTZ_ACCELERATOR", "cpu")

        info = get_scoring_context_info()
        assert info["backend"] == "cli"
        assert info["boltz_version"] == "2.2.1"
        assert info["docker_digest"] == "sha256:abc123"
        assert info["accelerator"] == "cpu"
        assert "cache_context_hash" in info
        assert "cli_args_hash" in info
        assert len(info["cache_context_hash"]) == 16
        assert len(info["cli_args_hash"]) == 16


# ===================================================================
# NIM backend tests
# ===================================================================


class _NimHandler(BaseHTTPRequestHandler):
    """Fake NIM server for testing."""

    # Class-level response overrides — set by tests
    response_body: dict | None = None
    status_code: int = 200
    health_ready: bool = True

    def do_GET(self):
        if self.path == "/v1/health/ready":
            body = b"true" if self.health_ready else b"false"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _payload = self.rfile.read(length)
        self.send_response(self.status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        body = self.response_body or {}
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, *_args):
        pass  # suppress stderr


@pytest.fixture
def nim_server(monkeypatch):
    """Start a local HTTP server faking NIM and point module at it."""
    # Reset handler defaults
    _NimHandler.response_body = None
    _NimHandler.status_code = 200
    _NimHandler.health_ready = True

    server = HTTPServer(("127.0.0.1", 0), _NimHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}"
    monkeypatch.setattr(mod, "_BOLTZ_BACKEND", "nim")
    monkeypatch.setattr(mod, "_BOLTZ_NIM_URL", url)
    monkeypatch.setattr(mod, "_BOLTZ_NIM_URLS", [url])

    yield server
    server.shutdown()


class TestNimHealthCheck:
    def test_health_ok(self, nim_server):
        assert _nim_health_check() is True

    def test_health_not_ready(self, nim_server):
        _NimHandler.health_ready = False
        assert _nim_health_check() is False

    def test_health_no_server(self, monkeypatch):
        monkeypatch.setattr(mod, "_BOLTZ_NIM_URL", "http://127.0.0.1:1")
        monkeypatch.setattr(mod, "_BOLTZ_NIM_URLS", ["http://127.0.0.1:1"])
        assert _nim_health_check() is False

    def test_health_any_url_ready(self, nim_server, monkeypatch):
        alive_url = mod._BOLTZ_NIM_URLS[0]
        monkeypatch.setattr(mod, "_BOLTZ_NIM_URL", "http://127.0.0.1:1")
        monkeypatch.setattr(mod, "_BOLTZ_NIM_URLS", ["http://127.0.0.1:1", alive_url])
        assert _nim_health_check() is True


class TestNimPredictAffinity:
    def test_success_averages_two_models(self, nim_server):
        _NimHandler.response_body = {
            "affinities": {
                "B": {
                    "affinity_pred_value": [-5.0, -6.0],
                    "affinity_pic50": [5.0, 6.0],
                    "affinity_probability_binary": [0.8, 0.9],
                },
            },
        }
        score = _nim_predict_affinity("MKTAY", "CCO")
        assert score == pytest.approx(-5.5)

    def test_success_single_value(self, nim_server):
        _NimHandler.response_body = {
            "affinities": {"B": {"affinity_pred_value": [-4.2]}},
        }
        score = _nim_predict_affinity("MKTAY", "CCO")
        assert score == pytest.approx(-4.2)

    def test_http_error_returns_nan(self, nim_server):
        _NimHandler.status_code = 500
        _NimHandler.response_body = {"error": "OOM"}
        score = _nim_predict_affinity("MKTAY", "CCO")
        assert math.isnan(score)

    def test_missing_affinities_returns_nan(self, nim_server):
        _NimHandler.response_body = {"structures": []}
        score = _nim_predict_affinity("MKTAY", "CCO")
        assert math.isnan(score)

    def test_connection_refused_raises(self, monkeypatch):
        monkeypatch.setattr(mod, "_BOLTZ_NIM_URL", "http://127.0.0.1:1")
        monkeypatch.setattr(mod, "_BOLTZ_NIM_URLS", ["http://127.0.0.1:1"])
        with pytest.raises(RuntimeError, match="No healthy NIM URL"):
            _nim_predict_affinity("MKTAY", "CCO")


class TestNimScoreBatchInternal:
    def test_batch_scores_valid_molecules(self, nim_server):
        call_count = 0

        def make_response():
            nonlocal call_count
            call_count += 1
            return {"affinities": {"B": {"affinity_pred_value": [-3.0 - call_count]}}}

        orig_do_post = _NimHandler.do_POST

        def counting_post(self_handler):
            _NimHandler.response_body = make_response()
            orig_do_post(self_handler)

        _NimHandler.do_POST = counting_post

        scores = _score_batch_internal_nim("tgt", ["CCO", "CCCO", "C"], "MKTAY", None)
        assert len(scores) == 3
        assert all(not math.isnan(s) for s in scores)

        _NimHandler.do_POST = orig_do_post

    def test_batch_skips_invalid_smiles(self, nim_server):
        _NimHandler.response_body = {
            "affinities": {"B": {"affinity_pred_value": [-5.0]}},
        }
        scores = _score_batch_internal_nim("tgt", ["CCO", "not_a_smiles!!!"], "MKTAY", None)
        assert not math.isnan(scores[0])
        assert math.isnan(scores[1])

    def test_batch_filters_unavailable_url(self, nim_server, monkeypatch):
        alive_url = mod._BOLTZ_NIM_URLS[0]
        monkeypatch.setattr(mod, "_BOLTZ_NIM_URLS", ["http://127.0.0.1:1", alive_url])
        _NimHandler.response_body = {"affinities": {"B": {"affinity_pred_value": [-5.0]}}}
        scores = _score_batch_internal_nim("tgt", ["CCO"], "MKTAY", None)
        assert not math.isnan(scores[0])


class TestNimScoringContext:
    def test_nim_context_hash_differs_from_cli(self, monkeypatch):
        # CLI context
        monkeypatch.setattr(mod, "_BOLTZ_BACKEND", "cli")
        monkeypatch.setattr(mod, "_boltz_version", lambda: "2.0")
        monkeypatch.setattr(mod, "_docker_digest", lambda: "d1")
        monkeypatch.setenv("BOLTZ_ACCELERATOR", "cpu")
        h_cli = _scoring_context_hash()

        # NIM context
        monkeypatch.setattr(mod, "_CONTEXT_HASH", None)
        monkeypatch.setattr(mod, "_BOLTZ_BACKEND", "nim")
        monkeypatch.setattr(mod, "_BOLTZ_NIM_URLS", ["http://localhost:8000"])
        h_nim = _scoring_context_hash()

        assert h_cli != h_nim

    def test_nim_context_same_across_urls(self, monkeypatch):
        """NIM context hash is independent of URLs (round-robin is implementation detail)."""
        monkeypatch.setattr(mod, "_BOLTZ_BACKEND", "nim")

        monkeypatch.setattr(mod, "_BOLTZ_NIM_URLS", ["http://host1:8000"])
        h1 = _scoring_context_hash()

        monkeypatch.setattr(mod, "_CONTEXT_HASH", None)
        monkeypatch.setattr(mod, "_BOLTZ_NIM_URLS", ["http://host2:8000"])
        h2 = _scoring_context_hash()

        assert h1 == h2

    def test_nim_scoring_context_info(self, monkeypatch):
        monkeypatch.setattr(mod, "_BOLTZ_BACKEND", "nim")
        monkeypatch.setattr(mod, "_BOLTZ_NIM_URLS", ["http://localhost:8000"])
        monkeypatch.setattr(mod, "_CONTEXT_HASH", None)
        info = get_scoring_context_info()
        assert info["backend"] == "nim"
        assert info["nim_urls"] == ["http://localhost:8000"]
        assert "cache_context_hash" in info


class TestNimScoreBatchE2E:
    """Integration: score_batch with NIM backend uses cache correctly."""

    def test_score_batch_nim_with_cache(self, tmp_path, monkeypatch, nim_server):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        score_cache_dir = cache_dir / "boltz_scores"
        score_cache_dir.mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        monkeypatch.setattr(mod, "CACHE_DIR", cache_dir)
        monkeypatch.setattr(mod, "SCORE_CACHE_DIR", score_cache_dir)
        monkeypatch.setattr(mod, "DATA_DIR", data_dir)
        monkeypatch.setattr(mod, "MSA_CACHE_DIR", tmp_path / "msa")

        (data_dir / "tgt.fasta").write_text(">tgt\nMKTAY\n")

        _NimHandler.response_body = {
            "affinities": {"B": {"affinity_pred_value": [-6.0, -6.2]}},
        }

        scores = mod.score_batch("tgt", ["CCO", "CCCO"])
        assert len(scores) == 2
        assert scores[0] == pytest.approx(-6.1)
        assert scores[1] == pytest.approx(-6.1)

        # Second call should hit cache (even if server is down)
        nim_server.shutdown()
        monkeypatch.setattr(mod, "_BOLTZ_NIM_URL", "http://127.0.0.1:1")
        monkeypatch.setattr(mod, "_BOLTZ_NIM_URLS", ["http://127.0.0.1:1"])
        scores2 = mod.score_batch("tgt", ["CCO", "CCCO"])
        assert scores2[0] == pytest.approx(-6.1)
        assert scores2[1] == pytest.approx(-6.1)


class TestNimScoreSingle:
    """Test score_single() dispatches to NIM correctly."""

    def test_score_single_nim(self, tmp_path, monkeypatch, nim_server):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "tgt.fasta").write_text(">tgt\nMKTAY\n")
        monkeypatch.setattr(mod, "DATA_DIR", data_dir)

        _NimHandler.response_body = {
            "affinities": {"B": {"affinity_pred_value": [-3.5, -4.0]}},
        }
        score = mod.score_single("tgt", "CCO")
        assert score == pytest.approx(-3.75)

    def test_score_single_nim_invalid_smiles(self, tmp_path, monkeypatch, nim_server):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "tgt.fasta").write_text(">tgt\nMKTAY\n")
        monkeypatch.setattr(mod, "DATA_DIR", data_dir)

        score = mod.score_single("tgt", "not_a_smiles!!!")
        assert math.isnan(score)


class TestNimRetry:
    """Test that _nim_predict_affinity retries on transient 5xx errors."""

    def test_retries_on_500_then_succeeds(self, nim_server):
        attempt_counter = {"n": 0}

        # Save original as a plain function reference
        _orig = _NimHandler.__dict__["do_POST"]

        def flaky_post(self_handler):
            attempt_counter["n"] += 1
            if attempt_counter["n"] <= 1:
                # First call: 500
                self_handler.send_response(500)
                self_handler.send_header("Content-Type", "application/json")
                self_handler.end_headers()
                self_handler.wfile.write(b'{"error": "temporary"}')
            else:
                # Subsequent: success
                _NimHandler.response_body = {
                    "affinities": {"B": {"affinity_pred_value": [-5.0]}},
                }
                _orig(self_handler)

        _NimHandler.do_POST = flaky_post
        score = _nim_predict_affinity("MKTAY", "CCO")
        assert score == pytest.approx(-5.0)
        assert attempt_counter["n"] >= 2
        _NimHandler.do_POST = _orig

    def test_no_retry_on_400(self, nim_server):
        attempt_counter = {"n": 0}
        _orig = _NimHandler.__dict__["do_POST"]

        def always_400(self_handler):
            attempt_counter["n"] += 1
            self_handler.send_response(400)
            self_handler.send_header("Content-Type", "application/json")
            self_handler.end_headers()
            self_handler.wfile.write(b'{"error": "bad request"}')

        _NimHandler.do_POST = always_400
        score = _nim_predict_affinity("MKTAY", "CCO")
        assert math.isnan(score)
        assert attempt_counter["n"] == 1  # no retries on 4xx
        _NimHandler.do_POST = _orig
