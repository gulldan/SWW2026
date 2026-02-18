"""Tests for config module."""

import pytest
import yaml

from dualdocking.config import (
    collect_all_target_ids,
    collect_blocked_target_ids,
    collect_target_pdb_map,
    load_experiment,
    load_metrics,
    load_pairs,
    load_yaml,
)


def test_load_yaml(tmp_path):
    data = {"key": "value", "nested": {"a": 1}}
    (tmp_path / "test.yaml").write_text(yaml.dump(data))
    result = load_yaml(tmp_path / "test.yaml")
    assert result == data


def test_load_pairs(tmp_path):
    cfg = {
        "pairs": {
            "pair1": {
                "target_a": {"id": "CDK2", "organism": "Homo sapiens"},
                "target_b": {"id": "ERK2", "organism": "Homo sapiens"},
            }
        }
    }
    (tmp_path / "pairs.yaml").write_text(yaml.dump(cfg))
    result = load_pairs(tmp_path)
    assert "pairs" in result
    assert "pair1" in result["pairs"]


def test_load_metrics(tmp_path):
    cfg = {"thresholds": {"pair1": {"target_a": -7.0}}}
    (tmp_path / "metrics.yaml").write_text(yaml.dump(cfg))
    result = load_metrics(tmp_path)
    assert "thresholds" in result


def test_load_pairs_missing_organism(tmp_path):
    cfg = {
        "pairs": {
            "pair1": {
                "target_a": {"id": "CDK2", "organism": "Homo sapiens"},
                "target_b": {"id": "ERK2"},
            }
        }
    }
    (tmp_path / "pairs.yaml").write_text(yaml.dump(cfg))
    with pytest.raises(ValueError, match="missing 'organism'"):
        load_pairs(tmp_path)


def test_load_pairs_non_human_blocked(tmp_path, monkeypatch):
    monkeypatch.delenv("DUALDOCKING_ALLOW_NON_HUMAN", raising=False)
    cfg = {
        "pairs": {
            "pair1": {
                "target_a": {"id": "COX1", "organism": "Mus musculus"},
                "target_b": {"id": "COX2", "organism": "Homo sapiens"},
            }
        }
    }
    (tmp_path / "pairs.yaml").write_text(yaml.dump(cfg))
    with pytest.raises(ValueError, match="non-Homo sapiens"):
        load_pairs(tmp_path)


def test_load_pairs_non_human_bypass(tmp_path, monkeypatch):
    monkeypatch.setenv("DUALDOCKING_ALLOW_NON_HUMAN", "true")
    cfg = {
        "pairs": {
            "pair1": {
                "target_a": {"id": "COX1", "organism": "Mus musculus"},
                "target_b": {"id": "COX2", "organism": "Homo sapiens"},
            }
        }
    }
    (tmp_path / "pairs.yaml").write_text(yaml.dump(cfg))
    result = load_pairs(tmp_path)
    assert "pair1" in result["pairs"]


def test_load_pairs_human_case_insensitive(tmp_path, monkeypatch):
    monkeypatch.delenv("DUALDOCKING_ALLOW_NON_HUMAN", raising=False)
    cfg = {
        "pairs": {
            "pair1": {
                "target_a": {"id": "CDK2", "organism": "homo SAPIENS"},
                "target_b": {"id": "ERK2", "organism": "  Homo Sapiens  "},
            }
        }
    }
    (tmp_path / "pairs.yaml").write_text(yaml.dump(cfg))
    result = load_pairs(tmp_path)
    assert "pair1" in result["pairs"]


def test_load_pairs_blocked_set_skips_organism_check(tmp_path, monkeypatch):
    """Blocked pairs (set: blocked) skip organism validation."""
    monkeypatch.delenv("DUALDOCKING_ALLOW_NON_HUMAN", raising=False)
    cfg = {
        "pairs": {
            "human_pair": {
                "set": "core",
                "target_a": {"id": "CDK2", "organism": "Homo sapiens"},
                "target_b": {"id": "ERK2", "organism": "Homo sapiens"},
            },
            "blocked_pair": {
                "set": "blocked",
                "target_a": {"id": "COX1", "organism": "Ovis aries"},
                "target_b": {"id": "COX2", "organism": "Mus musculus"},
            },
        }
    }
    (tmp_path / "pairs.yaml").write_text(yaml.dump(cfg))
    result = load_pairs(tmp_path)
    assert "human_pair" in result["pairs"]
    assert "blocked_pair" in result["pairs"]


def test_load_pairs_non_blocked_non_human_still_rejected(tmp_path, monkeypatch):
    """Non-blocked non-human pairs are still rejected."""
    monkeypatch.delenv("DUALDOCKING_ALLOW_NON_HUMAN", raising=False)
    cfg = {
        "pairs": {
            "bad_pair": {
                "set": "auxiliary",
                "target_a": {"id": "COX1", "organism": "Ovis aries"},
                "target_b": {"id": "COX2", "organism": "Homo sapiens"},
            },
        }
    }
    (tmp_path / "pairs.yaml").write_text(yaml.dump(cfg))
    with pytest.raises(ValueError, match="non-Homo sapiens"):
        load_pairs(tmp_path)


def test_load_pairs_production_yaml(monkeypatch):
    """Production pairs.yaml loads without error (blocked pairs skipped)."""
    monkeypatch.delenv("DUALDOCKING_ALLOW_NON_HUMAN", raising=False)
    result = load_pairs("configs")
    active = [k for k, v in result["pairs"].items() if v.get("set") != "blocked"]
    blocked = [k for k, v in result["pairs"].items() if v.get("set") == "blocked"]
    assert len(active) == 12
    assert len(blocked) == 1
    assert "COX1_COX2" in blocked


def test_load_experiment(tmp_path):
    cfg = {"experiment": {"name": "test"}}
    (tmp_path / "experiment.yaml").write_text(yaml.dump(cfg))
    result = load_experiment(tmp_path)
    assert "experiment" in result


# ---------------------------------------------------------------------------
# Config helpers (collect_*)
# ---------------------------------------------------------------------------

_PAIRS_FIXTURE = {
    "CDK2_ERK2": {
        "set": "core",
        "target_a": {"id": "CDK2", "pdb_id": "1HCK", "organism": "Homo sapiens"},
        "target_b": {"id": "ERK2", "pdb_id": "4GT3", "organism": "Homo sapiens"},
    },
    "COX1_COX2": {
        "set": "blocked",
        "target_a": {"id": "COX1", "pdb_id": "3KK6", "organism": "Ovis aries"},
        "target_b": {"id": "COX2", "pdb_id": "5KIR", "organism": "Mus musculus"},
    },
}


def test_collect_blocked_target_ids():
    blocked = collect_blocked_target_ids(_PAIRS_FIXTURE)
    assert blocked == {"COX1", "COX2"}


def test_collect_blocked_empty_when_no_blocked():
    pairs = {"CDK2_ERK2": _PAIRS_FIXTURE["CDK2_ERK2"]}
    assert collect_blocked_target_ids(pairs) == set()


def test_collect_all_target_ids():
    ids = collect_all_target_ids(_PAIRS_FIXTURE)
    assert ids == ["CDK2", "COX1", "COX2", "ERK2"]


def test_collect_all_target_ids_deduplicates():
    pairs = {
        "p1": {
            "target_a": {"id": "CDK2", "pdb_id": "1HCK"},
            "target_b": {"id": "CDK2", "pdb_id": "1HCK"},
        },
    }
    assert collect_all_target_ids(pairs) == ["CDK2"]


def test_collect_target_pdb_map():
    mapping = collect_target_pdb_map(_PAIRS_FIXTURE)
    assert mapping == {"CDK2": "1HCK", "ERK2": "4GT3", "COX1": "3KK6", "COX2": "5KIR"}
