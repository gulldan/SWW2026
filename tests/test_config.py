"""Tests for config module."""

import yaml

from dualdocking.config import load_experiment, load_metrics, load_pairs, load_yaml


def test_load_yaml(tmp_path):
    data = {"key": "value", "nested": {"a": 1}}
    (tmp_path / "test.yaml").write_text(yaml.dump(data))
    result = load_yaml(tmp_path / "test.yaml")
    assert result == data


def test_load_pairs(tmp_path):
    cfg = {"pairs": {"pair1": {"target_a": "A"}}}
    (tmp_path / "pairs.yaml").write_text(yaml.dump(cfg))
    result = load_pairs(tmp_path)
    assert "pairs" in result
    assert "pair1" in result["pairs"]


def test_load_metrics(tmp_path):
    cfg = {"thresholds": {"pair1": {"target_a": -7.0}}}
    (tmp_path / "metrics.yaml").write_text(yaml.dump(cfg))
    result = load_metrics(tmp_path)
    assert "thresholds" in result


def test_load_experiment(tmp_path):
    cfg = {"experiment": {"name": "test"}}
    (tmp_path / "experiment.yaml").write_text(yaml.dump(cfg))
    result = load_experiment(tmp_path)
    assert "experiment" in result
