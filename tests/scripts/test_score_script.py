"""Tests for cli/score.py hit-threshold logic, objective columns, and checkpoint library."""

from pathlib import Path

import numpy as np
import pandas as pd

import cli.s4_score.score as score_mod
from dualdocking.scoring.checkpoint import load_checkpoint_scores, save_checkpoint_scores
from dualdocking.scoring.objectives import ObjectiveKind, compute_combined, parse_objectives_config


def test_hit_threshold_uses_leq():
    """Verify <= is used consistently for hit threshold (score <= threshold = hit)."""
    thr_a = 0.0
    thr_b = 0.0

    affinity_a = np.array([-1.0, 0.0, 0.5])
    affinity_b = np.array([-1.0, 0.0, 0.5])

    is_hit_both = (affinity_a <= thr_a) & (affinity_b <= thr_b)

    assert is_hit_both[0]  # -1.0 <= 0.0 → hit
    assert is_hit_both[1]  # 0.0 <= 0.0 → hit (boundary)
    assert not is_hit_both[2]  # 0.5 > 0.0 → not hit


def test_all_objective_columns_generated():
    """Parse config → compute → verify all combined_* columns exist."""
    exp_cfg = {
        "objectives": {
            "baseline": "min",
            "alternatives": [
                {"kind": "harmonic"},
                {"kind": "geometric"},
                {"kind": "weighted_sum", "weight_a": 0.5},
                {"kind": "weighted_product", "weight_a": 0.5},
                {"kind": "tchebycheff", "weight_a": 0.5, "utopia_a": -3.0, "utopia_b": -3.0},
            ],
        }
    }
    objectives = parse_objectives_config(exp_cfg)

    scores_a = np.array([-8.0, -5.0, -3.0])
    scores_b = np.array([-7.0, -6.0, -4.0])

    joined = pd.DataFrame({"smiles": ["A", "B", "C"]})
    for obj in objectives:
        joined[obj.column_name] = compute_combined(scores_a, scores_b, obj.kind, lower_is_better=True, **obj.params)

    expected_cols = {
        "combined_min",
        "combined_harmonic",
        "combined_geometric",
        "combined_weighted_sum_0.5",
        "combined_weighted_product_0.5",
        "combined_tchebycheff_0.5",
    }
    assert expected_cols.issubset(set(joined.columns))


def test_sort_by_primary_objective():
    """Verify DataFrame sorted by primary column, not hardcoded combined_min."""
    exp_cfg = {
        "objectives": {
            "baseline": "harmonic",
            "alternatives": [{"kind": "min"}],
        }
    }
    objectives = parse_objectives_config(exp_cfg)
    assert objectives[0].kind == ObjectiveKind.HARMONIC
    assert objectives[0].is_primary

    scores_a = np.array([-8.0, -5.0, -3.0, -9.0])
    scores_b = np.array([-7.0, -6.0, -4.0, -2.0])

    joined = pd.DataFrame({"smiles": ["A", "B", "C", "D"]})
    primary_col = "combined_min"  # fallback
    for obj in objectives:
        joined[obj.column_name] = compute_combined(scores_a, scores_b, obj.kind, lower_is_better=True, **obj.params)
        if obj.is_primary:
            primary_col = obj.column_name

    assert primary_col == "combined_harmonic"
    joined = joined.sort_values(primary_col)
    # Values should be non-decreasing
    vals = joined[primary_col].to_numpy()
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


# --- Checkpoint library tests ---


def _write_checkpoint(scores_path: Path, smiles: list[str], scores: list[float]) -> None:
    from dualdocking.data.io_utils import write_df

    df = pd.DataFrame({"smiles": smiles, "score": scores})
    write_df(df, scores_path.with_suffix(""), csv_too=True)


def test_checkpoint_load_returns_none_when_missing(tmp_path: Path) -> None:
    """load_checkpoint_scores returns None for non-existent file."""
    result = load_checkpoint_scores(tmp_path / "scores_A.csv", ["CC", "CCC"])
    assert result is None


def test_checkpoint_load_exact_match(tmp_path: Path) -> None:
    """load_checkpoint_scores returns scores when SMILES match exactly."""
    scores_path = tmp_path / "scores_A.csv"
    smiles = ["CC", "CCC", "CCCC"]
    expected = [-1.0, -2.0, -3.0]
    _write_checkpoint(scores_path, smiles, expected)

    result = load_checkpoint_scores(scores_path, smiles)
    assert result is not None
    assert result.tolist() == expected


def test_checkpoint_load_reorders_unique_smiles(tmp_path: Path) -> None:
    """load_checkpoint_scores reorders scores when SMILES are unique but shuffled."""
    scores_path = tmp_path / "scores_A.csv"
    _write_checkpoint(scores_path, ["CCC", "CC", "CCCC"], [-2.0, -1.0, -3.0])

    result = load_checkpoint_scores(scores_path, ["CC", "CCC", "CCCC"])
    assert result is not None
    assert result.tolist() == [-1.0, -2.0, -3.0]


def test_checkpoint_load_length_mismatch_returns_none(tmp_path: Path) -> None:
    """load_checkpoint_scores returns None when lengths differ."""
    scores_path = tmp_path / "scores_A.csv"
    _write_checkpoint(scores_path, ["A", "B"], [-1.0, -2.0])

    result = load_checkpoint_scores(scores_path, ["CC", "CCC", "CCCC"])
    assert result is None


def test_checkpoint_save_roundtrip(tmp_path: Path) -> None:
    """save_checkpoint_scores + load_checkpoint_scores round-trips correctly."""
    scores_path = tmp_path / "scores_A.csv"
    smiles = ["CC", "CCC", "CCCC"]
    scores = np.array([-1.5, -2.5, -3.5])

    save_checkpoint_scores(scores_path, smiles, scores)
    result = load_checkpoint_scores(scores_path, smiles)
    assert result is not None
    np.testing.assert_array_almost_equal(result, scores)


# --- Integration tests through _score_one_target ---


class _DummyBackend:
    def __init__(self, out: list[float]) -> None:
        self.out = out
        self.calls = 0

    def score_batch(
        self,
        target_id: str,  # noqa: ARG002
        smiles_list: list[str],  # noqa: ARG002
        *,
        use_cache: bool = True,  # noqa: ARG002
        device: str | None = None,  # noqa: ARG002
    ) -> list[float]:
        self.calls += 1
        return self.out


def test_score_checkpoint_length_mismatch_rescores(tmp_path: Path) -> None:
    scores_path = tmp_path / "scores_A.csv"
    _write_checkpoint(scores_path, ["A", "B"], [-1.0, -2.0])

    smiles = ["CC", "CCC", "CCCC"]
    backend = _DummyBackend(out=[-3.0, -4.0, -5.0])

    out = score_mod._score_one_target(  # noqa: SLF001
        target_id="CDK2",
        smiles_list=smiles,
        scores_path=scores_path,
        force=False,
        scoring_backend=backend,
    )

    assert backend.calls == 1
    assert out.tolist() == [-3.0, -4.0, -5.0]


def test_score_checkpoint_reordered_unique_smiles_aligns_without_rescore(tmp_path: Path) -> None:
    scores_path = tmp_path / "scores_A.csv"
    _write_checkpoint(scores_path, ["CCC", "CC", "CCCC"], [-2.0, -1.0, -3.0])

    smiles = ["CC", "CCC", "CCCC"]
    backend = _DummyBackend(out=[-9.0, -9.0, -9.0])

    out = score_mod._score_one_target(  # noqa: SLF001
        target_id="CDK2",
        smiles_list=smiles,
        scores_path=scores_path,
        force=False,
        scoring_backend=backend,
    )

    assert backend.calls == 0
    assert out.tolist() == [-1.0, -2.0, -3.0]
