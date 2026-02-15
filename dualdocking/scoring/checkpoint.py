"""Score checkpoint persistence for resumable scoring runs.

Provides load/save of per-target score arrays so that scoring can be
interrupted and resumed without re-computing expensive predictions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def load_checkpoint_scores(
    scores_path: Path,
    expected_smiles: list[str],
) -> np.ndarray | None:
    """Load cached scores when they match the expected molecule list.

    Validates that the checkpoint file exists, contains ``score`` and
    ``smiles`` columns, has the same length, and the SMILES match.  As a
    fallback, reorders scores by SMILES when both lists are unique and
    equal as sets.

    Args:
        scores_path: Path to the score file (with extension; stem lookup
            is performed internally via :func:`~dualdocking.data.io_utils.stem_exists`).
        expected_smiles: Ordered SMILES list that the checkpoint must match.

    Returns:
        Score array aligned to *expected_smiles*, or ``None`` when the
        checkpoint is missing, invalid, or incompatible.
    """
    from dualdocking.data.io_utils import read_df, stem_exists

    stem = scores_path.with_suffix("")
    if not stem_exists(stem):
        return None

    saved = read_df(stem)

    if "score" not in saved.columns:
        logger.warning("Checkpoint missing 'score' column ({}). Re-scoring.", scores_path)
        return None

    if len(saved) != len(expected_smiles):
        logger.warning(
            "Checkpoint size mismatch ({}): saved={}, expected={}. Re-scoring.",
            scores_path,
            len(saved),
            len(expected_smiles),
        )
        return None

    if "smiles" not in saved.columns:
        logger.warning("Checkpoint missing 'smiles' column ({}). Re-scoring.", scores_path)
        return None

    saved_smiles = saved["smiles"].astype(str).tolist()
    if saved_smiles == [str(s) for s in expected_smiles]:
        return saved["score"].to_numpy(dtype=float)

    # Fallback: reorder by SMILES when both lists are unique and equal as sets.
    if (
        len(set(saved_smiles)) == len(saved_smiles)
        and len(set(expected_smiles)) == len(expected_smiles)
        and set(saved_smiles) == set(expected_smiles)
    ):
        score_map = dict(zip(saved_smiles, saved["score"].astype(float).tolist(), strict=False))
        return np.asarray([score_map[str(s)] for s in expected_smiles], dtype=float)

    logger.warning("Checkpoint SMILES mismatch ({}). Re-scoring.", scores_path)
    return None


def save_checkpoint_scores(
    scores_path: Path,
    smiles: list[str],
    scores: np.ndarray,
) -> None:
    """Persist a score checkpoint to disk (Parquet + CSV).

    Args:
        scores_path: Target path (extension is replaced internally).
        smiles: SMILES strings corresponding to *scores*.
        scores: Score array to persist.
    """
    from dualdocking.data.io_utils import write_df

    write_df(
        pd.DataFrame({"smiles": smiles, "score": scores}),
        scores_path.with_suffix(""),
        csv_too=True,
    )
