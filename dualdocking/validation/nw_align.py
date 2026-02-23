"""Needleman-Wunsch global alignment with BLOSUM62 substitution matrix.

Provides a pure-Python implementation of affine-gap Needleman-Wunsch
alignment suitable for pairwise protein sequence identity computation.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AlignmentResult:
    """Result of Needleman-Wunsch alignment."""

    identity_pct: float = 0.0
    aligned_length: int = 0
    matches: int = 0
    gaps: int = 0
    seq_a_len: int = 0
    seq_b_len: int = 0


# ---------------------------------------------------------------------------
# BLOSUM62 substitution matrix (20 standard amino acids)
# ---------------------------------------------------------------------------

_AA = "ARNDCQEGHILKMFPSTWYV"

# Flat upper-triangle values from NCBI BLOSUM62
_BLOSUM62_FLAT = [
    4,
    -1,
    -2,
    -2,
    0,
    -1,
    -1,
    0,
    -2,
    -1,
    -1,
    -1,
    -1,
    -2,
    -1,
    1,
    0,
    -3,
    -2,
    0,
    5,
    0,
    -2,
    -3,
    1,
    0,
    -2,
    0,
    -3,
    -2,
    2,
    -1,
    -3,
    -2,
    -1,
    -1,
    -3,
    -2,
    -3,
    6,
    1,
    -3,
    0,
    0,
    0,
    1,
    -3,
    -3,
    0,
    -2,
    -3,
    -2,
    1,
    0,
    -4,
    -2,
    -3,
    6,
    -3,
    0,
    2,
    -1,
    -1,
    -3,
    -4,
    -1,
    -3,
    -3,
    -1,
    0,
    -1,
    -4,
    -3,
    -3,
    9,
    -3,
    -4,
    -3,
    -3,
    -1,
    -1,
    -3,
    -1,
    -2,
    -3,
    -1,
    -1,
    -2,
    -2,
    -1,
    5,
    2,
    -2,
    0,
    -3,
    -2,
    1,
    0,
    -3,
    -1,
    0,
    -1,
    -2,
    -1,
    -2,
    5,
    -2,
    0,
    -3,
    -3,
    1,
    -2,
    -3,
    -1,
    0,
    -1,
    -3,
    -2,
    -2,
    6,
    -2,
    -4,
    -4,
    -2,
    -3,
    -3,
    -2,
    0,
    -2,
    -2,
    -3,
    -3,
    8,
    -3,
    -3,
    -1,
    -2,
    -1,
    -2,
    -1,
    -2,
    -2,
    2,
    -3,
    4,
    2,
    -3,
    1,
    0,
    -3,
    -2,
    -1,
    -3,
    -1,
    3,
    4,
    -2,
    2,
    0,
    -3,
    -2,
    -1,
    -2,
    -1,
    1,
    5,
    -1,
    -3,
    -1,
    0,
    -1,
    -3,
    -2,
    -2,
    5,
    0,
    -2,
    -1,
    -1,
    -1,
    -1,
    1,
    6,
    -4,
    -2,
    -2,
    1,
    3,
    -1,
    7,
    -1,
    -1,
    -4,
    -3,
    -2,
    4,
    1,
    -3,
    -2,
    -2,
    5,
    -2,
    -2,
    0,
    11,
    2,
    -3,
    7,
    -1,
    4,
]

BLOSUM62: dict[tuple[str, str], int] = {}


def _init_blosum62() -> None:
    idx = 0
    for i, a in enumerate(_AA):
        for j in range(i, len(_AA)):
            b = _AA[j]
            score = _BLOSUM62_FLAT[idx]
            BLOSUM62[(a, b)] = score
            BLOSUM62[(b, a)] = score
            idx += 1


_init_blosum62()

# ---------------------------------------------------------------------------
# Needleman-Wunsch global alignment
# ---------------------------------------------------------------------------

_GAP_OPEN = -10
_GAP_EXTEND = -0.5


def nw_align(seq_a: str, seq_b: str) -> AlignmentResult:
    """Needleman-Wunsch global alignment with affine gap penalties.

    Uses BLOSUM62 substitution matrix with gap open = -10, gap extend = -0.5
    (EMBOSS defaults).

    Returns:
        An :class:`AlignmentResult` with identity statistics.
    """
    m, n = len(seq_a), len(seq_b)
    if m == 0 or n == 0:
        return AlignmentResult()

    neg_inf = float("-inf")

    # dp_m[i][j] = best score ending with match/mismatch at (i,j)
    # dp_x[i][j] = best score ending with gap in seq_b (deletion from seq_a)
    # dp_y[i][j] = best score ending with gap in seq_a (insertion from seq_b)
    dp_m = [[neg_inf] * (n + 1) for _ in range(m + 1)]
    dp_x = [[neg_inf] * (n + 1) for _ in range(m + 1)]
    dp_y = [[neg_inf] * (n + 1) for _ in range(m + 1)]

    dp_m[0][0] = 0.0
    for i in range(1, m + 1):
        dp_x[i][0] = _GAP_OPEN + _GAP_EXTEND * i
    for j in range(1, n + 1):
        dp_y[0][j] = _GAP_OPEN + _GAP_EXTEND * j

    for i in range(1, m + 1):
        aa_a = seq_a[i - 1]
        for j in range(1, n + 1):
            aa_b = seq_b[j - 1]
            sub = BLOSUM62.get((aa_a, aa_b), -4)

            dp_m[i][j] = sub + max(dp_m[i - 1][j - 1], dp_x[i - 1][j - 1], dp_y[i - 1][j - 1])
            dp_x[i][j] = max(
                dp_m[i - 1][j] + _GAP_OPEN + _GAP_EXTEND,
                dp_x[i - 1][j] + _GAP_EXTEND,
            )
            dp_y[i][j] = max(
                dp_m[i][j - 1] + _GAP_OPEN + _GAP_EXTEND,
                dp_y[i][j - 1] + _GAP_EXTEND,
            )

    # Traceback
    best = max(dp_m[m][n], dp_x[m][n], dp_y[m][n])
    if best == dp_m[m][n]:
        state = 0
    elif best == dp_x[m][n]:
        state = 1
    else:
        state = 2

    align_a: list[str] = []
    align_b: list[str] = []
    i, j = m, n

    while i > 0 or j > 0:
        if state == 0:  # Match/mismatch
            if i == 0 or j == 0:
                break
            align_a.append(seq_a[i - 1])
            align_b.append(seq_b[j - 1])
            prev_best = max(dp_m[i - 1][j - 1], dp_x[i - 1][j - 1], dp_y[i - 1][j - 1])
            if dp_m[i - 1][j - 1] == prev_best:
                state = 0
            elif dp_x[i - 1][j - 1] == prev_best:
                state = 1
            else:
                state = 2
            i -= 1
            j -= 1
        elif state == 1:  # Gap in seq_b (deletion)
            align_a.append(seq_a[i - 1])
            align_b.append("-")
            state = 1 if dp_x[i - 1][j] + _GAP_EXTEND == dp_x[i][j] else 0
            i -= 1
        else:  # Gap in seq_a (insertion)
            align_a.append("-")
            align_b.append(seq_b[j - 1])
            state = 2 if dp_y[i][j - 1] + _GAP_EXTEND == dp_y[i][j] else 0
            j -= 1

    align_a.reverse()
    align_b.reverse()

    matches = sum(1 for a, b in zip(align_a, align_b) if a == b and a != "-")
    gaps = sum(1 for a, b in zip(align_a, align_b) if a == "-" or b == "-")
    aligned_len = len(align_a)

    identity = (matches / aligned_len * 100) if aligned_len > 0 else 0.0

    return AlignmentResult(
        identity_pct=round(identity, 1),
        aligned_length=aligned_len,
        matches=matches,
        gaps=gaps,
        seq_a_len=m,
        seq_b_len=n,
    )
