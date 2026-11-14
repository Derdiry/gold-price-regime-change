"""
Rolling-origin (expanding-window) folds for Phase 4.1.

The evaluation period is the part of the v2 validation + test period where a
complete window can exist: rows [s, n), where s is the first row at or after
the v2 train/val boundary such that the rows s − L … s are all valid (on
Sudan this is the end of the 404-day gap plus L rows). It is cut into
N_BLOCKS = 6 consecutive blocks of equal row count, B0 … B5. Fold k (1 … 5):

    train = rows [0, start(B_{k-1}))      expanding
    val   = B_{k-1}                       early stopping, ensemble weights
    test  = B_k

Rows after B_k are not used by fold k. The windows are built exactly as in
the main protocol: context may come from the previous split, targets never
cross a split boundary, and windows touching an invalid row are dropped.
"""

import numpy as np
import pandas as pd

N_BLOCKS = 6
N_FOLDS = N_BLOCKS - 1


def evaluation_start(valid: np.ndarray, boundary: int, input_length: int) -> int:
    """First row r >= boundary with rows r - L … r all valid."""
    v = np.asarray(valid, bool)
    for r in range(max(boundary, input_length), len(v)):
        if v[r - input_length:r + 1].all():
            return r
    raise ValueError("no complete window after the train/val boundary")


def block_edges(n: int, start: int) -> np.ndarray:
    """N_BLOCKS + 1 row edges splitting [start, n) into equal consecutive blocks."""
    return np.linspace(start, n, N_BLOCKS + 1).round().astype(int)


def fold_frames(df: pd.DataFrame, valid: np.ndarray, train_ratio: float, input_length: int, fold: int):
    """(train, val, test) frames for fold 1 … N_FOLDS."""
    if not 1 <= fold <= N_FOLDS:
        raise ValueError(f"fold must be 1..{N_FOLDS}")
    e = block_edges(len(df), evaluation_start(valid, int(len(df) * train_ratio), input_length))
    return df.iloc[:e[fold - 1]], df.iloc[e[fold - 1]:e[fold]], df.iloc[e[fold]:e[fold + 1]]
