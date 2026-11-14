"""Rolling-origin folds: ordering, disjoint test blocks, expanding train, gap handling."""

import numpy as np
import pandas as pd

from src.data.folds import N_FOLDS, block_edges, evaluation_start, fold_frames


def _df(n=1000):
    return pd.DataFrame({"x": np.arange(n)}, index=pd.date_range("2020-01-01", periods=n))


def test_folds_are_ordered_disjoint_and_expanding():
    df, valid = _df(), np.ones(1000, bool)
    tests, prev_train = [], 0
    for k in range(1, N_FOLDS + 1):
        tr, va, te = fold_frames(df, valid, 0.7, 60, k)
        assert tr.index.max() < va.index.min() and va.index.max() < te.index.min()
        assert len(tr) > prev_train
        prev_train = len(tr)
        tests.append(set(te.index))
        if k > 1:
            assert va.index.equals(fold_frames(df, valid, 0.7, 60, k - 1)[2].index)   # val_k = test_{k-1}
    assert all(not (a & b) for i, a in enumerate(tests) for b in tests[i + 1:])
    assert fold_frames(df, valid, 0.7, 60, N_FOLDS)[2].index.max() == df.index.max()


def test_evaluation_start_skips_gap_plus_context():
    valid = np.ones(1000, bool)
    valid[650:800] = False
    s = evaluation_start(valid, 700, 60)
    assert s == 860 and valid[s - 60:s + 1].all()
    e = block_edges(1000, s)
    assert e[0] == 860 and e[-1] == 1000 and len(e) == N_FOLDS + 2
