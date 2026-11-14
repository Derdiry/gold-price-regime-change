"""Window construction: long-gap rule and no use of target values at forecast time."""

import numpy as np
import pandas as pd

from src.data.processors import RAW_TARGET_COL, VALID_COL, build_sequences, row_validity
from src.data.quality import add_fill_flags


def _frame(n=40):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    price = np.arange(n, dtype=float) * 10 + 1000
    return pd.DataFrame({"price": price, "f": price / 10, RAW_TARGET_COL: price}, index=idx)


def test_windows_touching_long_gap_are_dropped():
    df = _frame()
    observed = df.index.delete(range(15, 25))          # 10 missing days
    flagged, report = add_fill_flags(df, observed, max_gap_days=5)
    assert report["long_gaps"][0]["flagged_days"] == 10
    flagged[VALID_COL] = row_validity(flagged, ["f"])
    L, h = 5, 2
    X, y, meta = build_sequences(flagged, ["f"], "price", L, h, return_meta=True)
    gap = flagged.index[flagged["in_long_gap"] == 1]
    for origin in pd.to_datetime(meta["origin_date"]):
        span = pd.date_range(origin - pd.Timedelta(days=L - 1), origin + pd.Timedelta(days=h))
        assert not span.isin(gap).any()
    # every possible window minus those overlapping the 10 flagged days
    assert len(X) == (40 - L - h + 1) - (10 + L + h - 1)


def test_last_obs_is_context_end_and_never_a_target_value():
    df = _frame()
    df[VALID_COL] = True
    L, h = 5, 3
    X, y, meta = build_sequences(df, ["f"], "price", L, h, return_meta=True)
    for i in range(len(X)):
        # last_obs equals the final context row ...
        assert meta["last_obs"][i] == X[i, -1, 0] * 10
        # ... and is strictly earlier than every target value (prices increase)
        assert meta["last_obs"][i] < y[i].min()
        assert pd.Timestamp(meta["target_first_date"][i]) == pd.Timestamp(meta["origin_date"][i]) + pd.Timedelta(days=1)


def test_last_obs_unchanged_when_targets_are_perturbed():
    """Proves no target-window value is used: changing future targets leaves last_obs intact."""
    df = _frame()
    df[VALID_COL] = True
    L, h = 5, 3
    _, _, base = build_sequences(df, ["f"], "price", L, h, return_meta=True)
    i = 10                                     # window 10: context rows 10..14, targets 15..17
    pert = df.copy()
    pert.iloc[15:18, pert.columns.get_loc(RAW_TARGET_COL)] = -1e9
    pert.iloc[15:18, pert.columns.get_loc("price")] = -1e9
    _, y2, m2 = build_sequences(pert, ["f"], "price", L, h, return_meta=True)
    assert m2["last_obs"][i] == base["last_obs"][i]
    assert (y2[i] == -1e9).all()
