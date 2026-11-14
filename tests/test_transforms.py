"""
Phase 3 transforms: converting forecasts back to price units uses only
information available at the forecast origin, never a target-window value.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.processors import RAW_TARGET_COL, VALID_COL, build_sequences
from src.data.transforms import WindowScaler, inverse_target, pretransform_inputs, target_context

FEATURES = ["price_sdg", "fx_rate_log", "macd", "rsi"]   # level, log_level, price_rel, free
L, H = 10, 3


def _frame(n=80, seed=0):
    rng = np.random.default_rng(seed)
    price = 1000 * np.exp(np.cumsum(rng.normal(0.01, 0.02, n)))
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    df = pd.DataFrame({"price_sdg": price, "fx_rate_log": np.log(price / 30), "macd": rng.normal(0, 5, n),
                       "rsi": rng.uniform(20, 80, n)}, index=idx)
    df[RAW_TARGET_COL] = df["price_sdg"]
    df[VALID_COL] = True
    return df


def _windows(df):
    X, y, meta = build_sequences(df, FEATURES, "price_sdg", L, H, return_meta=True)
    return X.astype(np.float64), meta


@pytest.mark.parametrize("cond", ["log_return", "revin"])
def test_round_trip_to_price(cond):
    X, m = _windows(_frame())
    ws = WindowScaler(cond, FEATURES).fit(X, m["y_raw"], m["ctx_target"], m["last_obs"])
    y_model = ws.transform_y(m["y_raw"], m["ctx_target"], m["last_obs"])
    back = ws.to_price(y_model, m["ctx_target"], m["last_obs"])
    assert np.allclose(back, m["y_raw"], rtol=1e-5)


@pytest.mark.parametrize("cond", ["log_return", "revin"])
def test_inverse_ignores_target_window(cond):
    """Overwriting every target value leaves the price mapping of a fixed forecast unchanged."""
    df = _frame()
    X, m = _windows(df)
    ws = WindowScaler(cond, FEATURES).fit(X, m["y_raw"], m["ctx_target"], m["last_obs"])
    y_hat = np.random.default_rng(1).uniform(0, 1, size=m["y_raw"].shape)
    ref = ws.to_price(y_hat, m["ctx_target"], m["last_obs"])

    i = 20                                    # window 20: context rows 20..29, targets 30..32
    pert = df.copy()
    pert.iloc[30:33, pert.columns.get_loc(RAW_TARGET_COL)] = 1e12
    pert.iloc[30:33, pert.columns.get_loc("price_sdg")] = 1e12
    Xp, mp = _windows(pert)
    assert (mp["y_raw"][i] == 1e12).all()     # the targets really changed ...
    got = ws.to_price(y_hat, mp["ctx_target"], mp["last_obs"])
    assert np.allclose(got[i], ref[i])        # ... the mapping for window i did not
    # and the window's model inputs did not change either
    assert np.allclose(ws.transform_X(Xp, mp["last_obs"])[i], ws.transform_X(X, m["last_obs"])[i])


def test_log_return_inverse_uses_only_last_context_value():
    X, m = _windows(_frame())
    assert np.allclose(m["last_obs"], m["ctx_target"][:, -1])
    tctx = target_context("log_return", m["ctx_target"], m["last_obs"])
    assert set(tctx) == {"P"} and np.allclose(tctx["P"], m["last_obs"])
    zero_return = inverse_target("log_return", np.zeros((len(X), H)), tctx)
    assert np.allclose(zero_return, m["last_obs"][:, None])      # ŷ = 0 is persistence


def test_revin_inverse_uses_only_context_statistics():
    X, m = _windows(_frame())
    tctx = target_context("revin", m["ctx_target"], m["last_obs"])
    assert set(tctx) == {"mu", "sd"}
    assert np.allclose(tctx["mu"], m["ctx_target"].mean(axis=1))


def test_inputs_transform_by_group():
    X, m = _windows(_frame())
    lr = pretransform_inputs("log_return", X, FEATURES, m["last_obs"])
    assert np.allclose(lr[:, -1, 0], 0)                     # level: log ratio to own last value
    assert np.allclose(lr[:, -1, 1], 0)                     # log level: difference to own last value
    assert np.allclose(lr[:, :, 2], X[:, :, 2] / m["last_obs"][:, None])   # price-relative
    assert np.array_equal(lr[:, :, 3], X[:, :, 3])          # scale-free: unchanged
    rv = pretransform_inputs("revin", X, FEATURES, m["last_obs"])
    assert np.allclose(rv[:, :, :3].mean(axis=1), 0, atol=1e-9)
    assert np.array_equal(rv[:, :, 3], X[:, :, 3])
