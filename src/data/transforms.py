"""
Target/input transformations for the Phase 3 scale experiment.

Problem: the Sudan test prices are 6.4–9.9× the training maximum (and 513 of
917 global test days are above it), while v1 scaled every input and the
target with a MinMax scaler fitted on the training rows. The models then see
inputs and must emit targets far outside anything seen in training.

A *condition* is a per-window pre-transform applied before the same MinMax
scaling as v1, so the only thing that changes between conditions is the
pre-transform:

  minmax      identity (the v1 setup; scalers fitted on the training rows)
  log_return  target  y'_k = log(y_{t+k} / P_t)        inverse  P_t · exp(ŷ'_k)
              inputs  price levels      -> log(x_τ / x_t)
                      log levels        -> x_τ − x_t
                      signed price-unit -> x_τ / P_t
  revin       per-window instance normalisation (RevIN without the learnable
              affine, so it is identical for every architecture)
              target  y'_k = (y_{t+k} − μ) / σ           inverse  ŷ'_k · σ + μ
              inputs  every scale-dependent feature -> (x_τ − μ_x) / σ_x

where t is the forecast origin (last context row), P_t the last observed
target, x_t a feature's last context value, and μ, σ the mean and standard
deviation over the L context rows (σ floored at 1e-3·|μ| to avoid division by
a flat window). Scale-free features (RSI, returns, dummies, spreads, rates,
volume) are left as they are in every condition.

Information at forecast time only: every statistic above comes from the
context window. ``inverse_target`` needs only (P_t) or (μ, σ of the target's
context), never a target value; tests/test_transforms.py proves this by
perturbing the target window.

After the pre-transform, features and target are MinMax-scaled with scalers
fitted on the training windows only.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

CONDITIONS = ("minmax", "log_return", "revin")

# Positive price-like levels: transformed by log ratio to the last context value
POSITIVE_LEVELS = {
    # global
    "gold_close", "gold_open", "gold_high", "gold_low", "gold_etf_close",
    "usd_index_close", "brent_crude_close", "cpi", "m2",
    # sudan
    "price_sdg", "intl_price_usd", "fx_rate", "open", "high", "low",
    "parity_price", "price_usd_gram",
    # both (technical indicators on the target price)
    "sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "bb_upper",
}
# Features that are already logs of a level: difference to the last context value
LOG_LEVELS = {"fx_rate_log"}
# Price-unit quantities that can be <= 0, or are spreads in price units:
# divided by the last observed target price
PRICE_RELATIVE = {"macd", "macd_signal", "macd_hist", "bb_lower", "atr"}

SIGMA_FLOOR = 1e-3


def feature_group(name: str) -> str:
    """'level', 'log_level', 'price_rel' or 'free' (left unchanged)."""
    if name in POSITIVE_LEVELS:
        return "level"
    if name in LOG_LEVELS:
        return "log_level"
    if name in PRICE_RELATIVE:
        return "price_rel"
    return "free"


def _ctx_stats(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and floored std over the context axis (axis 1)."""
    mu = a.mean(axis=1)
    sd = a.std(axis=1)
    return mu, np.maximum(sd, np.maximum(SIGMA_FLOOR * np.abs(mu), 1e-8))


def pretransform_inputs(condition: str, X: np.ndarray, feature_cols: list[str],
                        last_target: np.ndarray) -> np.ndarray:
    """
    Per-window pre-transform of raw input windows X (n, L, F). Uses only the
    context window itself and P_t (= last_target, the final context value).
    """
    X = np.asarray(X, np.float64)
    if condition == "minmax":
        return X
    out = X.copy()
    P = np.asarray(last_target, np.float64)[:, None]
    for j, name in enumerate(feature_cols):
        g = feature_group(name)
        x = X[:, :, j]
        if g == "free":
            continue
        if condition == "log_return":
            if g == "level":
                last = x[:, -1:]
                out[:, :, j] = np.log(np.maximum(x, 1e-12) / np.maximum(last, 1e-12))
            elif g == "log_level":
                out[:, :, j] = x - x[:, -1:]
            else:  # price_rel
                out[:, :, j] = x / P
        elif condition == "revin":
            mu, sd = _ctx_stats(x)
            out[:, :, j] = (x - mu[:, None]) / sd[:, None]
        else:
            raise ValueError(f"Unknown condition {condition}")
    return out


def target_context(condition: str, ctx_target: np.ndarray, last_target: np.ndarray) -> dict:
    """Forecast-time statistics needed to map targets to and from price units."""
    if condition == "log_return":
        return {"P": np.asarray(last_target, np.float64)}
    if condition == "revin":
        mu, sd = _ctx_stats(np.asarray(ctx_target, np.float64))
        return {"mu": mu, "sd": sd}
    return {}


def forward_target(condition: str, y: np.ndarray, tctx: dict) -> np.ndarray:
    """Price-unit targets (n, h) -> pre-transformed targets (before MinMax)."""
    y = np.asarray(y, np.float64)
    if condition == "log_return":
        return np.log(y / tctx["P"][:, None])
    if condition == "revin":
        return (y - tctx["mu"][:, None]) / tctx["sd"][:, None]
    return y


def inverse_target(condition: str, y_pre: np.ndarray, tctx: dict) -> np.ndarray:
    """Pre-transformed forecasts (n, h) -> price units, using forecast-time statistics only."""
    y_pre = np.asarray(y_pre, np.float64)
    if condition == "log_return":
        return tctx["P"][:, None] * np.exp(y_pre)
    if condition == "revin":
        return y_pre * tctx["sd"][:, None] + tctx["mu"][:, None]
    return y_pre


class WindowScaler:
    """
    Pre-transform + MinMax for one (track, horizon, condition), fitted on the
    training windows only. Used for log_return and revin; the minmax
    condition keeps the v1 row-level scalers (src/data/dataset.py).
    """

    def __init__(self, condition: str, feature_cols: list[str]):
        if condition not in ("log_return", "revin"):
            raise ValueError(condition)
        self.condition = condition
        self.feature_cols = feature_cols
        self.f_scaler = MinMaxScaler()
        self.t_scaler = MinMaxScaler()

    def fit(self, X, y, ctx_target, last_target) -> "WindowScaler":
        Xp = pretransform_inputs(self.condition, X, self.feature_cols, last_target)
        self.f_scaler.fit(Xp.reshape(-1, Xp.shape[2]))
        yp = forward_target(self.condition, y, target_context(self.condition, ctx_target, last_target))
        self.t_scaler.fit(yp.reshape(-1, 1))
        return self

    def transform_X(self, X, last_target) -> np.ndarray:
        Xp = pretransform_inputs(self.condition, X, self.feature_cols, last_target)
        return self.f_scaler.transform(Xp.reshape(-1, Xp.shape[2])).reshape(Xp.shape).astype(np.float32)

    def transform_y(self, y, ctx_target, last_target) -> np.ndarray:
        yp = forward_target(self.condition, y, target_context(self.condition, ctx_target, last_target))
        return self.t_scaler.transform(yp.reshape(-1, 1)).reshape(yp.shape).astype(np.float32)

    def to_price(self, y_model, ctx_target, last_target) -> np.ndarray:
        """Model outputs (n, h) -> price units. Needs only context statistics."""
        y_model = np.asarray(y_model, np.float64)
        yp = self.t_scaler.inverse_transform(y_model.reshape(-1, 1)).reshape(y_model.shape)
        return inverse_target(self.condition, yp, target_context(self.condition, ctx_target, last_target))
