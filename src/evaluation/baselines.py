"""
Baseline forecasters, evaluated on exactly the same windows as the neural models.

Every function forecasts ``max_h`` steps from one forecast origin using only
data up to and including that origin, and returns prices in original units.

  persistence  ŷ_{t+k} = y_t
  drift        ŷ_{t+k} = y_t + k · (y_t − y_{t−L+1}) / (L − 1)
               (random walk with drift; the drift is the mean daily change
               over the L-day context window the neural models see)
  arima        ARIMA(p,1,q) on log price, optional drift, refit at each origin
  ets          exponential smoothing on log price, refit at each origin
  vecm         VECM on log [price_sdg, intl_price_usd, fx_rate] with one
               cointegrating relation (Sudan only). Refit at each origin.

The statistical models are fitted on a *history* ending at the origin: at
most ``max_history`` rows, never extending back across an invalid row (the
long Sudan gap), so no forward-filled gap day is ever used. Log-scale
forecasts are mapped back with exp(), i.e. the median under a log-normal
error. When a fit fails, the origin falls back to persistence and the
fallback is counted.
"""

import warnings

import numpy as np

ARIMA_GRID = [
    {"order": (0, 1, 1), "trend": "n"}, {"order": (0, 1, 1), "trend": "t"},
    {"order": (1, 1, 0), "trend": "n"}, {"order": (1, 1, 0), "trend": "t"},
    {"order": (1, 1, 1), "trend": "n"}, {"order": (1, 1, 1), "trend": "t"},
    {"order": (2, 1, 2), "trend": "n"}, {"order": (2, 1, 2), "trend": "t"},
]
ETS_GRID = [
    {"trend": None, "damped_trend": False},
    {"trend": "add", "damped_trend": False},
    {"trend": "add", "damped_trend": True},
]
VECM_GRID = [
    {"k_ar_diff": k, "deterministic": d} for k in (1, 2, 5) for d in ("ci", "co")
]


def persistence(last_obs: np.ndarray, max_h: int) -> np.ndarray:
    """(n,) last values -> (n, max_h) forecasts."""
    return np.repeat(np.asarray(last_obs, float)[:, None], max_h, axis=1)


def drift(context_target: np.ndarray, max_h: int) -> np.ndarray:
    """(n, L) unscaled context targets -> (n, max_h) random-walk-with-drift forecasts."""
    ctx = np.asarray(context_target, float)
    slope = (ctx[:, -1] - ctx[:, 0]) / (ctx.shape[1] - 1)
    steps = np.arange(1, max_h + 1)
    return ctx[:, -1:] + slope[:, None] * steps[None, :]


def _arima_path(log_hist: np.ndarray, max_h: int, cfg: dict) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA
    res = ARIMA(log_hist, order=cfg["order"], trend=cfg["trend"]).fit()
    return res.forecast(max_h)


def _ets_path(log_hist: np.ndarray, max_h: int, cfg: dict) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    res = ExponentialSmoothing(log_hist, trend=cfg["trend"], damped_trend=cfg["damped_trend"],
                               initialization_method="estimated").fit()
    return res.forecast(max_h)


def _vecm_path(log_hist: np.ndarray, max_h: int, cfg: dict) -> np.ndarray:
    from statsmodels.tsa.vector_ar.vecm import VECM
    res = VECM(log_hist, k_ar_diff=cfg["k_ar_diff"], coint_rank=1, deterministic=cfg["deterministic"]).fit()
    return res.predict(steps=max_h)[:, 0]


FITTERS = {"arima": _arima_path, "ets": _ets_path, "vecm": _vecm_path}


def statistical_paths(family: str, cfg: dict, histories: list[np.ndarray], max_h: int) -> tuple[np.ndarray, int]:
    """
    Forecast every origin with one model configuration.

    histories: per origin, the unscaled history ending at the origin, shape
               (T,) for univariate families or (T, k) for VECM (first column
               is the target). Values are logged here.
    Returns ((n, max_h) price forecasts, number of persistence fallbacks).
    """
    fit = FITTERS[family]
    out = np.empty((len(histories), max_h))
    fallbacks = 0
    for i, hist in enumerate(histories):
        log_hist = np.log(np.asarray(hist, float))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                path = np.exp(np.asarray(fit(log_hist, max_h, cfg), float))
            if not np.all(np.isfinite(path)):
                raise FloatingPointError("non-finite forecast")
            out[i] = path
        except Exception:
            last = hist[-1] if np.ndim(hist) == 1 else hist[-1, 0]
            out[i] = last
            fallbacks += 1
    return out, fallbacks
