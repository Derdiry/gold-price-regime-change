"""
Feature engineering for both Global and Sudan tracks.

Global features:
  - Technical indicators computed on gold_close (SMA, EMA, RSI, MACD, Bollinger, ATR)
  - Price return lags and rolling volatility

Sudan-specific features:
  - FX premium (deviation of the official SDG/USD rate from its rolling mean)
  - Local-international price spread
  - Political regime dummies (Bashir / Transitional / Post-coup)
  - ACLED conflict intensity lags
  - Regional price differentials (Sudan vs international)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Technical Indicators — Global & Sudan
# ─────────────────────────────────────────────────────────────────────────────

def add_sma(df: pd.DataFrame, price_col: str, windows: list) -> pd.DataFrame:
    """Add simple moving averages sma_{w} for each window w."""
    for w in windows:
        df[f"sma_{w}"] = df[price_col].rolling(window=w, min_periods=1).mean()
    return df


def add_ema(df: pd.DataFrame, price_col: str, spans: list) -> pd.DataFrame:
    """Add exponential moving averages ema_{s} for each span s."""
    for s in spans:
        df[f"ema_{s}"] = df[price_col].ewm(span=s, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, price_col: str, window: int = 14) -> pd.DataFrame:
    """Add the Relative Strength Index (rolling-mean gains/losses) as `rsi`."""
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    price_col: str,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger(
    df: pd.DataFrame,
    price_col: str,
    window: int = 20,
    n_std: float = 2.0,
) -> pd.DataFrame:
    rolling = df[price_col].rolling(window=window, min_periods=1)
    mid = rolling.mean()
    std = rolling.std(ddof=0).fillna(0)
    df["bb_upper"] = mid + n_std * std
    df["bb_lower"] = mid - n_std * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (mid + 1e-8)
    df["bb_pct"] = (df[price_col] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-8)
    return df


def add_atr(
    df: pd.DataFrame,
    high_col: str = "gold_high",
    low_col: str = "gold_low",
    price_col: str = "gold_close",
    window: int = 14,
) -> pd.DataFrame:
    if high_col not in df.columns or low_col not in df.columns:
        df["atr"] = 0.0
        return df
    high = df[high_col]
    low  = df[low_col]
    close_prev = df[price_col].shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low  - close_prev).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(com=window - 1, min_periods=window).mean()
    return df


def add_return_lags(
    df: pd.DataFrame,
    price_col: str,
    lags: list = (1, 5, 20),
) -> pd.DataFrame:
    for lag in lags:
        df[f"return_{lag}d"] = df[price_col].pct_change(lag)
    return df


def add_rolling_volatility(
    df: pd.DataFrame,
    price_col: str,
    window: int = 20,
) -> pd.DataFrame:
    returns = df[price_col].pct_change()
    df["rolling_vol"] = returns.rolling(window=window, min_periods=1).std()
    return df


def add_all_technical(
    df: pd.DataFrame,
    price_col: str,
    cfg: dict,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply all technical indicators to a DataFrame.
    Uses hyperparameters from model_config.yaml features.technical section.
    """
    feat_cfg = cfg["features"]["technical"]
    lag_cfg  = cfg["features"]["lags"]

    df = add_sma(df, price_col, feat_cfg["sma_windows"])
    df = add_ema(df, price_col, feat_cfg["ema_windows"])
    df = add_rsi(df, price_col, feat_cfg["rsi_window"])
    df = add_macd(df, price_col, feat_cfg["macd_fast"], feat_cfg["macd_slow"], feat_cfg["macd_signal"])
    df = add_bollinger(df, price_col, feat_cfg["bollinger_window"], feat_cfg["bollinger_std"])

    # ATR only when high/low columns are available
    _high = high_col or (price_col.replace("close", "high") if "close" in price_col else None)
    _low  = low_col  or (price_col.replace("close", "low")  if "close" in price_col else None)
    if _high and _low and _high in df.columns and _low in df.columns:
        df = add_atr(df, high_col=_high, low_col=_low, price_col=price_col, window=feat_cfg["atr_window"])
    else:
        df["atr"] = 0.0

    df = add_return_lags(df, price_col, lag_cfg["return_lags"])
    df = add_rolling_volatility(df, price_col, lag_cfg["rolling_vol_window"])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Sudan-Specific Features
# ─────────────────────────────────────────────────────────────────────────────

def add_fx_premium(
    df: pd.DataFrame,
    fx_col: str = "fx_rate",
    window: int = 90,
) -> pd.DataFrame:
    """
    FX Premium: how much the exchange rate has deviated from its rolling average.

    High positive value → the official rate is above its recent average.
    """
    if fx_col not in df.columns:
        logger.warning(f"fx_premium skipped: column '{fx_col}' not found")
        df["fx_premium"]    = 0.0
        df["fx_rate_log"]   = 0.0
        df["fx_rate_change"] = 0.0
        return df

    rolling_mean = df[fx_col].rolling(window=window, min_periods=1).mean()
    df["fx_premium"]    = (df[fx_col] - rolling_mean) / (rolling_mean + 1e-8)
    df["fx_rate_log"]   = np.log(df[fx_col].clip(lower=1e-8))
    df["fx_rate_change"] = df[fx_col].pct_change().fillna(0)
    return df


def add_local_intl_spread(
    df: pd.DataFrame,
    local_price_col: str = "price_sdg",
    intl_price_col: str = "intl_price_usd",
    fx_col: str = "fx_rate",
    oz_to_gram: float = 31.1035,
) -> pd.DataFrame:
    """
    Local-international spread: an international reference price expressed in SDG
    (column parity_price) and the relative difference of the local price from it.
    """
    if not all(c in df.columns for c in [local_price_col, intl_price_col, fx_col]):
        logger.warning("Local-intl spread skipped: missing columns")
        df["parity_price"]    = 0.0
        df["local_intl_spread"] = 0.0
        return df
    parity = (df[intl_price_col] / oz_to_gram) * df[fx_col]
    df["parity_price"]    = parity
    df["local_intl_spread"] = (df[local_price_col] - parity) / (parity + 1e-8)
    return df


def add_regime_dummies(
    df: pd.DataFrame,
    breakpoints: Optional[list] = None,
) -> pd.DataFrame:
    """
    Add political regime dummy variables.

    Default breakpoints (from config):
      2019-04-11: Bashir ousted → transitional government
      2021-10-25: Military coup → post-coup period

    Dummies: regime_1 (transitional), regime_2 (post-coup)
    regime_0 (Bashir era) is the reference — not included.
    """
    if breakpoints is None:
        breakpoints = ["2019-04-11", "2021-10-25"]

    breakpoints = sorted(pd.to_datetime(breakpoints))

    df["regime"] = 0
    for i, bp in enumerate(breakpoints):
        df.loc[df.index >= bp, "regime"] = i + 1

    for i in range(1, len(breakpoints) + 1):
        df[f"regime_{i}"] = (df["regime"] == i).astype(int)

    df = df.drop(columns=["regime"])
    return df


def add_conflict_lags(
    df: pd.DataFrame,
    conflict_cols: list,
    lag_windows: list,
) -> pd.DataFrame:
    """
    Add lagged conflict intensity features.

    v1 code path. ACLED data is monthly; lags are in months (approximated as 30-day
    shifts since the merged DataFrame is daily). It runs only if ACLED columns are present,
    which was never the case in v1 or in the v2 main runs (docs/implementation_notes.md D5).
    The Phase 4.3 ACLED experiment uses src/data/acled.py instead (completed calendar
    months, no look-ahead). No result supports a particular conflict lag.
    """
    for col in conflict_cols:
        if col not in df.columns:
            continue
        for lag in lag_windows:
            shift_days = lag * 30
            df[f"{col}_lag{lag}m"] = df[col].shift(shift_days)
    return df


def add_regional_differential(
    df: pd.DataFrame,
    local_price_col: str = "price_sdg",
    intl_price_col: str = "intl_price_usd",
    fx_col: str = "fx_rate",
    regional_col: Optional[str] = "egypt_gold_proxy",
    oz_to_gram: float = 31.1035,
) -> pd.DataFrame:
    """
    Regional price ratios: the local price in USD, its ratio to the international
    reference price and, if a regional series is present (it is off in all reported
    runs), the ratio to that series.
    """
    if fx_col in df.columns and local_price_col in df.columns:
        df["price_usd_gram"] = df[local_price_col] / (df[fx_col] + 1e-8)

    if intl_price_col in df.columns:
        intl_per_gram = df[intl_price_col] / oz_to_gram
        if "price_usd_gram" in df.columns:
            df["sudan_intl_ratio"] = df["price_usd_gram"] / (intl_per_gram + 1e-8)

    if regional_col and regional_col in df.columns and "price_usd_gram" in df.columns:
        df["sudan_egypt_ratio"] = df["price_usd_gram"] / (df[regional_col] + 1e-8)

    return df


def add_all_sudan_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply all Sudan-specific feature engineering using config."""
    feat_cfg = cfg["features"]["sudan_specific"]

    df = add_fx_premium(df, fx_col=feat_cfg["fx_premium_col"])
    df = add_local_intl_spread(df)
    df = add_regime_dummies(df, breakpoints=feat_cfg["regime_breakpoints"])
    df = add_conflict_lags(
        df,
        conflict_cols=["conflict_score", "event_count", "fatalities"],
        lag_windows=feat_cfg["conflict_lag_windows"],
    )
    df = add_regional_differential(df)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stationarity & Cointegration Analysis (EDA / research layer)
# ─────────────────────────────────────────────────────────────────────────────

def adf_test(series: pd.Series, name: str = "") -> dict:
    """
    Augmented Dickey-Fuller stationarity test.

    p < 0.05 → stationary (reject unit root hypothesis)
    """
    result = adfuller(series.dropna(), autolag="AIC")
    is_stationary = result[1] < 0.05
    output = {
        "series": name,
        "adf_statistic": round(result[0], 4),
        "p_value": round(result[1], 4),
        "lags_used": result[2],
        "n_obs": result[3],
        "is_stationary": is_stationary,
        "conclusion": "stationary" if is_stationary else "non-stationary (unit root)",
    }
    logger.info(f"ADF [{name}]: p={result[1]:.4f} → {output['conclusion']}")
    return output


def cointegration_test(
    series1: pd.Series,
    series2: pd.Series,
    name1: str = "series1",
    name2: str = "series2",
) -> dict:
    """
    Engle-Granger cointegration test between two non-stationary series.

    p < 0.05 → cointegrated (long-run equilibrium relationship exists)
    """
    aligned = pd.concat([series1, series2], axis=1).dropna()
    score, p_value, crit_values = coint(aligned.iloc[:, 0], aligned.iloc[:, 1])

    is_cointegrated = p_value < 0.05
    output = {
        "series1": name1,
        "series2": name2,
        "test_statistic": round(score, 4),
        "p_value": round(p_value, 4),
        "critical_values": {
            "1%": round(crit_values[0], 4),
            "5%": round(crit_values[1], 4),
            "10%": round(crit_values[2], 4),
        },
        "is_cointegrated": is_cointegrated,
        "conclusion": (
            "cointegrated → long-run equilibrium exists"
            if is_cointegrated
            else "not cointegrated → model levels or use differencing"
        ),
        "recommendation": (
            "Use Error Correction Model or model spread as target"
            if is_cointegrated
            else "Model price changes (log returns) rather than levels"
        ),
    }
    logger.info(
        f"Cointegration [{name1} vs {name2}]: p={p_value:.4f} → {output['conclusion']}"
    )
    return output


def run_sudan_stationarity_report(df: pd.DataFrame) -> list:
    """
    Run ADF tests on all key Sudan series.
    Returns list of result dicts — used in EDA notebook.
    """
    series_to_test = {
        "price_sdg": "Sudan Price (SDG/gram)",
        "intl_price_usd": "Intl Price (USD/oz)",
        "fx_rate": "SDG/USD Rate",
        "local_intl_spread": "Local-Intl Spread",
        "fx_premium": "FX Premium",
    }
    results = []
    for col, name in series_to_test.items():
        if col in df.columns:
            results.append(adf_test(df[col], name))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Feature column lists (used by processors and trainers)
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_FEATURE_COLS = [
    # Price-derived
    "gold_close", "gold_open", "gold_high", "gold_low", "gold_volume",
    # Other tickers
    "gold_etf_close", "usd_index_close", "brent_crude_close",
    # Macro
    "treasury_10y", "cpi", "m2",
    # Technical
    "sma_20", "sma_50", "sma_200",
    "ema_12", "ema_26",
    "rsi",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "bb_pct",
    "atr",
    # Lag
    "return_1d", "return_5d", "return_20d",
    "rolling_vol",
]

SUDAN_FEATURE_COLS = [
    # Raw
    "price_sdg", "intl_price_usd", "fx_rate", "open", "high", "low", "volume",
    # Sudan-specific
    "fx_premium", "fx_rate_log", "fx_rate_change",
    "parity_price", "local_intl_spread",
    "regime_1", "regime_2",
    "conflict_score_lag1m", "conflict_score_lag2m", "conflict_score_lag3m",
    "price_usd_gram", "sudan_intl_ratio",
    # Technical on local price
    "sma_20", "sma_50", "sma_200",
    "ema_12", "ema_26",
    "rsi",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "bb_pct",
    "return_1d", "return_5d", "return_20d",
    "rolling_vol",
]


# Phase 4.5 transfer: the 17 technical indicators both tracks compute from their own
# price, in one fixed order, after the target price itself (18 inputs on each track).
SHARED_TECHNICAL_COLS = [
    "sma_20", "sma_50", "sma_200",
    "ema_12", "ema_26",
    "rsi",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "bb_pct",
    "return_1d", "return_5d", "return_20d",
    "rolling_vol",
]
assert all(c in GLOBAL_FEATURE_COLS and c in SUDAN_FEATURE_COLS for c in SHARED_TECHNICAL_COLS)


def shared_feature_cols(target_col: str) -> list:
    """Transfer inputs: the track's target price followed by SHARED_TECHNICAL_COLS."""
    return [target_col, *SHARED_TECHNICAL_COLS]


def get_available_feature_cols(df: pd.DataFrame, feature_col_list: list) -> list:
    """
    Filter feature_col_list to only columns that exist in df.
    Logs any missing columns for debugging.
    """
    available = [c for c in feature_col_list if c in df.columns]
    missing   = [c for c in feature_col_list if c not in df.columns]
    if missing:
        logger.debug(f"Feature columns not in DataFrame: {missing}")
    return available
