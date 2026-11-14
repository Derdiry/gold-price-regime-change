"""
Data processors: cleaning, merging, resampling, splitting, and scaling.

Handles both Global and Sudan tracks. Key responsibilities:
  - Fill missing values (forward-fill for prices, interpolate for macro)
  - Resample irregular Sudan data to daily frequency
  - Merge all feature sources into a single aligned DataFrame
  - Chronological train/val/test split (no shuffling)
  - Fit and save MinMaxScaler / StandardScaler per track
  - Build sliding-window sequences (X, y) ready for model input
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning for OHLCV price DataFrames.
      - Drop rows where close/price is NaN
      - Forward-fill remaining NaNs (gaps from non-trading days)
      - Remove duplicate indices
      - Sort by date ascending
    """
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    price_col = next((c for c in ["close", "price_sdg", "Close"] if c in df.columns), None)
    if price_col:
        df = df.dropna(subset=[price_col])

    df = df.ffill()
    return df


def clean_macro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean macro/economic DataFrames (FRED, World Bank).
      - Interpolate linearly (macro data is monthly/annual, prices are daily)
      - Forward-fill any remaining NaN at edges
    """
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.interpolate(method="time")
    df = df.ffill().bfill()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Resampling
# ─────────────────────────────────────────────────────────────────────────────

def resample_to_daily(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Reindex a DataFrame to a continuous daily date range.
    Fills gaps using forward-fill (default) or linear interpolation.

    Used for Sudan data which has irregular ~2-3 day spacing.
    """
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_index)
    if method == "ffill":
        df = df.ffill()
    elif method == "interpolate":
        df = df.interpolate(method="time")
    df.index.name = "date"
    return df


def align_to_trading_days(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """
    Align df to the trading day index of a reference DataFrame.
    Used to align macro/conflict data to price trading days.
    """
    df = df.reindex(reference.index, method="ffill")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Merging
# ─────────────────────────────────────────────────────────────────────────────

def merge_global_track(
    prices: dict,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all global track sources into a single DataFrame.

    prices: dict of {name: OHLCV DataFrame} from fetch_global_prices()
    macro:  FRED macro DataFrame

    Returns a clean daily DataFrame with all features aligned to gold_futures dates.
    """
    gold = prices.get("gold_futures")
    if gold is None:
        gold = prices.get("gold_etf")
    if gold is None:
        raise ValueError("No gold price data found — both gold_futures and gold_etf failed")

    gold = clean_ohlcv(gold)
    merged = gold.copy()
    merged.columns = [f"gold_{c}" for c in merged.columns]

    for name, df in prices.items():
        if name == "gold_futures":
            continue
        df = clean_ohlcv(df)
        df = align_to_trading_days(df, gold)
        # Keep only close price of auxiliary tickers
        close_col = next((c for c in ["close", "Close"] if c in df.columns), df.columns[0])
        merged[f"{name}_close"] = df[close_col]

    if not macro.empty:
        macro = clean_macro(macro)
        macro = align_to_trading_days(macro, gold)
        merged = pd.concat([merged, macro], axis=1)

    merged = merged.dropna(how="all")
    logger.info(f"Global track merged: {merged.shape}, {merged.index.min()} to {merged.index.max()}")
    return merged


def merge_sudan_track(
    raw: pd.DataFrame,
    world_bank: pd.DataFrame,
    acled: pd.DataFrame,
    regional: pd.DataFrame,
    max_gap_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Merge all Sudan track sources into a single daily DataFrame.

    raw:        Sudan raw CSV (already cleaned and renamed)
    world_bank: World Bank annual macro, will be resampled
    acled:      monthly conflict data
    regional:   regional gold prices

    Returns daily DataFrame aligned to Sudan raw data dates.

    If ``max_gap_days`` is given, adds the flag columns ``is_ffilled`` and
    ``in_long_gap`` (src/data/quality.add_fill_flags); they are not model inputs.
    The report is stored in ``df.attrs["fill_report"]``.
    """
    df = clean_ohlcv(raw)
    observed_dates = df.index
    df = resample_to_daily(df)
    fill_report = None
    if max_gap_days is not None:
        from src.data.quality import add_fill_flags
        df, fill_report = add_fill_flags(df, observed_dates, max_gap_days)

    if not world_bank.empty:
        wb = clean_macro(world_bank)
        wb = resample_to_daily(wb)
        wb = align_to_trading_days(wb, df)
        df = pd.concat([df, wb], axis=1)

    if not acled.empty:
        acled_daily = acled.reindex(df.index, method="ffill")
        df = pd.concat([df, acled_daily], axis=1)

    if not regional.empty:
        reg = clean_macro(regional)
        reg = align_to_trading_days(reg, df)
        df = pd.concat([df, reg], axis=1)

    df = df.ffill().dropna(subset=["price_sdg"])
    if fill_report is not None:
        df.attrs["fill_report"] = fill_report
    logger.info(f"Sudan track merged: {df.shape}, {df.index.min()} to {df.index.max()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Train / Val / Test Split
# ─────────────────────────────────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-indexed DataFrame chronologically.
    No shuffling. test_ratio = 1 - train_ratio - val_ratio.
    """
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]

    logger.info(
        f"Split: train={len(train)} ({train.index.min().date()}–{train.index.max().date()}), "
        f"val={len(val)} ({val.index.min().date()}–{val.index.max().date()}), "
        f"test={len(test)} ({test.index.min().date()}–{test.index.max().date()})"
    )
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Scaling
# ─────────────────────────────────────────────────────────────────────────────

def fit_scalers(
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    scaler_type: str = "minmax",
    save_dir: Optional[str] = None,
    track: str = "global",
) -> Tuple[object, object]:
    """
    Fit feature scaler and target scaler on training data only.
    Returns (feature_scaler, target_scaler).

    Saves scalers to disk if save_dir is provided.
    """
    ScalerClass = MinMaxScaler if scaler_type == "minmax" else StandardScaler

    feature_scaler = ScalerClass()
    feature_scaler.fit(train[feature_cols].values)

    target_scaler = ScalerClass()
    target_scaler.fit(train[[target_col]].values)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / f"{track}_feature_scaler.pkl", "wb") as f:
            pickle.dump(feature_scaler, f)
        with open(Path(save_dir) / f"{track}_target_scaler.pkl", "wb") as f:
            pickle.dump(target_scaler, f)
        logger.info(f"Saved scalers to {save_dir}/{track}_*.pkl")

    return feature_scaler, target_scaler


def load_scalers(
    track: str,
    scaler_dir: str = "data/processed/scalers",
) -> Tuple[object, object]:
    """Load previously saved scalers from disk."""
    with open(Path(scaler_dir) / f"{track}_feature_scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)
    with open(Path(scaler_dir) / f"{track}_target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)
    return feature_scaler, target_scaler


RAW_TARGET_COL = "__target_raw"   # unscaled copy of the target, kept by apply_scaling
VALID_COL = "__valid"             # row may appear in a window (see row_validity)


def apply_scaling(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    feature_scaler: object,
    target_scaler: object,
) -> pd.DataFrame:
    """
    Apply pre-fitted scalers to a DataFrame. Returns scaled copy.

    The unscaled target is kept in ``RAW_TARGET_COL`` so that sequence building
    can record each window's last observed value in original price units.
    """
    scaled = df.copy()
    scaled[RAW_TARGET_COL] = df[target_col].values
    scaled[feature_cols] = feature_scaler.transform(df[feature_cols].values)
    scaled[target_col] = target_scaler.transform(df[[target_col]].values)
    return scaled


def row_validity(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """
    True for rows that may appear in a context or target window.

    A row is invalid if any feature is undefined (indicator warm-up at the start
    of the series) or if it lies inside a long forward-filled gap
    (``in_long_gap`` == 1, see src/data/quality.py). Windows that touch an
    invalid row are dropped rather than filled.
    """
    valid = df[feature_cols].notna().all(axis=1)
    if "in_long_gap" in df.columns:
        valid &= df["in_long_gap"] == 0
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# Sliding Window Sequence Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    input_length: int = 60,
    horizon: int = 1,
    return_meta: bool = False,
):
    """
    Build sliding-window sequences for time series models.

    Returns:
        X: shape (n_samples, input_length, n_features)
        y: shape (n_samples, horizon)
        meta (only if return_meta): dict with
            origin_date: date of the last context row (the forecast origin), shape (n,)
            target_first_date / target_last_date: dates of the first and last target, shape (n,)
            y_raw:       unscaled targets, shape (n, horizon)
            ctx_target:  unscaled target over the context window, shape (n, input_length)
            last_obs:    unscaled target at the forecast origin, shape (n,)
                         (requires RAW_TARGET_COL, added by apply_scaling)

    No data leakage: y[i] uses only data after X[i]'s window ends, and
    last_obs[i] is the final context value, never a target value.

    If ``VALID_COL`` is present, a window is kept only when every row of its
    context and target span is valid (see row_validity); otherwise all windows
    are kept, as in v1. Remaining NaN values are forward-filled then
    back-filled before sequence construction; with VALID_COL present no
    window contains a filled value.
    """
    # Fill NaN: ffill first (most recent known value), then bfill for leading NaNs
    df = df.copy()
    df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0)
    df[target_col] = df[[target_col]].ffill().bfill().fillna(0)[target_col]

    feature_data = df[feature_cols].values
    target_data = df[target_col].values
    valid = df[VALID_COL].values.astype(bool) if VALID_COL in df.columns else np.ones(len(df), bool)
    raw_target = df[RAW_TARGET_COL].values if RAW_TARGET_COL in df.columns else None
    # Prefix sums let each window's validity be checked in O(1)
    invalid_cum = np.concatenate([[0], np.cumsum(~valid)])

    X, y, origin_pos = [], [], []
    total_len = len(df)
    n_dropped = 0

    for i in range(input_length, total_len - horizon + 1):
        if invalid_cum[i + horizon] - invalid_cum[i - input_length] > 0:
            n_dropped += 1
            continue
        X.append(feature_data[i - input_length : i])
        y.append(target_data[i : i + horizon])
        origin_pos.append(i - 1)

    n_feat = len(feature_cols)
    X = np.array(X, dtype=np.float32).reshape(-1, input_length, n_feat)
    y = np.array(y, dtype=np.float32).reshape(-1, horizon)
    origin_pos = np.array(origin_pos, dtype=np.int64)

    logger.info(f"Built sequences: X={X.shape}, y={y.shape} (dropped {n_dropped} windows touching invalid rows)")
    if not return_meta:
        return X, y
    meta = {
        "origin_date": df.index.values[origin_pos],
        "target_first_date": df.index.values[origin_pos + 1],
        "target_last_date": df.index.values[origin_pos + horizon],
        "last_obs": (raw_target[origin_pos].astype(np.float64) if raw_target is not None
                     else np.full(len(origin_pos), np.nan)),
        "y_raw": (np.stack([raw_target[origin_pos + 1 + k] for k in range(horizon)], axis=1).astype(np.float64)
                  if raw_target is not None else None),
        "ctx_target": (np.stack([raw_target[origin_pos - input_length + 1 + k] for k in range(input_length)],
                                axis=1).astype(np.float64) if raw_target is not None else None),
        "n_dropped": n_dropped,
    }
    return X, y, meta


def build_all_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    input_length: int = 60,
    horizon: int = 1,
) -> dict:
    """
    Build X/y sequences for all three splits.

    Returns dict with keys: X_train, y_train, X_val, y_val, X_test, y_test.
    Concatenates train+val tail (input_length rows) before val to preserve
    context at the val boundary without leaking val labels into training.
    """
    # Prepend tail of train to val/test for context continuity at boundaries
    train_tail = train.iloc[-input_length:]

    val_ctx = pd.concat([train_tail, val])
    test_ctx = pd.concat([pd.concat([train_tail, val]).iloc[-input_length:], test])

    out = {}
    for split, frame in (("train", train), ("val", val_ctx), ("test", test_ctx)):
        X, y, meta = build_sequences(frame, feature_cols, target_col, input_length, horizon, return_meta=True)
        out[f"X_{split}"] = X
        out[f"y_{split}"] = y
        out[f"origin_{split}"] = meta["origin_date"]
        out[f"tfirst_{split}"] = meta["target_first_date"]
        out[f"tlast_{split}"] = meta["target_last_date"]
        out[f"last_{split}"] = meta["last_obs"]
        out[f"yraw_{split}"] = meta["y_raw"].reshape(-1, horizon)
        out[f"ctxt_{split}"] = meta["ctx_target"].reshape(-1, input_length)
        out[f"dropped_{split}"] = np.int64(meta["n_dropped"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load processed datasets
# ─────────────────────────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame, track: str, save_dir: str = "data/processed") -> None:
    """Write the merged, feature-engineered frame to {save_dir}/{track}_merged.csv."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = Path(save_dir) / f"{track}_merged.csv"
    df.to_csv(path)
    logger.info(f"Saved processed {track} data to {path}")


def load_processed(track: str, load_dir: str = "data/processed") -> pd.DataFrame:
    """Read {load_dir}/{track}_merged.csv with a DatetimeIndex."""
    path = Path(load_dir) / f"{track}_merged.csv"
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


def save_sequences(sequences: dict, track: str, horizon: int, save_dir: str = "data/features") -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = Path(save_dir) / f"{track}_h{horizon}.npz"
    np.savez_compressed(path, **sequences)
    logger.info(f"Saved sequences to {path}")


def load_sequences(track: str, horizon: int, load_dir: str = "data/features") -> dict:
    path = Path(load_dir) / f"{track}_h{horizon}.npz"
    data = np.load(path)
    return {k: data[k] for k in data.files}
