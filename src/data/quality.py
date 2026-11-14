"""
Documented data-quality steps for the Sudan raw file (data/raw/new_Data.csv).

The raw file is never modified. Every correction happens here, at load time,
and is returned as a report so the number of affected rows can be logged and
saved alongside the processed data.

Issues handled:
  1. Decimal-shift outliers in the international OHLC columns
     (e.g. Open = 117,472 on 2016-12-03, where High/Low are 1,182.6/1,165.1;
     the value is restored to 1,174.72).
  2. Several quotes on the same date (635 dates carry 2-3 rows with different
     Price / FX values). One row per date is kept by an explicit rule.
  3. Missing calendar days. The series is reindexed to daily frequency and
     forward-filled; every filled day is flagged (`is_ffilled`), and days
     inside an unusually long gap (`in_long_gap`, e.g. the 404-day gap
     2019-11-25 -> 2021-01-02) are flagged so that no training or evaluation
     window overlaps them.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# International gold OHLC columns (USD/oz) in the Sudan raw file
OHLC_COLS = ["open", "high", "low"]


def fix_decimal_shift_outliers(
    df: pd.DataFrame,
    ref_col: str = "intl_price_usd",
    cols: tuple = tuple(OHLC_COLS),
    factor: float = 100.0,
    tol: float = 0.10,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Restore values that are a power-of-ten shift of the true value.

    A value v in ``cols`` is treated as decimal-shifted when it is more than
    10x the same row's reference price, and v / factor lies within ``tol``
    (10%) of that reference. It is replaced by v / factor. Anything else that
    looks wrong is left alone and only logged, so the rule cannot silently
    "fix" a genuine move.

    Returns (corrected copy, list of {date, column, old, new}).
    """
    df = df.copy()
    fixes = []
    ref = df[ref_col]
    for col in cols:
        if col not in df.columns:
            continue
        ratio = df[col] / ref
        shifted = (ratio > 10) & ((df[col] / factor / ref - 1).abs() <= tol)
        for idx in df.index[shifted]:
            old = float(df.at[idx, col])
            new = old / factor
            df.at[idx, col] = new
            fixes.append({"date": str(pd.Timestamp(df.at[idx, "date"]).date()) if "date" in df.columns else str(idx),
                          "column": col, "old": old, "new": new})
        unexplained = (ratio > 10) & ~shifted
        if unexplained.any():
            logger.warning(f"  {col}: {int(unexplained.sum())} rows >10x {ref_col} not matching a decimal shift (left unchanged)")
    for f in fixes:
        logger.info(f"  Decimal-shift fix: {f['date']} {f['column']} {f['old']:g} -> {f['new']:g}")
    return df, fixes


def collapse_duplicate_dates(df: pd.DataFrame, rule: str = "first") -> tuple[pd.DataFrame, dict]:
    """
    Keep one row per date.

    rule:
      "first"  — first row of the day in file order (the v1 behaviour)
      "last"   — last row of the day in file order
      "median" — column-wise median of the day's rows

    Expects a ``date`` column and rows in file order. Returns (frame indexed
    by date, report with the number of affected dates and dropped rows).
    """
    n_rows = len(df)
    dup_dates = int(df["date"].duplicated(keep=False).groupby(df["date"]).any().sum())
    if rule in ("first", "last"):
        out = df.drop_duplicates(subset=["date"], keep=rule)
    elif rule == "median":
        out = df.groupby("date", as_index=False).median(numeric_only=True)
    else:
        raise ValueError(f"Unknown duplicate-date rule: {rule}")
    out = out.sort_values("date").set_index("date")
    report = {
        "rule": rule,
        "file_rows": n_rows,
        "unique_dates": len(out),
        "dates_with_multiple_rows": dup_dates,
        "rows_dropped": n_rows - len(out),
    }
    logger.info(
        f"  Duplicate dates: {dup_dates} dates have >1 row; rule='{rule}' keeps "
        f"{len(out)} of {n_rows} rows"
    )
    return out, report


def find_long_gaps(dates: pd.DatetimeIndex, max_gap_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Return (last_observed, next_observed) pairs whose spacing exceeds ``max_gap_days``.
    """
    dates = pd.DatetimeIndex(sorted(dates))
    gaps = []
    for prev, nxt in zip(dates[:-1], dates[1:]):
        if (nxt - prev).days > max_gap_days:
            gaps.append((prev, nxt))
    return gaps


def add_fill_flags(
    daily: pd.DataFrame,
    observed_dates: pd.DatetimeIndex,
    max_gap_days: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Flag forward-filled days in a daily-reindexed frame.

    is_ffilled  = 1 on calendar days with no observation in the raw file.
    in_long_gap = 1 on days strictly between the two observations bounding a
                  gap longer than ``max_gap_days``. Windows touching these days
                  are dropped (see processors.build_sequences).

    Neither column is a model input.
    """
    daily = daily.copy()
    observed = daily.index.isin(observed_dates)
    daily["is_ffilled"] = (~observed).astype(np.int8)
    daily["in_long_gap"] = np.int8(0)
    gaps = find_long_gaps(observed_dates, max_gap_days)
    for prev, nxt in gaps:
        inside = (daily.index > prev) & (daily.index < nxt)
        daily.loc[inside, "in_long_gap"] = 1
    report = {
        "daily_rows": len(daily),
        "observed_days": int(observed.sum()),
        "ffilled_days": int((~observed).sum()),
        "max_gap_days": max_gap_days,
        "long_gaps": [
            {"last_observed": str(p.date()), "next_observed": str(n.date()),
             "days": (n - p).days, "flagged_days": int(((daily.index > p) & (daily.index < n)).sum())}
            for p, n in gaps
        ],
    }
    logger.info(
        f"  Fill flags: {report['ffilled_days']} of {len(daily)} daily rows forward-filled; "
        f"long gaps: {[(g['last_observed'], g['next_observed'], g['days']) for g in report['long_gaps']]}"
    )
    return daily, report
