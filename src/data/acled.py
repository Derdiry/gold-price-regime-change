"""
ACLED monthly conflict series for Sudan (Phase 4.3).

Sources (data/raw/sudan/, gitignored, never redistributed):
  - weekly:  ACLED "Aggregated data on Africa" (one row per week x Admin1 x
             event type x sub-event type; WEEK is the Saturday that starts the
             week, so a week covers WEEK … WEEK + 6 days)
  - monthly: ACLED "Number of political violence events by
             country-month-year", used only to cross-check the weekly build

Weekly -> monthly rules:
  - "prorata" (primary): a week spanning two months is split between them in
    proportion to its days in each month (events and fatalities alike, so
    monthly values can be fractional);
  - "start" (sensitivity): the whole week goes to the month of its start date.

Political violence (``definition``):
  - "disorder_type" (primary, ACLED's own definition; matches the
    country-month-year file): DISORDER_TYPE contains "Political violence";
  - "event_types" (sensitivity): EVENT_TYPE in POLITICAL_VIOLENCE.

No look-ahead: the lag-k feature on day t is the value of calendar month
month(t) − k, k ≥ 1, so only months completed before t are used.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

POLITICAL_VIOLENCE = ["Battles", "Explosions/Remote violence", "Violence against civilians"]
COUNTRY = "Sudan"
SERIES = ["pv_events", "pv_fatalities", "all_events", "all_fatalities"]
DEFINITIONS = ("disorder_type", "event_types")
ROOT = Path(__file__).resolve().parents[2]
WEEKLY_FILE = ROOT / "data/raw/sudan/Africa_aggregated_data_up_to_week_of-2026-10-17.xlsx"
CACHE_DIR = ROOT / "logs/v2/acled"   # ACLED-derived monthly series: gitignored, never committed


def load_weekly_sudan(path: str | Path) -> pd.DataFrame:
    """Rows of the weekly Africa file whose COUNTRY is exactly 'Sudan'."""
    df = pd.read_excel(path)
    out = df[df["COUNTRY"] == COUNTRY].copy()
    assert not out["COUNTRY"].str.contains("South", case=False).any(), "South Sudan rows remain"
    assert (out["WEEK"].dt.dayofweek == 5).all(), "ACLED weeks are expected to start on Saturday"
    return out


def load_monthly_cmy_sudan(path: str | Path) -> pd.Series:
    """Political-violence events per month for exactly 'Sudan' from the country-month-year file."""
    df = pd.read_excel(path)
    s = df[df["COUNTRY"] == COUNTRY]
    idx = pd.to_datetime(s["MONTH"] + " " + s["YEAR"].astype(str), format="%B %Y").dt.to_period("M")
    out = pd.Series(s["EVENTS"].to_numpy(float), index=idx, name="cmy_pv_events").sort_index()
    assert not out.index.duplicated().any()
    return out


def week_month_shares(week_start: pd.Timestamp, rule: str) -> dict:
    """{month period: share of the week's 7 days} under ``rule``."""
    if rule == "start":
        return {week_start.to_period("M"): 1.0}
    if rule != "prorata":
        raise ValueError(rule)
    days = pd.date_range(week_start, periods=7, freq="D").to_period("M")
    return {m: float((days == m).sum()) / 7 for m in days.unique()}


def is_political_violence(weekly: pd.DataFrame, definition: str) -> pd.Series:
    if definition == "disorder_type":
        return weekly["DISORDER_TYPE"].str.contains("Political violence", na=False)
    if definition == "event_types":
        return weekly["EVENT_TYPE"].isin(POLITICAL_VIOLENCE)
    raise ValueError(definition)


def monthly_series(weekly: pd.DataFrame, rule: str = "prorata", definition: str = "disorder_type") -> pd.DataFrame:
    """Monthly political-violence and all-event counts and fatalities (columns SERIES)."""
    w = weekly.assign(pv=is_political_violence(weekly, definition))
    per_week = pd.DataFrame({
        "pv_events": w["EVENTS"].where(w["pv"], 0), "pv_fatalities": w["FATALITIES"].where(w["pv"], 0),
        "all_events": w["EVENTS"], "all_fatalities": w["FATALITIES"], "WEEK": w["WEEK"],
    }).groupby("WEEK")[SERIES].sum()
    rows = []
    for week, vals in per_week.iterrows():
        for month, share in week_month_shares(week, rule).items():
            rows.append({"month": month, **{c: share * vals[c] for c in SERIES}})
    out = pd.DataFrame(rows).groupby("month")[SERIES].sum().sort_index()
    full = pd.period_range(out.index.min(), out.index.max(), freq="M")
    return out.reindex(full, fill_value=0.0)


def n_weeks_spanning_two_months(weekly: pd.DataFrame) -> int:
    weeks = pd.DatetimeIndex(weekly["WEEK"].unique())
    return int(((weeks + pd.Timedelta(days=6)).to_period("M") != weeks.to_period("M")).sum())


def lagged_features(monthly: pd.DataFrame, dates: pd.DatetimeIndex, lags=(1, 2, 3), cols=None) -> pd.DataFrame:
    """
    Daily features {col}_lag{k}m: for day t, the value of month(t) − k.

    A month absent from ``monthly`` gives NaN (never a later month's value).
    """
    cols = list(cols or monthly.columns)
    months = pd.DatetimeIndex(dates).to_period("M")
    out = {}
    for k in lags:
        if k < 1:
            raise ValueError("lags must be >= 1 month (no current or future month)")
        src = months - k
        for c in cols:
            out[f"{c}_lag{k}m"] = monthly[c].reindex(src).to_numpy()
    return pd.DataFrame(out, index=pd.DatetimeIndex(dates))


def cached_monthly(rule: str = "prorata", definition: str = "disorder_type") -> tuple[pd.DataFrame, str]:
    """Monthly series from CACHE_DIR, built from WEEKLY_FILE on first use. Returns (series, SHA-256 of the cache file)."""
    path = CACHE_DIR / f"sudan_monthly_{definition}_{rule}.csv"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        monthly_series(load_weekly_sudan(WEEKLY_FILE), rule, definition).to_csv(path, index_label="month")
    m = pd.read_csv(path, index_col="month")
    m.index = pd.PeriodIndex(m.index, freq="M")
    return m, hashlib.sha256(path.read_bytes()).hexdigest()


def conflict_features(dates: pd.DatetimeIndex, lag: int) -> tuple[pd.DataFrame, str]:
    """Phase 4.3 model inputs: log1p political-violence events and fatalities of month(t) − lag."""
    m, sha = cached_monthly()
    return lagged_features(np.log1p(m[["pv_events", "pv_fatalities"]]), dates, lags=(lag,)), sha


def crosscheck(built: pd.Series, cmy: pd.Series, start: str, end: str, tol_pct: float = 5.0) -> dict:
    """Month-by-month comparison of the weekly build with the country-month-year file over [start, end]."""
    idx = pd.period_range(start, end, freq="M")
    a, b = built.reindex(idx), cmy.reindex(idx)
    assert not a.isna().any() and not b.isna().any(), "months missing from one of the series"
    d = a - b
    pct = 100 * d / b.where(b != 0, np.nan)
    flagged = [{"month": str(m), "built": float(a[m]), "cmy": float(b[m]), "diff": float(d[m]),
                "diff_pct": None if np.isnan(pct[m]) else float(pct[m])}
               for m in idx if (b[m] == 0 and a[m] != 0) or (b[m] != 0 and abs(pct[m]) > tol_pct)]
    return {"months": len(idx), "pearson_r": float(np.corrcoef(a, b)[0, 1]),
            "mean_abs_diff": float(d.abs().mean()), "max_abs_diff": float(d.abs().max()),
            "max_abs_diff_month": str(d.abs().idxmax()), "total_built": float(a.sum()), "total_cmy": float(b.sum()),
            "n_months_over_tol": len(flagged), "tol_pct": tol_pct, "flagged": flagged}
