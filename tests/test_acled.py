import numpy as np
import pandas as pd
import pytest

from src.data.acled import lagged_features, monthly_series, n_weeks_spanning_two_months, week_month_shares


DISORDER = {"Battles": "Political violence", "Violence against civilians": "Political violence",
            "Protests": "Demonstrations"}


def _weekly(rows):
    return pd.DataFrame(rows, columns=["WEEK", "EVENT_TYPE", "EVENTS", "FATALITIES"]).assign(
        WEEK=lambda d: pd.to_datetime(d["WEEK"]), DISORDER_TYPE=lambda d: d["EVENT_TYPE"].map(DISORDER))


def test_prorata_splits_by_days():
    # Saturday 2021-01-30 covers Jan 30-31 (2 days) and Feb 1-5 (5 days)
    s = week_month_shares(pd.Timestamp("2021-01-30"), "prorata")
    assert s == {pd.Period("2021-01"): 2 / 7, pd.Period("2021-02"): 5 / 7}
    assert week_month_shares(pd.Timestamp("2021-01-30"), "start") == {pd.Period("2021-01"): 1.0}


def test_monthly_series_conserves_totals_and_separates_pv():
    w = _weekly([("2021-01-30", "Battles", 7, 14), ("2021-01-30", "Protests", 70, 0), ("2021-02-06", "Violence against civilians", 1, 0)])
    for rule in ("prorata", "start"):
        m = monthly_series(w, rule)
        assert np.isclose(m["pv_events"].sum(), 8) and np.isclose(m["all_events"].sum(), 78)
        assert np.isclose(m["pv_fatalities"].sum(), 14)
    m = monthly_series(w, "prorata")
    assert np.isclose(m.loc[pd.Period("2021-01"), "pv_events"], 2) and np.isclose(m.loc[pd.Period("2021-02"), "pv_events"], 6)
    assert n_weeks_spanning_two_months(w) == 1


def test_definitions_differ_only_on_disorder_coded_protests():
    w = _weekly([("2021-02-06", "Battles", 3, 1), ("2021-02-06", "Protests", 5, 2), ("2021-02-06", "Protests", 10, 0)])
    w.loc[1, "DISORDER_TYPE"] = "Political violence; Demonstrations"   # e.g. excessive force against protesters
    a, b = monthly_series(w, "prorata", "disorder_type"), monthly_series(w, "prorata", "event_types")
    assert np.isclose(a["pv_events"].sum(), 8) and np.isclose(a["pv_fatalities"].sum(), 3)
    assert np.isclose(b["pv_events"].sum(), 3) and np.isclose(b["pv_fatalities"].sum(), 1)
    assert np.isclose(a["all_events"].sum(), 18) and a["all_events"].equals(b["all_events"])


def test_lag_uses_only_completed_months():
    months = pd.period_range("2021-01", "2021-06", freq="M")
    monthly = pd.DataFrame({"x": np.arange(len(months), dtype=float)}, index=months)  # Jan=0 … Jun=5
    dates = pd.to_datetime(["2021-04-01", "2021-04-30", "2021-02-15"])
    f = lagged_features(monthly, dates, lags=(1, 2, 3), cols=["x"])
    assert f.loc["2021-04-01", "x_lag1m"] == 2 and f.loc["2021-04-30", "x_lag1m"] == 2   # March, never April
    assert f.loc["2021-04-30", "x_lag3m"] == 0
    assert np.isnan(f.loc["2021-02-15", "x_lag2m"])                                     # Dec 2020 not in data
    # Changing the current or any later month never changes the features
    later = monthly.copy()
    later.loc[pd.Period("2021-04"):, "x"] = 1e6
    assert f.equals(lagged_features(later, dates, lags=(1, 2, 3), cols=["x"]))
    with pytest.raises(ValueError):
        lagged_features(monthly, dates, lags=(0,), cols=["x"])
