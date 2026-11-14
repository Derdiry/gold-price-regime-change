"""Sudan raw-data cleaning (src/data/quality.py) and the real file's known issues."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.quality import collapse_duplicate_dates, fix_decimal_shift_outliers

RAW = Path(__file__).resolve().parents[1] / "data/raw/new_Data.csv"


def test_decimal_shift_fix_only_touches_shifted_values():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2016-12-02", "2016-12-03", "2016-12-04"]),
        "open": [1174.3, 117472.0, 50000.0],
        "high": [1180.3, 1182.6, 1183.8], "low": [1168.4, 1165.1, 1171.1],
        "intl_price_usd": [1177.8, 1177.8, 1173.4],
    })
    out, fixes = fix_decimal_shift_outliers(df)
    assert out["open"].tolist()[:2] == [1174.3, 1174.72]
    assert out["open"].iloc[2] == 50000.0          # 50000/100 = 500 is not near 1173 -> left alone
    assert len(fixes) == 1 and fixes[0]["date"] == "2016-12-03"


@pytest.mark.parametrize("rule,expected", [("first", 1.0), ("last", 3.0), ("median", 2.0)])
def test_duplicate_date_rules(rule, expected):
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"]),
                       "price_sdg": [1.0, 2.0, 3.0, 9.0]})
    out, report = collapse_duplicate_dates(df, rule)
    assert out.loc["2020-01-01", "price_sdg"] == expected
    assert report["dates_with_multiple_rows"] == 1 and report["rows_dropped"] == 2


@pytest.mark.skipif(not RAW.exists(), reason="raw Sudan file not present")
def test_real_file_known_issues():
    from src.data.fetchers import load_sudan_raw
    df, report = load_sudan_raw(str(RAW), return_report=True)
    assert [f["date"] for f in report["decimal_shift_fixes"]] == ["2016-12-03"]
    assert report["duplicate_dates"]["file_rows"] == 2528
    assert report["duplicate_dates"]["dates_with_multiple_rows"] == 635
    # after the fix every open lies within 10% of the day's low/high range
    assert ((df["open"] >= 0.9 * df["low"]) & (df["open"] <= 1.1 * df["high"])).all()
