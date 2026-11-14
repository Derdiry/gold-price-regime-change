"""
Data fetchers for both Global and Sudan tracks.

Sources:
  - yfinance       : gold prices, USD index, Brent crude
  - FRED API       : US macro (10Y yield, CPI, M2)
  - World Bank     : Sudan macro (CPI, GDP, official FX)
  - ACLED          : Sudan conflict event data (manual CSV download)
  - stooq          : regional gold proxies (Egypt, Dubai)
"""

import os
import logging
from pathlib import Path
from typing import Optional

import time
import pandas as pd
import yfinance as yf
import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Global Track — yfinance
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yfinance(
    tickers: dict,
    start: str,
    end: str,
    save_dir: Optional[str] = None,
    delay: float = 1.5,
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV data for a dict of {name: ticker} from Yahoo Finance.

    yfinance 1.2+ manages its own curl_cffi session internally — do NOT pass
    a requests.Session. Small delay between tickers avoids rate limiting.

    Returns a dict of {name: DataFrame} with columns: Open, High, Low, Close, Volume.
    """
    results = {}

    for name, ticker in tickers.items():
        logger.info(f"Fetching {name} ({ticker}) from {start} to {end}")
        try:
            df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                time.sleep(delay)
                continue
            df.index = pd.to_datetime(df.index).tz_localize(None)  # strip timezone
            df.index.name = "date"
            df.columns = [c.lower() for c in df.columns]
            results[name] = df
            if save_dir:
                path = Path(save_dir) / f"{name}.csv"
                df.to_csv(path)
                logger.info(f"Saved {name} to {path}")
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")

        time.sleep(delay)

    return results


def fetch_global_prices(cfg: dict, save_dir: str = "data/raw/global") -> dict[str, pd.DataFrame]:
    """Fetch all global track price data using config."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    tickers = cfg["data"]["global"]["tickers"]
    start = cfg["data"]["global"]["start_date"]
    end = cfg["data"]["global"]["end_date"]
    return fetch_yfinance(tickers, start, end, save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Global Track — FRED API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_fred_series(
    series_ids: dict,
    start: str,
    end: str,
    api_key: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch macro time series from FRED.

    Requires FRED_API_KEY in environment or passed as argument.
    Series are merged into a single DataFrame on date index.
    """
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY not set. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
    except ImportError:
        raise ImportError("pip install fredapi")

    frames = {}
    for name, series_id in series_ids.items():
        logger.info(f"Fetching FRED series {series_id} ({name})")
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            frames[name] = s.rename(name)
        except Exception as e:
            logger.error(f"Failed to fetch FRED {series_id}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames.values(), axis=1)
    df.index.name = "date"
    df = df.sort_index()

    if save_dir:
        path = Path(save_dir) / "fred_macro.csv"
        df.to_csv(path)
        logger.info(f"Saved FRED data to {path}")

    return df


def fetch_global_macro(cfg: dict, save_dir: str = "data/raw/global") -> pd.DataFrame:
    """Fetch all FRED macro series using config."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    series = cfg["data"]["global"]["fred_series"]
    start = cfg["data"]["global"]["start_date"]
    end = cfg["data"]["global"]["end_date"]
    return fetch_fred_series(series, start, end, save_dir=save_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Sudan Track — Load existing CSV
# ─────────────────────────────────────────────────────────────────────────────

SUDAN_RAW_COLS = {
    "date": "date",
    "Price": "price_sdg",        # SDG per gram — PRIMARY TARGET
    "P/O 24": "intl_price_usd",  # international price per ounce (USD)
    "p/g 21": "price_per_gram_21k",
    "local USD": "fx_rate",      # SDG/USD exchange rate
    "Open": "open",
    "High": "high",
    "Low": "low",
    "vol": "volume",
}


def load_sudan_raw(
    filepath: str = "data/raw/new_Data.csv",
    duplicate_rule: str = "first",
    return_report: bool = False,
):
    """
    Load and normalise the existing Sudan gold dataset.

    Column renaming:
      Price      → price_sdg      (target)
      P/O 24     → intl_price_usd
      local USD  → fx_rate        (SDG/USD exchange rate)

    Cleaning (src/data/quality.py; the file on disk is never changed):
      - decimal-shift outliers in open/high/low are restored
      - dates with several rows are collapsed by ``duplicate_rule``

    With ``return_report=True`` returns (df, report) where report lists every
    correction and the row counts it affected.
    """
    from src.data.quality import collapse_duplicate_dates, fix_decimal_shift_outliers

    df = pd.read_csv(filepath)
    df = df.rename(columns=SUDAN_RAW_COLS)

    # Parse dates — original format: "6-Jan-2015". A stable sort keeps file order within a date.
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    # Coerce numerics
    numeric_cols = [c for c in df.columns if c != "date"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df, fixes = fix_decimal_shift_outliers(df)
    df, dup_report = collapse_duplicate_dates(df, rule=duplicate_rule)

    logger.info(f"Loaded Sudan raw data: {len(df)} rows, {df.index.min()} to {df.index.max()}")
    if return_report:
        return df, {"decimal_shift_fixes": fixes, "duplicate_dates": dup_report}
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Sudan Track — World Bank API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_world_bank_sudan(
    indicators: dict,
    country: str = "SDN",
    start_year: int = 2015,
    end_year: int = 2022,
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch annual macro indicators from the World Bank for Sudan.

    Indicators (examples from config):
      FP.CPI.TOTL.ZG  → CPI inflation %
      NY.GDP.MKTP.CD  → GDP current USD
      PA.NUS.FCRF     → official exchange rate
    """
    try:
        import wbdata
    except ImportError:
        raise ImportError("pip install wbdata")

    import datetime
    date_range = (datetime.datetime(start_year, 1, 1), datetime.datetime(end_year, 12, 31))

    logger.info(f"Fetching World Bank data for {country}: {list(indicators.keys())}")
    try:
        df = wbdata.get_dataframe(indicators, country=country, date=date_range)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.index.name = "date"
    except Exception as e:
        logger.error(f"World Bank fetch failed: {e}")
        return pd.DataFrame()

    if save_dir:
        path = Path(save_dir) / "world_bank_sudan.csv"
        df.to_csv(path)
        logger.info(f"Saved World Bank data to {path}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Sudan Track — ACLED Conflict Data
# ─────────────────────────────────────────────────────────────────────────────

def load_acled_sudan(
    filepath: str = "data/raw/sudan/acled_sudan.csv",
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load ACLED conflict event data for Sudan.

    Download from https://acleddata.com (free for researchers).
    Filter: Country=Sudan, date range 2015–2022.

    Returns monthly aggregated conflict intensity:
      - event_count   : total events per month
      - fatalities    : total fatalities per month
      - conflict_score: weighted composite (event_count + 2 * fatalities), normalised
    """
    path = Path(filepath)
    if not path.exists():
        logger.warning(
            f"ACLED file not found at {filepath}. "
            "Download from https://acleddata.com and place at that path. "
            "Returning empty DataFrame."
        )
        return pd.DataFrame()

    df = pd.read_csv(filepath, low_memory=False)

    # ACLED standard column names
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col is None:
        logger.error("No date column found in ACLED data")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])

    fatality_col = next((c for c in df.columns if "fatal" in c.lower()), None)

    # Aggregate to monthly
    df["month"] = df["date"].dt.to_period("M")
    agg = {"date": "count"}
    if fatality_col:
        agg[fatality_col] = "sum"

    monthly = df.groupby("month").agg(agg).reset_index()
    monthly = monthly.rename(columns={"date": "event_count"})
    if fatality_col:
        monthly = monthly.rename(columns={fatality_col: "fatalities"})
    else:
        monthly["fatalities"] = 0

    monthly["conflict_score"] = (
        monthly["event_count"] + 2 * monthly["fatalities"]
    )
    # Normalise to [0, 1]
    max_score = monthly["conflict_score"].max()
    if max_score > 0:
        monthly["conflict_score"] = monthly["conflict_score"] / max_score

    monthly["date"] = monthly["month"].dt.to_timestamp()
    monthly = monthly.set_index("date").drop(columns=["month"])
    monthly.index.name = "date"

    if save_dir:
        path_out = Path(save_dir) / "acled_monthly.csv"
        monthly.to_csv(path_out)
        logger.info(f"Saved ACLED monthly data to {path_out}")

    logger.info(f"Loaded ACLED data: {len(monthly)} months")
    return monthly


# ─────────────────────────────────────────────────────────────────────────────
# Sudan Track — Regional Gold Prices (Egypt, Dubai)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_regional_gold(
    start: str = "2015-01-01",
    end: str = "2022-12-31",
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch regional gold price proxies for arbitrage features.

    Uses stooq via pandas_datareader as a fallback since direct Egypt/Dubai
    gold prices are not available on yfinance.

    Falls back to international gold price if regional data is unavailable.
    """
    frames = {}

    # Try to get Egypt ETF as proxy
    try:
        df_egypt = yf.download("EGPT", start=start, end=end, progress=False, auto_adjust=True)
        if not df_egypt.empty:
            df_egypt.columns = [c.lower() for c in df_egypt.columns]
            frames["egypt_etf"] = df_egypt["close"].rename("egypt_gold_proxy")
            logger.info("Fetched Egypt gold proxy (EGPT ETF)")
    except Exception as e:
        logger.warning(f"Egypt proxy fetch failed: {e}")

    # Try to get gold in USD as regional reference
    try:
        df_gold = yf.download("GC=F", start=start, end=end, progress=False, auto_adjust=True)
        if not df_gold.empty:
            df_gold.columns = [c.lower() for c in df_gold.columns]
            frames["intl_gold_ref"] = df_gold["close"].rename("intl_gold_close")
    except Exception as e:
        logger.warning(f"International gold reference fetch failed: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames.values(), axis=1)
    df.index.name = "date"
    df = df.sort_index()

    if save_dir:
        path = Path(save_dir) / "regional_gold.csv"
        df.to_csv(path)
        logger.info(f"Saved regional gold data to {path}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: fetch everything
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_global(cfg: dict) -> dict:
    """Fetch all global track data. Returns dict of DataFrames."""
    prices = fetch_global_prices(cfg)
    macro = fetch_global_macro(cfg)
    return {"prices": prices, "macro": macro}


def fetch_all_sudan(cfg: dict) -> dict:
    """Load and fetch all Sudan track data. Returns dict of DataFrames."""
    wb_cfg = cfg["data"]["sudan"]["world_bank"]
    sudan_cfg = cfg["data"]["sudan"]

    raw = load_sudan_raw(sudan_cfg["raw_file"])
    world_bank = fetch_world_bank_sudan(
        indicators=wb_cfg["indicators"],
        country=wb_cfg["country_code"],
        start_year=int(sudan_cfg["start_date"][:4]),
        end_year=int(sudan_cfg["end_date"][:4]),
        save_dir="data/raw/sudan",
    )
    acled = load_acled_sudan(
        filepath=sudan_cfg.get("acled_file", "data/raw/sudan/acled_sudan.csv"),
        save_dir="data/raw/sudan",
    )
    regional = fetch_regional_gold(
        start=sudan_cfg["start_date"],
        end=sudan_cfg["end_date"],
        save_dir="data/raw/sudan",
    )

    return {
        "raw": raw,
        "world_bank": world_bank,
        "acled": acled,
        "regional": regional,
    }
