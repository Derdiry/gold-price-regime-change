"""
Deterministic, offline construction of the processed datasets (protocol v2).

Both tracks are rebuilt from raw files on disk on every run, so the processed
data always matches the committed code:

  Sudan:  data/raw/new_Data.csv (the author's dataset)
  Global: data/raw/global/*.csv, a snapshot of the Yahoo Finance / FRED
          downloads made on 2026-04-04 (re-download with
          ``train.py --refetch-global``; that changes the snapshot)

Why: the v1 processed files (data/processed/{global,sudan}_merged.csv) were
written by an earlier, uncommitted version of the feature code (log returns
instead of percentage returns, rolling features with min_periods=window), so
the committed code cannot reproduce them. v2 never reads them.

The World Bank, regional-price and ACLED sources are off by default
(``data.sudan.sources``). In v1 their fetches returned nothing, so the
reported v1 models used only new_Data.csv; v2 keeps that feature set.

Every build writes ``<processed_dir>/data_manifest_<track>.json`` with the
SHA-256 of each input file, all data-quality corrections, and the row counts.
"""

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from src.data.features import add_all_sudan_features, add_all_technical
from src.data.fetchers import fetch_all_global, load_sudan_raw
from src.data.processors import merge_global_track, merge_sudan_track, save_processed
from src.utils.provenance import REPO_ROOT

logger = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(cfg: dict, track: str, manifest: dict) -> Path:
    out = Path(cfg["paths"]["processed_data"]) / f"data_manifest_{track}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info(f"Data manifest: {out}")
    return out


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_sudan(cfg: dict) -> pd.DataFrame:
    """Load, clean, merge and feature-engineer the Sudan track from new_Data.csv."""
    s_cfg = cfg["data"]["sudan"]
    raw_path = Path(s_cfg["raw_file"])
    raw, quality = load_sudan_raw(
        str(raw_path), duplicate_rule=s_cfg["duplicate_date_rule"], return_report=True
    )
    df = merge_sudan_track(
        raw, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        max_gap_days=s_cfg["max_ffill_gap_days"],
    )
    fill_report = df.attrs.get("fill_report")
    df = add_all_sudan_features(df, cfg)
    df = add_all_technical(df, price_col=cfg["data"]["target_col"], cfg=cfg, high_col="high", low_col="low")
    df.attrs = {}
    save_processed(df, "sudan", save_dir=cfg["paths"]["processed_data"])

    _write_manifest(cfg, "sudan", {
        "inputs": {_rel(raw_path): sha256(raw_path)},
        "quality": {**quality, "fill_flags": fill_report},
        "rows": len(df),
        "date_range": [str(df.index.min().date()), str(df.index.max().date())],
        "columns": list(df.columns),
    })
    return df


def build_global(cfg: dict, refetch: bool = False) -> pd.DataFrame:
    """Merge and feature-engineer the global track from the raw snapshot (or a fresh download)."""
    raw_dir = Path(cfg["paths"]["raw_data"]) / "global"
    if refetch:
        logger.warning("Re-downloading global data: this replaces the raw snapshot in data/raw/global/")
        fetch_all_global(cfg)
    names = list(cfg["data"]["global"]["tickers"])
    files = {n: raw_dir / f"{n}.csv" for n in names}
    files["fred_macro"] = raw_dir / "fred_macro.csv"
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Global raw snapshot incomplete: {missing}. Run train.py --refetch-global.")

    prices = {n: pd.read_csv(files[n], index_col="date", parse_dates=True) for n in names}
    macro = pd.read_csv(files["fred_macro"], index_col="date", parse_dates=True)
    df = merge_global_track(prices, macro)
    df = add_all_technical(
        df, price_col=cfg["data"]["global_target_col"], cfg=cfg,
        high_col="gold_high", low_col="gold_low",
    )
    save_processed(df, "global", save_dir=cfg["paths"]["processed_data"])

    _write_manifest(cfg, "global", {
        "inputs": {_rel(p): sha256(p) for p in files.values()},
        "rows": len(df),
        "date_range": [str(df.index.min().date()), str(df.index.max().date())],
        "columns": list(df.columns),
    })
    return df
