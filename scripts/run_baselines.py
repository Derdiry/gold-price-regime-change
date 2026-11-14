"""
Phase 2 baselines on exactly the v2 validation and test windows.

For each track and horizon, scores persistence, drift, ARIMA and ETS on log
price, and (Sudan) a VECM, in original price units with all metrics.
Model settings of the statistical families are chosen per horizon by
validation MAE; the test windows are forecast only for the chosen settings.

Outputs
  results/baselines/<track>_h<h>.json   val/test metrics, the selection table, provenance
  results/baselines/summary.md          test table (generated)
  logs/v2/preds/baselines/*.npz         forecasts (gitignored; reused by later phases)

Usage
    python scripts/run_baselines.py
    python scripts/run_baselines.py --track sudan --duplicate-rule last   # D1 sensitivity
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import make_dataset
from src.data.features import GLOBAL_FEATURE_COLS, SUDAN_FEATURE_COLS
from src.data.fetchers import load_config
from src.data.pipeline import build_global, build_sudan
from src.evaluation import baselines as bl
from src.evaluation.metrics import compute_all_metrics, compute_mae
from src.utils.provenance import provenance
from src.utils.resources import require_memory
from src.utils.run_logging import setup_run_logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("baselines")
for noisy in ("src.data", "src.data.processors", "src.data.dataset", "src.data.fetchers", "src.data.quality"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

MAX_HISTORY = 750   # rows of history for the statistical models
VECM_COLS = ["price_sdg", "intl_price_usd", "fx_rate"]


def _run_chunk(args):
    family, cfg, histories, max_h = args
    return bl.statistical_paths(family, cfg, histories, max_h)


def forecast_all(family, cfg, histories, max_h, workers) -> tuple[np.ndarray, int]:
    """Parallel statistical_paths over origin chunks, order preserved."""
    n_chunks = max(1, min(len(histories), workers * 4))
    chunks = np.array_split(np.arange(len(histories)), n_chunks)
    jobs = [(family, cfg, [histories[i] for i in idx], max_h) for idx in chunks if len(idx)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_run_chunk, jobs))
    return np.concatenate([p for p, _ in parts]), sum(f for _, f in parts)


def histories_for(df, positions, invalid, family, target_col):
    """History ending at each origin: <= MAX_HISTORY rows, not crossing an invalid row."""
    cols = VECM_COLS if family == "vecm" else [target_col]
    values = df[cols].to_numpy(float)
    last_invalid = np.maximum.accumulate(np.where(invalid, np.arange(len(df)), -1))
    out = []
    for p in positions:
        start = max(p - MAX_HISTORY + 1, last_invalid[p] + 1)
        h = values[start:p + 1]
        out.append(h[:, 0] if family != "vecm" else h)
    return out


def run_track(track, cfg, workers, out_dir, pred_dir):
    df = build_global(cfg) if track == "global" else build_sudan(cfg)
    target_col = cfg["data"]["global_target_col"] if track == "global" else cfg["data"]["target_col"]
    feature_list = GLOBAL_FEATURE_COLS if track == "global" else SUDAN_FEATURE_COLS
    L = cfg["sequence"]["input_length"]
    horizons = cfg["sequence"]["horizons"]
    max_h = max(horizons)
    target = df[target_col].to_numpy(float)
    invalid = (df["in_long_gap"] == 1).to_numpy() if "in_long_gap" in df.columns else np.zeros(len(df), bool)

    # Windows for every horizon, exactly as the neural models see them
    win = {}
    for h in horizons:
        splits, (_, t_scaler), _, split_info = make_dataset(df, cfg, track, h, feature_list, target_col, save=False)
        win[h] = {"split_info": split_info}
        for s in ("val", "test"):
            pos = df.index.get_indexer(pd.to_datetime(splits[f"origin_{s}"]))
            assert (pos >= 0).all()
            y_true = np.stack([target[pos + 1 + k] for k in range(h)], axis=1)
            y_model = t_scaler.inverse_transform(splits[f"y_{s}"].reshape(-1, 1)).reshape(-1, h)
            assert np.allclose(y_true, y_model, rtol=1e-4), "window targets do not match the model windows"
            assert np.allclose(target[pos], splits[f"last_{s}"])
            win[h][s] = {"pos": pos, "y": y_true, "last": target[pos],
                         "ctx": np.stack([target[p - L + 1:p + 1] for p in pos])}

    # Statistical families: fit once per origin of the largest origin set (h = min)
    h0 = min(horizons)
    families = ["arima", "ets"] + (["vecm"] if track == "sudan" else [])
    grids = {"arima": bl.ARIMA_GRID, "ets": bl.ETS_GRID, "vecm": bl.VECM_GRID}
    paths = {s: {} for s in ("val", "test")}           # paths[split][(family, i_cfg)] -> (n_origins, max_h)
    fallbacks = {s: {} for s in ("val", "test")}
    origin_pos = {s: win[h0][s]["pos"] for s in ("val", "test")}
    for h in horizons:
        for s in ("val", "test"):
            assert np.isin(win[h][s]["pos"], origin_pos[s]).all(), "origins are not nested across horizons"

    def rows(h, s):
        return np.searchsorted(origin_pos[s], win[h][s]["pos"])

    selected = {}
    for fam in families:
        hist_val = histories_for(df, origin_pos["val"], invalid, fam, target_col)
        for i, c in enumerate(grids[fam]):
            t0 = time.time()
            paths["val"][(fam, i)], fallbacks["val"][(fam, i)] = forecast_all(fam, c, hist_val, max_h, workers)
            logger.info(f"[{track}] {fam} {c}: {len(hist_val)} val origins in {time.time() - t0:.0f}s "
                        f"({fallbacks['val'][(fam, i)]} fallbacks)")
        for h in horizons:
            val_mae = [compute_mae(win[h]["val"]["y"], paths["val"][(fam, i)][rows(h, "val"), :h])
                       for i in range(len(grids[fam]))]
            selected[(fam, h)] = int(np.argmin(val_mae))
            win[h].setdefault("selection", {})[fam] = [
                {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()},
                 "val_mae": m, "val_fallbacks": fallbacks["val"][(fam, i)]}
                for i, (c, m) in enumerate(zip(grids[fam], val_mae))]
        # Test: only the settings chosen on validation for some horizon
        hist_test = histories_for(df, origin_pos["test"], invalid, fam, target_col)
        for i in sorted({selected[(fam, h)] for h in horizons}):
            t0 = time.time()
            paths["test"][(fam, i)], fallbacks["test"][(fam, i)] = forecast_all(fam, grids[fam][i], hist_test, max_h, workers)
            logger.info(f"[{track}] {fam} {grids[fam][i]}: {len(hist_test)} test origins in {time.time() - t0:.0f}s")

    prov = provenance(cfg)
    for h in horizons:
        result = {"track": track, "horizon": h, "provenance": prov,
                  "duplicate_date_rule": cfg["data"]["sudan"]["duplicate_date_rule"],
                  "max_history_rows": MAX_HISTORY, "context_length": L,
                  "windows": {s: int(len(win[h][s]["pos"])) for s in ("val", "test")},
                  "split_info": win[h]["split_info"], "baselines": {}}
        preds = {}
        for s in ("val", "test"):
            preds[("persistence", s)] = bl.persistence(win[h][s]["last"], h)
            preds[("drift", s)] = bl.drift(win[h][s]["ctx"], h)
            for fam in families:
                preds[(fam, s)] = paths[s][(fam, selected[(fam, h)])][rows(h, s), :h]
        for name in ["persistence", "drift", *families]:
            entry = {s: compute_all_metrics(win[h][s]["y"], preds[(name, s)], win[h][s]["last"])
                     for s in ("val", "test")}
            if name in families:
                i = selected[(name, h)]
                entry["selected_config"] = win[h]["selection"][name][i]["config"]
                entry["selected_by"] = "validation MAE"
                entry["test_fallbacks"] = fallbacks["test"][(name, i)]
                entry["selection_table"] = win[h]["selection"][name]
            result["baselines"][name] = entry
        (out_dir / f"{track}_h{h}.json").write_text(json.dumps(result, indent=2, default=str))
        np.savez_compressed(pred_dir / f"{track}_h{h}.npz",
                            **{f"{n}_{s}": p for (n, s), p in preds.items()},
                            **{f"y_{s}": win[h][s]["y"] for s in ("val", "test")},
                            **{f"last_{s}": win[h][s]["last"] for s in ("val", "test")})
        logger.info(f"[{track} h={h}] test MAPE: " + ", ".join(
            f"{n}={result['baselines'][n]['test']['mape']:.2f}%" for n in result["baselines"]))


def write_summary(out_dir: Path):
    lines = ["# Baselines on the v2 test windows (generated by scripts/run_baselines.py)", "",
             "Statistical-model settings chosen by validation MAE. MAE in price units "
             "(USD/oz global, SDG/g Sudan). DA = directional accuracy vs the previous origin; "
             "non-zero = only steps with a non-zero true change; max = ceiling for a forecast that "
             "never predicts an exact-zero change. DA-origin non-zero = direction of change from the last "
             "observed value, on steps where the price moved (persistence predicts no change, so scores 0).", "",
             "| track | h | baseline | setting | MAE | RMSE | MAPE % | DA % | DA non-zero % | DA max % | DA-origin non-zero % | test windows |",
             "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for f in sorted(out_dir.glob("*_h*.json"), key=lambda p: (p.stem.split("_h")[0], int(p.stem.split("_h")[1]))):
        r = json.loads(f.read_text())
        for name, e in r["baselines"].items():
            t = e["test"]
            setting = json.dumps(e.get("selected_config", "")).replace('"', "") if "selected_config" in e else "—"
            lines.append(f"| {r['track']} | {r['horizon']} | {name} | {setting} | {t['mae']:.2f} | {t['rmse']:.2f} | "
                         f"{t['mape']:.2f} | {t['directional_accuracy']:.1f} | "
                         f"{t['directional_accuracy_nonzero']:.1f} | {t['directional_accuracy_max']:.1f} | "
                         f"{t['da_origin_nonzero']:.1f} | {t['n_windows']} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["global", "sudan", "both"], default="both")
    ap.add_argument("--duplicate-rule", choices=["first", "last", "median"], default=None,
                    help="Override data.sudan.duplicate_date_rule (sensitivity run; separate output folder)")
    ap.add_argument("--fold", type=int, choices=[1, 2, 3, 4, 5], default=None,
                    help="Phase 4.1 rolling-origin fold: results/rolling/fold<k>/baselines")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--summary-only", action="store_true", help="Rebuild summary.md from the saved JSON files")
    ap.add_argument("--config", default="configs/model_config.yaml")
    args = ap.parse_args()

    setup_run_logging("baselines")
    cfg = load_config(args.config)
    out_dir = Path(cfg["paths"]["results"]) / "baselines"
    pred_dir = Path(cfg["paths"]["runs"]) / "preds" / "baselines"
    if args.duplicate_rule and args.duplicate_rule != cfg["data"]["sudan"]["duplicate_date_rule"]:
        cfg["data"]["sudan"]["duplicate_date_rule"] = args.duplicate_rule
        tag = f"sensitivity_duplicate_{args.duplicate_rule}"
        out_dir = out_dir / tag
        pred_dir = pred_dir / tag
        cfg["paths"]["processed_data"] = str(Path(cfg["paths"]["runs"]) / tag / "processed")
    if args.fold:
        cfg["split"]["fold"] = args.fold
        out_dir = Path(cfg["paths"]["results"]) / "rolling" / f"fold{args.fold}" / "baselines"
        pred_dir = Path(cfg["paths"]["runs"]) / "rolling" / f"fold{args.fold}" / "preds" / "baselines"
        cfg["paths"]["processed_data"] = str(Path(cfg["paths"]["runs"]) / "rolling" / f"fold{args.fold}" / "processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        write_summary(out_dir)
        return
    require_memory(cfg["training"]["min_free_memory_gb"], "baselines")
    logger.info(f"workers={args.workers} cpu={os.cpu_count()} out={out_dir}")

    for track in (["global", "sudan"] if args.track == "both" else [args.track]):
        run_track(track, cfg, args.workers, out_dir, pred_dir)
    write_summary(out_dir)


if __name__ == "__main__":
    main()
