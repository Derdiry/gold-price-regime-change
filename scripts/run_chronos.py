"""
Phase 4.2: Chronos zero-shot on the v2 validation and test windows.

Rules are fixed in results/chronos/PREREGISTRATION.md. Two steps:

  predict  30-step sample paths at every val/test origin, per (size, track, seed);
           saved to logs/v2/preds/chronos/<size>_<track>_s<seed>.npz (resumable)
  score    metrics, MASE and DM vs persistence per horizon; refuses to run
           unless the preregistration is committed

Outputs: results/chronos/{chronos_results.json, chronos_table.md, dm_tests.csv}

    python scripts/run_chronos.py predict [--sizes mini small] [--tracks sudan global]
    python scripts/run_chronos.py score
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.data.dataset import make_dataset  # noqa: E402
from src.data.features import GLOBAL_FEATURE_COLS, SUDAN_FEATURE_COLS  # noqa: E402
from src.data.fetchers import load_config  # noqa: E402
from src.data.pipeline import build_global, build_sudan  # noqa: E402
from src.evaluation.metrics import compute_all_metrics  # noqa: E402
from src.evaluation.stats import diebold_mariano, holm, mase_vs_persistence  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402
from src.utils.resources import InsufficientMemory, require_memory  # noqa: E402
from src.utils.run_logging import setup_run_logging  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("chronos")
for noisy in ("src.data", "src.data.processors", "src.data.dataset", "src.data.fetchers", "src.data.quality"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

SIZES = {"mini": "amazon/chronos-t5-mini", "small": "amazon/chronos-t5-small"}
TRACKS, HORIZONS, SEEDS = ["sudan", "global"], [1, 7, 30], [1, 2, 3]
MAX_CONTEXT, MAX_H, NUM_SAMPLES, BATCH = 512, 30, 20, 8   # batch 8: peak RSS 2.7 GB (mini), 3.7 GB (small)
TORCH_THREADS = 8                                          # measured: 16 threads is ~2x slower than 8
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
LOSSES, ALPHA = ["squared", "absolute"], 0.05
PRED_DIR, BASE_DIR = Path("logs/v2/preds/chronos"), Path("logs/v2/preds/baselines")
OUT = Path("results/chronos")
PREREG = OUT / "PREREGISTRATION.md"


def windows(track, cfg):
    """Origin positions per (h, split), the h=1 origin set (a superset) and target histories."""
    df = build_global(cfg) if track == "global" else build_sudan(cfg)
    tc = cfg["data"]["global_target_col"] if track == "global" else cfg["data"]["target_col"]
    fl = GLOBAL_FEATURE_COLS if track == "global" else SUDAN_FEATURE_COLS
    target = df[tc].to_numpy(float)
    invalid = (df["in_long_gap"] == 1).to_numpy() if "in_long_gap" in df.columns else np.zeros(len(df), bool)
    last_invalid = np.maximum.accumulate(np.where(invalid, np.arange(len(df)), -1))
    pos = {}
    for h in HORIZONS:
        sp, _, _, _ = make_dataset(df, cfg, track, h, fl, tc, save=False)
        for s in ("val", "test"):
            pos[(h, s)] = df.index.get_indexer(pd.to_datetime(sp[f"origin_{s}"]))
    origins = {s: pos[(min(HORIZONS), s)] for s in ("val", "test")}
    for (h, s), p in pos.items():
        assert np.isin(p, origins[s]).all(), "origins are not nested across horizons"
    hist = {s: [target[max(p - MAX_CONTEXT + 1, last_invalid[p] + 1):p + 1] for p in origins[s]] for s in ("val", "test")}
    return pos, origins, hist


def predict(sizes, tracks, cfg):
    import torch
    from chronos import ChronosPipeline
    torch.set_num_threads(TORCH_THREADS)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for size in sizes:
        pipe = None
        for track in tracks:
            todo = [s for s in SEEDS if not (PRED_DIR / f"{size}_{track}_s{s}.npz").exists()]
            if not todo:
                logger.info(f"{size} {track}: all seeds done")
                continue
            require_memory(cfg["training"].get("min_free_memory_gb", 3.0), f"Chronos {size} {track}")
            if pipe is None:
                pipe = ChronosPipeline.from_pretrained(SIZES[size], device_map="cpu", torch_dtype=torch.float32)
                n_params = int(sum(p.numel() for p in pipe.model.parameters()))
                logger.info(f"{SIZES[size]}: {n_params:,} parameters")
            _, _, hist = windows(track, cfg)
            for seed in todo:
                torch.manual_seed(seed)
                t0, out = time.time(), {}
                for s in ("val", "test"):
                    samples = []
                    for i in range(0, len(hist[s]), BATCH):
                        ctx = [torch.tensor(x, dtype=torch.float32) for x in hist[s][i:i + BATCH]]
                        samples.append(pipe.predict(ctx, prediction_length=MAX_H, num_samples=NUM_SAMPLES,
                                                    limit_prediction_length=False).numpy())
                    smp = np.concatenate(samples)                        # (n_origins, samples, MAX_H)
                    out[f"{s}_samples"] = smp.astype(np.float32)
                    out[f"{s}_median"] = np.median(smp, axis=1)
                    out[f"{s}_quantiles"] = np.quantile(smp, QUANTILES, axis=1).transpose(1, 0, 2)
                    out[f"{s}_context_len"] = np.array([len(x) for x in hist[s]])
                secs = time.time() - t0
                np.savez_compressed(PRED_DIR / f"{size}_{track}_s{seed}.npz", **out, quantile_levels=np.array(QUANTILES),
                                    n_params=n_params, seconds=secs, model_id=SIZES[size])
                logger.info(f"{size} {track} seed {seed}: {sum(len(v) for v in hist.values())} origins in {secs:.0f}s")


def require_committed_preregistration():
    import subprocess
    rel = str(PREREG)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = not subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if not (tracked and clean):
        sys.exit(f"{rel} must be committed (and unchanged) before scoring.")
    return subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def score_one(y, f, last, pers, h, mask=None):
    if mask is None:
        m = compute_all_metrics(y, f, last)
        out = {"mae": m["mae"], "rmse": m["rmse"], "mape": m["mape"], "da_nonzero": m["directional_accuracy_nonzero"],
               "da_origin_nonzero": m["da_origin_nonzero"], "mase": mase_vs_persistence(y, f, pers), "n": m["n_windows"]}
        for loss in LOSSES:
            out[f"dm_{loss}"] = diebold_mariano(y, f, pers, h, loss)
        return out
    from posthoc_observed_days import score as masked_score
    return masked_score(y, f, last, pers, h, mask)


def verdict(seed_scores, loss, pkey="p_value"):
    dms = [s[f"dm_{loss}"] for s in seed_scores]
    if all(d["dm_hln"] < 0 and d[pkey] < ALPHA for d in dms):
        return "beats persistence"
    if all(d["dm_hln"] > 0 and d[pkey] < ALPHA for d in dms):
        return "worse"
    return "no significant difference"


def score(cfg):
    prereg_commit = require_committed_preregistration()
    from posthoc_observed_days import observed_masks
    obs_masks = observed_masks(cfg)
    results, n_params, seconds = {}, {}, {}
    for track in TRACKS:
        pos, origins, hist = windows(track, cfg)
        for h in HORIZONS:
            b = np.load(BASE_DIR / f"{track}_h{h}.npz")
            cell = {"val_persistence_mae": float(np.abs(b["y_val"] - b["persistence_val"]).mean()),
                    "test_persistence_mae": float(np.abs(b["y_test"] - b["persistence_test"]).mean()), "sizes": {}}
            for size in SIZES:
                seeds = []
                for seed in SEEDS:
                    z = np.load(PRED_DIR / f"{size}_{track}_s{seed}.npz")
                    n_params[size] = int(z["n_params"])
                    seconds[f"{size}_{track}_s{seed}"] = float(z["seconds"])
                    rec = {}
                    for s in ("val", "test"):
                        rows = np.searchsorted(origins[s], pos[(h, s)])
                        f = z[f"{s}_median"][rows, :h]
                        assert np.allclose([hist[s][r][-1] for r in rows], b[f"last_{s}"]), "context does not end at the origin"
                        assert np.allclose(z[f"{s}_median"].shape[0], len(origins[s]))
                        rec[s] = score_one(b[f"y_{s}"], f, b[f"last_{s}"], b[f"persistence_{s}"], h)
                        if track == "sudan" and s == "test":
                            rec["test_S1"] = score_one(b["y_test"], f, b["last_test"], b["persistence_test"], h, obs_masks[h]["obs"])
                    seeds.append(rec)
                cell["sizes"][size] = {"seeds": seeds}
            results[f"{track}_h{h}"] = cell

    for part in ("test", "test_S1"):
        for loss in LOSSES:
            refs = [(c, i) for cell in results.values() for c in cell["sizes"].values() for i in range(len(SEEDS))
                    if part in c["seeds"][i]]
            for (c, i), p in zip(refs, holm([c["seeds"][i][part][f"dm_{loss}"]["p_value"] for c, i in refs])):
                c["seeds"][i][part][f"dm_{loss}"]["p_holm"] = p
    for cell in results.values():
        for c in cell["sizes"].values():
            for part in ("test", "test_S1"):
                if part not in c["seeds"][0]:
                    continue
                S = [s[part] for s in c["seeds"]]
                c[part] = {"mase_mean": float(np.mean([s["mase"] for s in S])),
                           **{f"verdict_{loss}": verdict(S, loss) for loss in LOSSES},
                           **{f"verdict_{loss}_holm": verdict(S, loss, "p_holm") for loss in LOSSES}}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chronos_results.json").write_text(json.dumps(
        {"provenance": provenance(cfg), "preregistration_commit": prereg_commit, "preregistration_text": PREREG.read_text(),
         "models": SIZES, "n_params": n_params, "predict_seconds": seconds, "max_context": MAX_CONTEXT,
         "num_samples": NUM_SAMPLES, "point_forecast": "median of samples", "alpha": ALPHA, "results": results},
        indent=2, default=float))
    rows = []
    for key, cell in results.items():
        for size, c in cell["sizes"].items():
            for seed, sc in zip(SEEDS, c["seeds"]):
                for part in ("test", "test_S1"):
                    if part in sc:
                        for loss in LOSSES:
                            d = sc[part][f"dm_{loss}"]
                            rows.append({"cell": key, "size": size, "seed": seed, "subset": part, "loss": loss,
                                         "dm_hln": d["dm_hln"], "p_value": d["p_value"], "p_holm": d["p_holm"], "n": d["n"]})
    pd.DataFrame(rows).to_csv(OUT / "dm_tests.csv", index=False)
    write_markdown(results, n_params, prereg_commit)


def ms(vals, fmt="{:.2f}"):
    v = np.asarray(vals, float)
    return f"{fmt.format(v.mean())} ± {fmt.format(v.std(ddof=1))}"


def write_markdown(results, n_params, prereg_commit):
    final = json.loads(Path("results/final/test_results.json").read_text())["results"]
    L = ["# Chronos zero-shot (generated by scripts/run_chronos.py)", "",
         f"Rules fixed in `{PREREG}` (commit `{prereg_commit[:7]}`) before scoring. Parameters: "
         + ", ".join(f"{s} {n:,}" for s, n in n_params.items()) + ". Point forecast = median of 20 samples; "
         "context ≤ 512 rows, not crossing the 404-day gap. Mean ± sample std over sampling seeds 1–3. "
         "DM vs persistence (HLN, Newey–West lag h − 1); p for seeds 1/2/3 with sign (− = lower loss than persistence). "
         "Verdict: sign agrees and p < 0.05 in all three seeds.", "",
         "Reference rows: best (lowest seed-mean test MASE) validation-selected Phase 3 configuration, and persistence.", ""]
    beats = []
    for key, cell in results.items():
        track, h = key.split("_h")
        L += [f"## {track} h={h}", "",
              "| model | subset | val MAE | val MASE | test MAE | test MAPE % | test MASE | DA non-zero % | DM p sq | DM p abs | verdict sq | verdict abs | Holm sq / abs |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|"]
        for size, c in cell["sizes"].items():
            for part in ("test", "test_S1"):
                if part not in c:
                    continue
                S = c["seeds"]
                T = [s[part] for s in S]
                pv = {loss: "/".join("%.3f%s" % (t[f"dm_{loss}"]["p_value"], "−" if t[f"dm_{loss}"]["dm_hln"] < 0 else "+") for t in T)
                      for loss in LOSSES}
                da = ms([t.get("da_nonzero", t.get("da_origin_nonzero")) for t in T], "{:.1f}")
                val = (ms([s["val"]["mae"] for s in S]), ms([s["val"]["mase"] for s in S], "{:.3f}")) if part == "test" else ("—", "—")
                v = c[part]
                L.append(f"| chronos-{size} | {'all' if part == 'test' else 'S1 (observed)'} | {val[0]} | {val[1]} | "
                         f"{ms([t['mae'] for t in T])} | {ms([t['mape'] for t in T])} | {ms([t['mase'] for t in T], '{:.3f}')} | {da} | "
                         f"{pv['squared']} | {pv['absolute']} | {v['verdict_squared']} | {v['verdict_absolute']} | "
                         f"{v['verdict_squared_holm']} / {v['verdict_absolute_holm']} |")
                for loss in LOSSES:
                    if v[f"verdict_{loss}"] == "beats persistence":
                        beats.append(f"{track} h={h} chronos-{size} ({part}, {loss}{', survives Holm' if v[f'verdict_{loss}_holm'] == 'beats persistence' else ', not after Holm'})")
        fc = final[key]["configs"]
        best = min((n for n in fc if n.endswith("|selected")), key=lambda n: np.mean([s["mase"] for s in fc[n]["seeds"]]))
        S = fc[best]["seeds"]
        L.append(f"| *{best.split('|')[0]} ({fc[best]['condition']}), best Phase 3* | all | — | — | {ms([s['mae'] for s in S])} | "
                 f"{ms([s['mape'] for s in S])} | {ms([s['mase'] for s in S], '{:.3f}')} | {ms([s['da_nonzero'] for s in S], '{:.1f}')} | "
                 f"— | — | {fc[best]['verdict_squared']} | {fc[best]['verdict_absolute']} | — |")
        L += [f"| *persistence* | all | {cell['val_persistence_mae']:.2f} | 1.000 | {cell['test_persistence_mae']:.2f} | — | 1.000 | — | — | — | — | — | — |", ""]
    L += ["## Where Chronos significantly beats persistence", ""]
    L += [f"- {b}" for b in beats] if beats else ["- **None.**"]
    (OUT / "chronos_table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["predict", "score"])
    ap.add_argument("--sizes", nargs="+", default=list(SIZES), choices=list(SIZES))
    ap.add_argument("--tracks", nargs="+", default=TRACKS, choices=TRACKS)
    args = ap.parse_args()
    setup_run_logging(f"chronos_{args.step}")
    cfg = load_config("configs/model_config.yaml")
    try:
        predict(args.sizes, args.tracks, cfg) if args.step == "predict" else score(cfg)
    except InsufficientMemory as e:
        logger.error(str(e))
        sys.exit(3)


if __name__ == "__main__":
    main()
