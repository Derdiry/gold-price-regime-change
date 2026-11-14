"""
Phase 4.1 report: rolling-origin folds 1-5 (seed 1), per fold and pooled.

Rules: results/rolling/PREREGISTRATION.md (must be committed).
Outputs: results/rolling/{rolling_results.json, rolling_table.md, dm_tests.csv}

    python scripts/rolling_report.py
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.fetchers import load_config  # noqa: E402
from src.evaluation.metrics import compute_all_metrics  # noqa: E402
from src.evaluation.stats import diebold_mariano, holm, mase_vs_persistence  # noqa: E402
from src.models.ensemble import apply_ensemble, fit_validation_ensemble  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

TRACKS, HORIZONS, FOLDS = ["sudan", "global"], [1, 7, 30], [1, 2, 3, 4, 5]
MODELS = ["lstm", "gru", "tsmixer", "patchtst", "tft"]
BASELINES = {"sudan": ["persistence", "drift", "arima", "ets", "vecm"], "global": ["persistence", "drift", "arima", "ets"]}
LOSSES, ALPHA = ["squared", "absolute"], 0.05
OUT = Path("results/rolling")
PREREG = OUT / "PREREGISTRATION.md"


def require_committed_preregistration():
    rel = str(PREREG)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = not subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if not (tracked and clean):
        sys.exit(f"{rel} must be committed (and unchanged) before scoring.")
    return subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def score(y, f, last, pers, h):
    m = compute_all_metrics(y, f, last)
    out = {"mae": m["mae"], "mape": m["mape"], "da_nonzero": m["directional_accuracy_nonzero"],
           "mase": mase_vs_persistence(y, f, pers), "n": m["n_windows"]}
    for loss in LOSSES:
        out[f"dm_{loss}"] = diebold_mariano(y, f, pers, h, loss)
    return out


def verdict(d, pkey="p_value"):
    if d[pkey] < ALPHA:
        return "beats persistence" if d["dm_hln"] < 0 else "worse"
    return "no significant difference"


def main():
    prereg_commit = require_committed_preregistration()
    cfg = load_config("configs/model_config.yaml")
    sel = json.loads(Path("results/selection/selection.json").read_text())["selection"]
    results = {}
    for track in TRACKS:
        for h in HORIZONS:
            folds, pooled = {}, {"y": [], "last": [], "pers": [], "f": {}}
            for k in FOLDS:
                pdir = Path(f"logs/v2/rolling/fold{k}/preds")
                b = np.load(pdir / "baselines" / f"{track}_h{h}.npz")
                yv, yt, lv, lt, pers = b["y_val"], b["y_test"], b["last_val"], b["last_test"], b["persistence_test"]
                fv, ft, meta = {}, {}, {}
                for m in MODELS:
                    c = sel[f"{m}_{track}_h{h}"]["condition"]
                    key = f"{m}_{track}_h{h}_{c}_s1"
                    z = np.load(pdir / f"{key}.npz")
                    rec = json.loads(Path(f"results/rolling/fold{k}/runs/{key}.json").read_text())
                    assert np.isclose(np.abs(yt - z["test_pred"]).mean(), rec["metrics"]["test"]["mae"], rtol=1e-6), (k, key)
                    fv[m], ft[m] = z["val_pred"], z["test_pred"]
                    meta[m] = {"condition": c, "epochs": rec["epochs_trained"], "val_mae": rec["metrics"]["val"]["mae"]}
                e = fit_validation_ensemble(fv, yv, top_n=2)
                ft["ensemble"] = apply_ensemble(ft, e)
                meta["ensemble"] = {"members": e["selected"], "weights": e["weights"]}
                for n in BASELINES[track]:
                    ft[n] = b[f"{n}_test"]
                folds[k] = {"n_test": int(len(yt)), "test_range": None, "meta": meta,
                            "scores": {n: score(yt, f, lt, pers, h) for n, f in ft.items()}}
                pooled["y"].append(yt); pooled["last"].append(lt); pooled["pers"].append(pers)
                for n, f in ft.items():
                    pooled["f"].setdefault(n, []).append(f)
            Y, Lst, P = (np.concatenate(pooled[x]) for x in ("y", "last", "pers"))
            pool = {n: score(Y, np.concatenate(fs), Lst, P, h) for n, fs in pooled["f"].items()}
            for n in pool:
                pool[n]["folds_mase_below_1"] = int(sum(folds[k]["scores"][n]["mase"] < 1 for k in FOLDS))
            results[f"{track}_h{h}"] = {"folds": folds, "pooled": pool}

    for loss in LOSSES:
        refs = [cell["pooled"][n] for cell in results.values() for n in [*MODELS, "ensemble"]]
        for r, p in zip(refs, holm([r[f"dm_{loss}"]["p_value"] for r in refs])):
            r[f"dm_{loss}"]["p_holm"] = p
    for cell in results.values():
        for n, r in cell["pooled"].items():
            for loss in LOSSES:
                r[f"verdict_{loss}"] = verdict(r[f"dm_{loss}"]) if n != "persistence" else "—"
                if "p_holm" in r[f"dm_{loss}"]:
                    r[f"verdict_{loss}_holm"] = verdict(r[f"dm_{loss}"], "p_holm")
    beats = [(k, n, r.get("verdict_absolute_holm")) for k, cell in results.items() for n, r in cell["pooled"].items()
             if r.get("verdict_absolute") == "beats persistence"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rolling_results.json").write_text(json.dumps(
        {"provenance": provenance(cfg), "preregistration_commit": prereg_commit, "preregistration_text": PREREG.read_text(),
         "seeds": [1], "beats_persistence_absolute": beats, "results": results}, indent=2, default=float))
    rows = [{"cell": key, "fold": k, "config": n, "loss": loss, "dm_hln": s[f"dm_{loss}"]["dm_hln"], "p_value": s[f"dm_{loss}"]["p_value"],
             "mase": s["mase"], "n": s["n"]}
            for key, cell in results.items() for k, f in cell["folds"].items() for n, s in f["scores"].items()
            if n != "persistence" for loss in LOSSES]
    rows += [{"cell": key, "fold": "pooled", "config": n, "loss": loss, "dm_hln": r[f"dm_{loss}"]["dm_hln"],
              "p_value": r[f"dm_{loss}"]["p_value"], "p_holm": r[f"dm_{loss}"].get("p_holm"), "mase": r["mase"], "n": r["n"]}
             for key, cell in results.items() for n, r in cell["pooled"].items() if n != "persistence" for loss in LOSSES]
    pd.DataFrame(rows).to_csv(OUT / "dm_tests.csv", index=False)
    write_markdown(results, beats, prereg_commit)


def write_markdown(results, beats, prereg_commit):
    L = ["# Rolling-origin evaluation, 5 folds, seed 1 (generated by scripts/rolling_report.py)", "",
         f"Rules fixed in `{PREREG}` (commit `{prereg_commit[:7]}`) before training. **One seed per fold.** Neural models use "
         "their Phase 3 selected condition; ensemble = top-2 on each fold's validation block. Pooled = the five test blocks "
         "concatenated. DM vs persistence (HLN, Newey–West lag h − 1); p with sign (− = lower loss than persistence). "
         "Verdict on pooled windows: p < 0.05 with the sign; Holm over the 36 pooled neural/ensemble tests per loss.", "",
         "## Pooled verdicts: beats persistence (absolute loss)", ""]
    L += [f"- {k} {n} (Holm: {hv})" for k, n, hv in beats] if beats else ["- **None.**"]
    L.append("")
    for key, cell in results.items():
        track, h = key.split("_h")
        L += [f"## {track} h={h}", "",
              "| configuration | MASE fold 1 | 2 | 3 | 4 | 5 | folds MASE<1 | pooled MAE | pooled MASE | DM p abs | DM p sq | verdict abs / sq | Holm abs / sq |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|"]
        for n, r in cell["pooled"].items():
            fm = " | ".join("%.3f" % cell["folds"][k]["scores"][n]["mase"] for k in FOLDS)
            if n == "persistence":
                L.append(f"| *persistence* | {fm} | — | {r['mae']:.2f} | 1.000 | — | — | — | — |")
                continue
            pv = {loss: "%.3f%s" % (r[f"dm_{loss}"]["p_value"], "−" if r[f"dm_{loss}"]["dm_hln"] < 0 else "+") for loss in LOSSES}
            holm_s = f"{r.get('verdict_absolute_holm', '—')} / {r.get('verdict_squared_holm', '—')}"
            label = n if n in MODELS + ["ensemble"] else f"*{n}*"
            L.append(f"| {label} | {fm} | {r['folds_mase_below_1']}/5 | {r['mae']:.2f} | {r['mase']:.3f} | {pv['absolute']} | {pv['squared']} | "
                     f"{r['verdict_absolute']} / {r['verdict_squared']} | {holm_s} |")
        L += ["", "Test windows per fold: " + ", ".join(str(cell["folds"][k]["n_test"]) for k in FOLDS), ""]
    (OUT / "rolling_table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
