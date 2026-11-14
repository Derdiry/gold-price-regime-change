"""
Phase 4 step 0 (post-hoc, no retraining): Sudan test set restricted to target
days that are real observations (is_ffilled == 0).

Rules, subsets and the decision rule for the hypothesis are fixed in
results/posthoc_observed/PREREGISTRATION.md, which must be committed before
this script runs. The saved Phase 3 test forecasts are scored on:

  full  every (window, step) pair (reproduces results/final/test_results.json)
  S1    only the (window, step) pairs whose target day is observed
  S2    h > 1: windows whose day t+h is observed, scored on step h only

Outputs: results/posthoc_observed/{observed_days.md, observed_days.json, dm_tests.csv}

    python scripts/posthoc_observed_days.py
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import make_dataset  # noqa: E402
from src.data.features import SUDAN_FEATURE_COLS  # noqa: E402
from src.data.fetchers import load_config  # noqa: E402
from src.data.pipeline import build_sudan  # noqa: E402
from src.evaluation.stats import diebold_mariano, holm, mase_vs_persistence  # noqa: E402
from src.models.ensemble import apply_ensemble  # noqa: E402
from src.utils.provenance import REPO_ROOT, provenance  # noqa: E402

logging.disable(logging.WARNING)
HORIZONS, SEEDS = [1, 7, 30], [1, 2, 3]
MODELS = ["lstm", "gru", "tsmixer", "patchtst", "tft"]
BASELINES = ["persistence", "drift", "arima", "ets", "vecm"]
LOSSES = ["squared", "absolute"]
ALPHA = 0.05
PREDS, ENS = Path("logs/v2/preds"), Path("results/ensembles")
OUT = Path("results/posthoc_observed")
PREREG = OUT / "PREREGISTRATION.md"


def require_committed_preregistration():
    rel = str(PREREG)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO_ROOT, capture_output=True).returncode == 0
    clean = not subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=REPO_ROOT,
                               capture_output=True, text=True).stdout.strip()
    if not (tracked and clean):
        sys.exit(f"{rel} must be committed (and unchanged) before scoring.")
    commit = subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout.strip()
    return commit


def observed_masks(cfg):
    """Per horizon: (n_test, h) bool, True where the target day is a real observation."""
    df = build_sudan(cfg)
    tc = cfg["data"]["target_col"]
    ff = df["is_ffilled"].to_numpy().astype(bool)
    target = df[tc].to_numpy(float)
    out = {}
    for h in HORIZONS:
        sp, _, _, _ = make_dataset(df, cfg, "sudan", h, SUDAN_FEATURE_COLS, tc, save=False)
        pos = df.index.get_indexer(pd.to_datetime(sp["origin_test"]))
        out[h] = {"obs": np.stack([~ff[pos + 1 + k] for k in range(h)], 1),
                  "y": np.stack([target[pos + 1 + k] for k in range(h)], 1)}
    return out


def score(y, f, last, f_pers, h, mask):
    m = mask.astype(bool)
    e = (y - f)[m]
    yt = y[m]
    ch_true = (y - last[:, None])[m]
    ch_pred = (f - last[:, None])[m]
    nz = ch_true != 0
    out = {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
           "mape": float(100 * np.mean(np.abs(e) / np.abs(yt))),
           "da_origin_nonzero": float(100 * np.mean(np.sign(ch_true[nz]) == np.sign(ch_pred[nz]))) if nz.any() else float("nan"),
           "n_nonzero_changes": int(nz.sum()),
           "mase": mase_vs_persistence(y, f, f_pers, m),
           "n_windows": int(m.any(1).sum()), "n_pairs": int(m.sum())}
    for loss in LOSSES:
        out[f"dm_{loss}"] = diebold_mariano(y, f, f_pers, h, loss, mask=m)
    return out


def subsets(h, obs):
    full = np.ones_like(obs, bool)
    s = {"full": full, "S1": obs}
    if h > 1:
        s2 = np.zeros_like(obs, bool)
        s2[:, -1] = obs[:, -1]
        s["S2"] = s2
    return s


def verdict(seed_scores, loss, pkey="p_value"):
    dms = [s[f"dm_{loss}"] for s in seed_scores]
    if all(d["dm_hln"] < 0 and d[pkey] < ALPHA for d in dms):
        return "beats persistence"
    if all(d["dm_hln"] > 0 and d[pkey] < ALPHA for d in dms):
        return "worse"
    return "no significant difference"


def base_verdict(d):
    if not d["p_value"] < ALPHA:
        return "no significant difference"
    return "beats persistence" if d["dm_hln"] < 0 else "worse"


def main():
    prereg_commit = require_committed_preregistration()
    cfg = load_config("configs/model_config.yaml")
    sel = json.loads(Path("results/selection/selection.json").read_text())["selection"]
    phase3 = json.loads(Path("results/final/test_results.json").read_text())["results"]
    masks = observed_masks(cfg)
    results = {}
    for h in HORIZONS:
        b = np.load(PREDS / "baselines" / f"sudan_h{h}.npz")
        yt, lt, pers = b["y_test"], b["last_test"], b["persistence_test"]
        assert np.allclose(yt, masks[h]["y"]), "baseline windows differ from the rebuilt test windows"
        forecasts = {}
        for model in MODELS:
            for label, cond in (("minmax", "minmax"), ("selected", sel[f"{model}_sudan_h{h}"]["condition"])):
                forecasts[f"{model}|{label}"] = (cond, [np.load(PREDS / f"{model}_sudan_h{h}_{cond}_s{s}.npz")["test_pred"] for s in SEEDS])
        for label in ("minmax", "selected"):
            fs = []
            for s in SEEDS:
                e = json.loads((ENS / (f"sudan_h{h}_minmax_s{s}.json" if label == "minmax" else f"selected_sudan_h{h}_s{s}.json")).read_text())
                fs.append(apply_ensemble({rec.split("_sudan_")[0]: np.load(PREDS / f"{rec}.npz")["test_pred"]
                                          for rec in e["member_records"]}, e))
            forecasts[f"ensemble|{label}"] = (label, fs)
        cell = {}
        for sub, m in subsets(h, masks[h]["obs"]).items():
            c = {"configs": {}, "baselines": {}}
            for name, (cond, fs) in forecasts.items():
                seeds = [score(yt, f, lt, pers, h, m) for f in fs]
                if sub == "full":   # must reproduce the Phase 3 report
                    ref = phase3[f"sudan_h{h}"]["configs"][name]["seeds"]
                    assert all(np.isclose(a["mae"], r["mae"]) and np.isclose(a["dm_absolute"]["p_value"], r["dm_absolute"]["p_value"])
                               for a, r in zip(seeds, ref)), name
                c["configs"][name] = {"condition": cond, "seeds": seeds}
            for name in BASELINES:
                c["baselines"][name] = score(yt, b[f"{name}_test"], lt, pers, h, m)
            cell[sub] = c
        results[h] = cell

    # Verdicts, with Holm over all seed-level neural/ensemble tests of a loss within a subset
    for sub in ("full", "S1", "S2"):
        cells = [results[h][sub] for h in HORIZONS if sub in results[h]]
        for loss in LOSSES:
            refs = [(c, i) for cell in cells for c in cell["configs"].values() for i in range(len(SEEDS))]
            for (c, i), p in zip(refs, holm([c["seeds"][i][f"dm_{loss}"]["p_value"] for c, i in refs])):
                c["seeds"][i][f"dm_{loss}"]["p_holm"] = p
        for cell in cells:
            for c in cell["configs"].values():
                for loss in LOSSES:
                    c[f"verdict_{loss}"] = verdict(c["seeds"], loss)
                    c[f"verdict_{loss}_holm"] = verdict(c["seeds"], loss, "p_holm")
                c["mase_mean"] = float(np.mean([s["mase"] for s in c["seeds"]]))
            for name, sc in cell["baselines"].items():
                for loss in LOSSES:
                    sc[f"verdict_{loss}"] = base_verdict(sc[f"dm_{loss}"]) if name != "persistence" else "—"

    # Preregistered decision on H
    sel_cfgs = [(h, n) for h in HORIZONS for n in [f"{m}|selected" for m in MODELS] + ["ensemble|selected"]]
    worse_full = sum(results[h]["full"]["configs"][n]["verdict_absolute"] == "worse" for h, n in sel_cfgs)
    worse_s1 = sum(results[h]["S1"]["configs"][n]["verdict_absolute"] == "worse" for h, n in sel_cfgs)
    med_mase = float(np.median([results[h]["S1"]["configs"][n]["mase_mean"] for h, n in sel_cfgs]))
    if worse_s1 <= 6 and med_mase <= 1.0:
        decision = "supported"
    elif worse_s1 >= 12:
        decision = "not supported"
    else:
        decision = "partly supported"
    decision_rec = {"n_configs": len(sel_cfgs), "worse_absolute_full": worse_full, "worse_absolute_S1": worse_s1,
                    "median_seed_mean_mase_S1": med_mase, "decision": decision}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "observed_days.json").write_text(json.dumps(
        {"provenance": provenance(cfg), "preregistration": str(PREREG), "preregistration_commit": prereg_commit,
         "preregistration_text": PREREG.read_text(), "alpha": ALPHA, "decision_H": decision_rec,
         "results": {f"sudan_h{h}": v for h, v in results.items()}}, indent=2, default=float))
    rows = []
    for h in HORIZONS:
        for sub, cell in results[h].items():
            for name, c in cell["configs"].items():
                for s, sc in zip(SEEDS, c["seeds"]):
                    for loss in LOSSES:
                        d = sc[f"dm_{loss}"]
                        rows.append({"h": h, "subset": sub, "config": name, "seed": s, "loss": loss, "dm_hln": d["dm_hln"],
                                     "p_value": d["p_value"], "p_holm": d["p_holm"], "n": d["n"], "mean_loss_diff": d["mean_loss_diff"]})
            for name, sc in cell["baselines"].items():
                if name == "persistence":
                    continue
                for loss in LOSSES:
                    d = sc[f"dm_{loss}"]
                    rows.append({"h": h, "subset": sub, "config": name, "seed": "", "loss": loss, "dm_hln": d["dm_hln"],
                                 "p_value": d["p_value"], "p_holm": "", "n": d["n"], "mean_loss_diff": d["mean_loss_diff"]})
    pd.DataFrame(rows).to_csv(OUT / "dm_tests.csv", index=False)
    write_markdown(results, decision_rec, prereg_commit)


def ms(vals, fmt="{:.2f}"):
    v = np.asarray(vals, float)
    return f"{fmt.format(v.mean())} ± {fmt.format(v.std(ddof=1))}"


SUB_LABEL = {"full": "all test pairs (Phase 3, reproduced)", "S1": "S1: target day observed",
             "S2": "S2: day t+h observed, step h only"}


def write_markdown(results, dec, prereg_commit):
    L = ["# Sudan test restricted to observed days (generated by scripts/posthoc_observed_days.py)", "",
         f"Post-hoc diagnostic on the already-used test set; no retraining or re-selection. Rules fixed in "
         f"`{PREREG}` (commit `{prereg_commit[:7]}`) before these numbers were computed.", "",
         "Neural/ensembles: mean ± sample std over seeds 1–3. MASE = MAE / MAE(persistence) on the same pairs. "
         "DM vs persistence: HLN, Newey–West lag h − 1, p for seeds 1/2/3 with sign (− lower loss than persistence). "
         "Verdict: sign agrees and p < 0.05 in all three seeds. DA = direction vs last observed value, non-zero true changes only.", "",
         "All h target days observed: 0 windows at h = 7 and h = 30 (not reportable); identical to S1 at h = 1.", "",
         "## Decision on H (preregistered: S1, absolute loss, 18 selected configurations)", "",
         f"- \"worse\" than persistence: {dec['worse_absolute_full']}/18 on the full test set → **{dec['worse_absolute_S1']}/18 on S1**",
         f"- median seed-mean MASE on S1: **{dec['median_seed_mean_mase_S1']:.3f}**",
         f"- Rule: supported if ≤ 6 worse and median MASE ≤ 1.00; not supported if ≥ 12 worse; partly otherwise.",
         f"- **Decision: H {dec['decision']}.**", ""]
    for h in HORIZONS:
        for sub, cell in results[h].items():
            pers = cell["baselines"]["persistence"]
            L += [f"## h={h} · {SUB_LABEL[sub]}", "",
                  f"Windows: {pers['n_windows']} · scored pairs: {pers['n_pairs']} · non-zero true changes: {pers['n_nonzero_changes']} · "
                  f"persistence MAE {pers['mae']:.2f}", "",
                  "| configuration | condition | MAE | MAPE % | MASE | MASE full | DA non-zero % | DM p abs (s1/s2/s3) | DM p sq (s1/s2/s3) | verdict abs | verdict sq | Holm abs / sq |",
                  "|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|"]
            for name, c in cell["configs"].items():
                S = c["seeds"]
                full_mase = results[h]["full"]["configs"][name]["mase_mean"]
                pv = {loss: "/".join("%.3f%s" % (s[f"dm_{loss}"]["p_value"], "−" if s[f"dm_{loss}"]["dm_hln"] < 0 else "+") for s in S)
                      for loss in LOSSES}
                L.append(f"| {name.split('|')[0]} | {c['condition']} | {ms([s['mae'] for s in S])} | {ms([s['mape'] for s in S])} | "
                         f"{ms([s['mase'] for s in S], '{:.3f}')} | {full_mase:.3f} | {ms([s['da_origin_nonzero'] for s in S], '{:.1f}')} | "
                         f"{pv['absolute']} | {pv['squared']} | {c['verdict_absolute']} | {c['verdict_squared']} | "
                         f"{c['verdict_absolute_holm']} / {c['verdict_squared_holm']} |")
            for name, sc in cell["baselines"].items():
                pv = {loss: ("%.3f%s" % (sc[f"dm_{loss}"]["p_value"], "−" if sc[f"dm_{loss}"]["dm_hln"] < 0 else "+")
                             if name != "persistence" else "—") for loss in LOSSES}
                full_mase = results[h]["full"]["baselines"][name]["mase"]
                L.append(f"| *{name}* | baseline | {sc['mae']:.2f} | {sc['mape']:.2f} | {sc['mase']:.3f} | {full_mase:.3f} | "
                         f"{sc['da_origin_nonzero']:.1f} | {pv['absolute']} | {pv['squared']} | {sc['verdict_absolute']} | {sc['verdict_squared']} | — |")
            L.append("")
    (OUT / "observed_days.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:14]))


if __name__ == "__main__":
    main()
