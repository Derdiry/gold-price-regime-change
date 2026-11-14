"""
Phase 3 final test report: every final configuration vs persistence.

Final configurations, seeds 1-3, per (track, horizon):
  - each model under MinMax and under its validation-selected condition
  - the MinMax ensemble (top-2 by validation MAE within MinMax, from train.py)
  - the validation-selected ensemble: top-2 by validation MAE among the five
    models' selected configurations, SLSQP weights on validation
    (fitted here; saved to results/ensembles/selected_<track>_h<h>_s<seed>.json)
  - the Phase 2 baselines (deterministic)

For each forecast: test MAE, MAPE, MASE vs persistence, directional accuracy
on non-zero changes, and Diebold–Mariano (HLN-corrected, Newey–West lag h−1)
against persistence for squared and absolute loss (src/evaluation/stats.py).

Verdict per (configuration, loss), fixed before looking at the results:
  "beats persistence"  DM* < 0 and p < 0.05 in all three seeds
  "worse"              DM* > 0 and p < 0.05 in all three seeds
  "no significant difference" otherwise
The same verdicts after Holm adjustment over all seed-level neural/ensemble
tests of that loss are reported alongside.

Outputs: results/final/test_table.md, results/final/test_results.json,
results/final/dm_tests.csv

    python scripts/final_test_report.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metrics import compute_all_metrics  # noqa: E402
from src.evaluation.stats import diebold_mariano, holm, mase_vs_persistence  # noqa: E402
from src.models.ensemble import apply_ensemble, fit_validation_ensemble  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

TRACKS, HORIZONS, SEEDS = ["sudan", "global"], [1, 7, 30], [1, 2, 3]
MODELS = ["lstm", "gru", "tsmixer", "patchtst", "tft"]
BASELINES = {"sudan": ["persistence", "drift", "arima", "ets", "vecm"], "global": ["persistence", "drift", "arima", "ets"]}
LOSSES = ["squared", "absolute"]
ALPHA = 0.05
PREDS, RUNS, ENS = Path("logs/v2/preds"), Path("results/runs"), Path("results/ensembles")
OUT = Path("results/final")


def load_preds(key):
    z = np.load(PREDS / f"{key}.npz")
    if "test_pred" not in z:
        raise RuntimeError(f"{key} has no test predictions (stage 2 incomplete?)")
    return z["val_pred"], z["test_pred"]


def score(y, f, last, f_pers, h):
    m = compute_all_metrics(y, f, last)
    out = {"mae": m["mae"], "mape": m["mape"], "rmse": m["rmse"],
           "da_nonzero": m["directional_accuracy_nonzero"], "da_origin_nonzero": m["da_origin_nonzero"],
           "mase": mase_vs_persistence(y, f, f_pers), "n": m["n_windows"]}
    for loss in LOSSES:
        out[f"dm_{loss}"] = diebold_mariano(y, f, f_pers, h, loss)
    return out


def verdict(seed_scores, loss, pkey="p_value"):
    dms = [s[f"dm_{loss}"] for s in seed_scores]
    if all(d["dm_hln"] < 0 and d[pkey] < ALPHA for d in dms):
        return "beats persistence"
    if all(d["dm_hln"] > 0 and d[pkey] < ALPHA for d in dms):
        return "worse"
    return "no significant difference"


def main():
    sel = json.loads(Path("results/selection/selection.json").read_text())["selection"]
    OUT.mkdir(parents=True, exist_ok=True)
    results, dm_rows = {}, []
    for track in TRACKS:
        for h in HORIZONS:
            b = np.load(PREDS / "baselines" / f"{track}_h{h}.npz")
            yv, yt, lv, lt = b["y_val"], b["y_test"], b["last_val"], b["last_test"]
            pers_v, pers_t = b["persistence_val"], b["persistence_test"]
            cell = {"configs": {}, "baselines": {}, "val_persistence_mae": float(np.abs(yv - pers_v).mean())}
            sel_val, sel_test = {s: {} for s in SEEDS}, {s: {} for s in SEEDS}
            for model in MODELS:
                for cond_label, cond in (("minmax", "minmax"), ("selected", sel[f"{model}_{track}_h{h}"]["condition"])):
                    seeds = []
                    for s in SEEDS:
                        key = f"{model}_{track}_h{h}_{cond}_s{s}"
                        vp, tp = load_preds(key)
                        rec = json.loads((RUNS / f"{key}.json").read_text())
                        assert np.isclose(np.abs(yt - tp).mean(), rec["metrics"]["test"]["mae"], rtol=1e-6), key
                        sc = score(yt, tp, lt, pers_t, h)
                        sc["val_mae"] = rec["metrics"]["val"]["mae"]
                        sc["epochs"] = rec["epochs_trained"]
                        sc["key"] = key
                        seeds.append(sc)
                        if cond_label == "selected":
                            sel_val[s][model], sel_test[s][model] = vp, tp
                    cell["configs"][f"{model}|{cond_label}"] = {"model": model, "condition": cond, "label": cond_label, "seeds": seeds}
                    if cond_label == "selected" and cond == "minmax":
                        pass  # never happens in this selection (no model selected MinMax)
            # Ensembles
            for label in ("ensemble|minmax", "ensemble|selected"):
                seeds = []
                for s in SEEDS:
                    if label == "ensemble|minmax":
                        e = json.loads((ENS / f"{track}_h{h}_minmax_s{s}.json").read_text())
                        tests = {m: load_preds(f"{m}_{track}_h{h}_minmax_s{s}")[1] for m in e["selected"]}
                        ft = apply_ensemble(tests, e)
                        assert np.isclose(np.abs(yt - ft).mean(), e["metrics"]["test"]["mae"], rtol=1e-6)
                        val_mae = e["metrics"]["val"]["mae"]
                    else:
                        e = fit_validation_ensemble(sel_val[s], yv, top_n=2)
                        ft = apply_ensemble(sel_test[s], e)
                        fv = apply_ensemble(sel_val[s], e)
                        val_mae = float(np.abs(yv - fv).mean())
                        rec = {"key": f"ensemble_selected_{track}_h{h}_s{s}", "protocol": "v2", "track": track,
                               "horizon": h, "seed": s, **e,
                               "member_records": [f"{m}_{track}_h{h}_{sel[f'{m}_{track}_h{h}']['condition']}_s{s}" for m in e["selected"]],
                               "metrics": {"val": compute_all_metrics(yv, fv, lv), "test": compute_all_metrics(yt, ft, lt)},
                               "provenance": provenance({"mlflow": {"protocol": "v2"}})}
                        (ENS / f"selected_{track}_h{h}_s{s}.json").write_text(json.dumps(rec, indent=2))
                    sc = score(yt, ft, lt, pers_t, h)
                    sc.update({"val_mae": val_mae, "members": e["selected"], "weights": e["weights"]})
                    seeds.append(sc)
                cell["configs"][label] = {"model": "ensemble", "condition": label.split("|")[1],
                                          "label": label.split("|")[1], "seeds": seeds}
            for name in BASELINES[track]:
                cell["baselines"][name] = score(yt, b[f"{name}_test"], lt, pers_t, h)
            results[f"{track}_h{h}"] = cell

    # Holm adjustment over all seed-level neural/ensemble tests, per loss
    for loss in LOSSES:
        refs = [(c, i) for cell in results.values() for c in cell["configs"].values() for i in range(len(SEEDS))]
        adj = holm([c["seeds"][i][f"dm_{loss}"]["p_value"] for c, i in refs])
        for (c, i), p in zip(refs, adj):
            c["seeds"][i][f"dm_{loss}"]["p_holm"] = p
    for cell_key, cell in results.items():
        for name, c in cell["configs"].items():
            for loss in LOSSES:
                c[f"verdict_{loss}"] = verdict(c["seeds"], loss)
                c[f"verdict_{loss}_holm"] = verdict(c["seeds"], loss, "p_holm")
            for s, sc in zip(SEEDS, c["seeds"]):
                for loss in LOSSES:
                    d = sc[f"dm_{loss}"]
                    dm_rows.append({"cell": cell_key, "config": name, "seed": s, "loss": loss,
                                    "dm_hln": d["dm_hln"], "p_value": d["p_value"], "p_holm": d["p_holm"],
                                    "mean_loss_diff": d["mean_loss_diff"], "n": d["n"], "hac_lag": d["hac_lag"],
                                    "variance_fallback": d["variance_fallback"]})

    (OUT / "test_results.json").write_text(json.dumps(
        {"provenance": provenance({"mlflow": {"protocol": "v2"}}), "alpha": ALPHA,
         "verdict_rule": "beats persistence = DM* < 0 and p < 0.05 in all three seeds (worse: DM* > 0 likewise)",
         "results": results}, indent=2, default=float))
    pd.DataFrame(dm_rows).to_csv(OUT / "dm_tests.csv", index=False)
    write_markdown(results)


def ms(vals, fmt="{:.2f}"):
    v = np.asarray(vals, float)
    return f"{fmt.format(v.mean())} ± {fmt.format(v.std(ddof=1))}"


def write_markdown(results):
    L = ["# Phase 3 final test results (generated by scripts/final_test_report.py)", "",
         "Test windows identical to those of the original MinMax setup and of the Phase 2 baselines. Neural models and ensembles: "
         "mean ± std over seeds 1–3 (sample std). MASE = MAE / MAE(persistence) on the same windows "
         "(< 1 beats persistence). DM = Diebold–Mariano vs persistence, HLN small-sample correction, "
         "Newey–West variance with lag h − 1, two-sided p for seeds 1/2/3. Verdict: \"beats\" or \"worse\" "
         "only if the sign agrees and p < 0.05 in all three seeds; Holm = the same after Holm adjustment over "
         "all seed-level neural/ensemble tests of that loss.", "",
         "Conditions: `minmax` = MinMax scaling fitted on the training rows (the original setup); `selected` = each model's validation-selected transform "
         "(results/selection/selection.json).", ""]
    summary_beats = []
    for cell_key, cell in results.items():
        track, h = cell_key.split("_h")
        L += [f"## {track} h={h}", "",
              "| configuration | condition | MAE | MAPE % | MASE | DA non-zero % | DM p squared (s1/s2/s3) | DM p absolute (s1/s2/s3) | verdict squared | verdict absolute | verdict (Holm) sq / abs |",
              "|---|---|---:|---:|---:|---:|---|---|---|---|---|"]
        for name, c in cell["configs"].items():
            S = c["seeds"]
            label = name.split("|")[0]
            cond = c["condition"] if label != "ensemble" else ("MinMax members" if c["condition"] == "minmax" else "selected members")
            pv = {loss: "/".join("%.3f%s" % (s[f"dm_{loss}"]["p_value"], "−" if s[f"dm_{loss}"]["dm_hln"] < 0 else "+")
                                 for s in S) for loss in LOSSES}
            L.append(f"| {label} | {cond} | {ms([s['mae'] for s in S])} | {ms([s['mape'] for s in S])} | "
                     f"{ms([s['mase'] for s in S], '{:.3f}')} | {ms([s['da_nonzero'] for s in S], '{:.1f}')} | "
                     f"{pv['squared']} | {pv['absolute']} | {c['verdict_squared']} | {c['verdict_absolute']} | "
                     f"{c['verdict_squared_holm']} / {c['verdict_absolute_holm']} |")
            for loss in LOSSES:
                if c[f"verdict_{loss}"] == "beats persistence":
                    summary_beats.append(f"{track} h={h} {label} ({cond}, {loss} loss{', survives Holm' if c[f'verdict_{loss}_holm'] == 'beats persistence' else ', not after Holm'})")
        for name, sc in cell["baselines"].items():
            pv = {loss: ("%.3f%s" % (sc[f"dm_{loss}"]["p_value"], "−" if sc[f"dm_{loss}"]["dm_hln"] < 0 else "+")
                         if name != "persistence" else "—") for loss in LOSSES}
            L.append(f"| *{name}* | baseline | {sc['mae']:.2f} | {sc['mape']:.2f} | {sc['mase']:.3f} | "
                     f"{sc['da_nonzero']:.1f} | {pv['squared']} | {pv['absolute']} | — | — | — |")
        L += ["", "p-value suffix: − = lower loss than persistence, + = higher.", ""]
    L += ["## Where a model significantly beats persistence on test", ""]
    L += [f"- {s}" for s in summary_beats] if summary_beats else ["- **None.** No neural model or ensemble beats persistence with p < 0.05 in all three seeds, for either loss."]
    L += ["", "## Sudan h=30 (validation advantage check)", "",
          f"Persistence validation MAE: {results['sudan_h30']['val_persistence_mae']:.2f}. "
          f"Persistence test MAE: {results['sudan_h30']['baselines']['persistence']['mae']:.2f}.", "",
          "| configuration | condition | val MAE s1 / s2 / s3 | beats persistence on val (seeds) | test MAE s1 / s2 / s3 | beats persistence on test (seeds) | verdict sq / abs |",
          "|---|---|---|---:|---|---:|---|"]
    cell = results["sudan_h30"]
    pv_val, pv_test = cell["val_persistence_mae"], cell["baselines"]["persistence"]["mae"]
    for name, c in cell["configs"].items():
        S = c["seeds"]
        vals = " / ".join("%.1f" % s["val_mae"] for s in S)
        tests = " / ".join("%.1f" % s["mae"] for s in S)
        L.append(f"| {name.split('|')[0]} | {c['condition']} | {vals} | "
                 f"{sum(s['val_mae'] < pv_val for s in S)}/3 | {tests} | "
                 f"{sum(s['mae'] < pv_test for s in S)}/3 | {c['verdict_squared']} / {c['verdict_absolute']} |")
    (OUT / "test_table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
