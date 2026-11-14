"""
Phase 4.4 report: regime dummies, with vs without, on rolling folds 4-5
(family F) and the main split (family M); TFT variable importance.

Rules: results/regime/PREREGISTRATION.md (must be committed).
Outputs: results/regime/{regime_results.json, regime_table.md, dm_tests.csv}

    python scripts/regime_report.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_common import (HORIZONS, LOSSES, ROOT, SEEDS, add_holm, baseline, load_run, ms, paired,  # noqa: E402
                           pfmt, require_committed, score, verdict)

sys.path.insert(0, str(ROOT))
from src.data.fetchers import load_config  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

MODELS, REGIME = ["tft", "lstm"], ["regime_1", "regime_2"]
OUT = ROOT / "results/regime"
PREREG = OUT / "PREREGISTRATION.md"
WORDS = ("regime dummies help", "regime dummies hurt")
# Real (not forward-filled, not in the gap) regime_2 = 1 training rows, from the preregistration
REAL_REGIME2_TRAIN = {"fold4": 21, "fold5": 50, "main": 0}


def places(where):
    """(with runs, with preds, without runs, without preds, importance dir, baseline preds) for 'main' or 'fold<k>'."""
    if where == "main":
        w = (ROOT / "results/runs", ROOT / "logs/v2/preds", ROOT / "results/tft_importance")
        base = ROOT / "logs/v2/preds"
    else:
        w = (ROOT / f"results/rolling/{where}/runs", ROOT / f"logs/v2/rolling/{where}/preds",
             ROOT / f"results/rolling/{where}/tft_importance")
        base = ROOT / f"logs/v2/rolling/{where}/preds"
    wo = (ROOT / f"results/regime/{where}/no_regime/runs", ROOT / f"logs/v2/regime/{where}/no_regime/preds")
    return w[0], w[1], wo[0], wo[1], w[2], base


def importance(imp_dir, key):
    d = json.loads((imp_dir / f"{key}.json").read_text())["importance"]
    ranked = sorted(d, key=d.get, reverse=True)
    return {"regime": {c: {"weight": d[c], "rank": ranked.index(c) + 1} for c in REGIME},
            "n_features": len(d), "top5": [(c, d[c]) for c in ranked[:5]]}


def main():
    prereg_commit = require_committed(PREREG)
    cfg = load_config(str(ROOT / "configs/model_config.yaml"))
    sel = json.loads((ROOT / "results/selection/selection.json").read_text())["selection"]
    families = {"F": ["fold4", "fold5"], "M": ["main"]}
    cells = {}
    for fam, wheres in families.items():
        for where in wheres:
            wr, wp, nr, np_, imp_dir, base = places(where)
            for m in MODELS:
                for h in HORIZONS:
                    c = sel[f"{m}_sudan_h{h}"]["condition"]
                    ref = baseline(base, h)
                    seeds = []
                    for s in SEEDS:
                        key = f"{m}_sudan_h{h}_{c}_s{s}"
                        a, b = load_run(wr, wp, key, ref), load_run(nr, np_, key, ref)
                        assert not set(REGIME) & set(b["rec"]["feature_cols"]), f"{key}: regime inputs in the ablation"
                        assert set(REGIME) <= set(a["rec"]["feature_cols"]), f"{key}: regime inputs missing"
                        sc = {"with": score(ref, a["f"], h), "without": score(ref, b["f"], h),
                              "with_vs_without": paired(ref, a["f"], b["f"], h),
                              "with_commit": a["rec"]["provenance"]["git_commit"], "without_commit": b["rec"]["provenance"]["git_commit"]}
                        if m == "tft":
                            sc["importance_with"] = importance(imp_dir, key)
                        seeds.append(sc)
                    cells[(fam, where, m, h)] = {"family": fam, "where": where, "model": m, "h": h, "condition": c, "seeds": seeds}
    for fam in families:
        for loss in LOSSES:
            add_holm([sc["with_vs_without"][loss] for cell in cells.values() if cell["family"] == fam for sc in cell["seeds"]])
    for cell in cells.values():
        for loss in LOSSES:
            dms = [sc["with_vs_without"][loss] for sc in cell["seeds"]]
            cell[f"verdict_{loss}"] = verdict(dms, "p_value", WORDS)
            cell[f"verdict_{loss}_holm"] = verdict(dms, "p_holm", WORDS)
            for v in ("with", "without"):
                cell[f"{v}_vs_persistence_{loss}"] = verdict([sc[v]["vs_persistence"][loss] for sc in cell["seeds"]], "p_value",
                                                             ("beats persistence", "worse"))
    summary = {}
    for fam in families:
        fc = [c for c in cells.values() if c["family"] == fam]
        summary[fam] = {f"{loss}{suf}": {w: sum(c[f"verdict_{loss}{suf}"] == w for c in fc) for w in (*WORDS, "no consistent difference")}
                        for loss in LOSSES for suf in ("", "_holm")}
        summary[fam]["n_cells"] = len(fc)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "regime_results.json").write_text(json.dumps(
        {"provenance": provenance(cfg), "preregistration_commit": prereg_commit, "preregistration_text": PREREG.read_text(),
         "real_regime2_train_rows": REAL_REGIME2_TRAIN, "summary": summary,
         "cells": [{k: v for k, v in c.items()} for c in cells.values()]}, indent=2, default=float))
    rows = [{"family": c["family"], "where": c["where"], "model": c["model"], "h": c["h"], "seed": s, "loss": loss,
             "dm_hln": sc["with_vs_without"][loss]["dm_hln"], "p_value": sc["with_vs_without"][loss]["p_value"],
             "p_holm": sc["with_vs_without"][loss]["p_holm"], "n": sc["with_vs_without"][loss]["n"]}
            for c in cells.values() for s, sc in zip(SEEDS, c["seeds"]) for loss in LOSSES]
    pd.DataFrame(rows).to_csv(OUT / "dm_tests.csv", index=False)
    write_markdown(cells, summary, prereg_commit)


def write_markdown(cells, summary, prereg_commit):
    L = ["# Regime dummies: with vs without (generated by scripts/regime_report.py)", "",
         f"Rules fixed in `results/regime/PREREGISTRATION.md` (commit `{prereg_commit[:7]}`) before training. "
         "TFT and LSTM under their Phase 3 selected conditions, seeds 1–3 paired. DM of with vs without (HLN, Newey–West "
         "lag h − 1); p per seed with sign (− = with regime dummies has lower loss). Verdict needs the sign to agree and "
         "p < 0.05 in all three seeds; Holm over the seed-level tests of each family.", "",
         "**Power.** Real (not forward-filled) `regime_2` = 1 training rows: fold 4 **21**, fold 5 **50**, main split **0** "
         "(on the main split `regime_2` is constant in training and validation and switches on at test). "
         "A null result on folds 4–5 is low-power, not evidence that the regime has no effect.", "",
         "## Summary (cells: help / hurt / no consistent difference)", "",
         "| family | cells | absolute | absolute, Holm | squared | squared, Holm |", "|---|---:|---|---|---|---|"]
    for fam, name in (("F", "F: rolling folds 4–5"), ("M", "M: main split")):
        s = summary[fam]
        fmt = lambda k: " / ".join(str(s[k][w]) for w in (*WORDS, "no consistent difference"))  # noqa: E731
        L.append(f"| {name} | {s['n_cells']} | {fmt('absolute')} | {fmt('absolute_holm')} | {fmt('squared')} | {fmt('squared_holm')} |")
    L += ["", "## Cells", "",
          "| family | split | model | h | cond. | MAE with | MAE without | MASE with | MASE without | DM p abs (s1/s2/s3) | verdict abs / Holm | DM p sq | verdict sq / Holm | vs persistence abs: with / without |",
          "|---|---|---|---:|---|---:|---:|---:|---:|---|---|---|---|---|"]
    for c in cells.values():
        g = lambda v, k: [sc[v][k] for sc in c["seeds"]]  # noqa: E731
        d = lambda loss: [sc["with_vs_without"][loss] for sc in c["seeds"]]  # noqa: E731
        L.append(f"| {c['family']} | {c['where']} | {c['model']} | {c['h']} | {c['condition']} | {ms(g('with', 'mae'))} | "
                 f"{ms(g('without', 'mae'))} | {ms(g('with', 'mase'), '{:.3f}')} | {ms(g('without', 'mase'), '{:.3f}')} | "
                 f"{pfmt(d('absolute'))} | {c['verdict_absolute']} / {c['verdict_absolute_holm']} | {pfmt(d('squared'))} | "
                 f"{c['verdict_squared']} / {c['verdict_squared_holm']} | "
                 f"{c['with_vs_persistence_absolute']} / {c['without_vs_persistence_absolute']} |")
    L += ["", "## TFT variable importance (with-regime runs, validation windows; descriptive)", "",
          "| split | h | regime_1 weight (rank / n) | regime_2 weight (rank / n) | top 5 inputs (seed 1) |", "|---|---:|---|---|---|"]
    for c in cells.values():
        if c["model"] != "tft":
            continue
        imp = [sc["importance_with"] for sc in c["seeds"]]
        n = imp[0]["n_features"]
        reg = {r: f"{ms([i['regime'][r]['weight'] for i in imp], '{:.4f}')} ({'/'.join(str(i['regime'][r]['rank']) for i in imp)} of {n})"
               for r in REGIME}
        L.append(f"| {c['where']} | {c['h']} | {reg['regime_1']} | {reg['regime_2']} | "
                 + ", ".join(f"{k} {v:.3f}" for k, v in imp[0]["top5"]) + " |")
    L += ["", "Ranks are per seed (s1/s2/s3). Uniform weight would be 1/33 ≈ 0.030."]
    (OUT / "regime_table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
