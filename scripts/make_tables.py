"""
Regenerate every results table from saved files, then the cross-experiment
headline summary. Nothing is trained or refitted: each step reads saved run
records (results/), saved forecasts (logs/v2/, local only) and the data files.

Steps, in dependency order, each a report-only mode of an existing script:
  data splits, baseline summaries (main, duplicate-rule sensitivity, folds 1-5),
  audit persistence check, Phase 3 selection and final test report,
  duplicate-rule comparisons, observed-days diagnostic, Chronos scoring,
  conformal intervals, FX target (selection, report), rolling-origin report
  and fold-1 check, regime, transfer, ACLED (lag-selection check, report),
  ACLED cross-check (only when the raw ACLED files are present).
Then writes results/summary/headline.{md,json}: for every experiment, how
many forecast configurations beat persistence under the Diebold–Mariano rule
(preregistered for the Phase 4 experiments; fixed in committed scripts for Phase 3 and the baselines).

    python scripts/make_tables.py [--from STEP]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paired_common import ALPHA  # noqa: E402

PY = sys.executable
FOLDS = [1, 2, 3, 4, 5]
ACLED_RAW = ROOT / "data/raw/sudan"

STEPS = [
    ("splits", ["scripts/report_splits.py"]),
    ("baselines", ["scripts/run_baselines.py", "--summary-only"]),
    ("baselines_last", ["scripts/run_baselines.py", "--summary-only", "--duplicate-rule", "last"]),
    *[(f"baselines_fold{k}", ["scripts/run_baselines.py", "--summary-only", "--fold", str(k)]) for k in FOLDS],
    ("baselines_last_compare", ["scripts/compare_duplicate_sensitivity.py"]),
    ("audit_persistence", ["scripts/check_audit_persistence.py"]),
    ("selection", ["scripts/select_transforms.py"]),
    ("final_test", ["scripts/final_test_report.py"]),
    ("duplicate_neural", ["scripts/compare_duplicate_neural.py"]),
    ("observed_days", ["scripts/posthoc_observed_days.py"]),
    ("chronos", ["scripts/run_chronos.py", "score"]),
    ("conformal", ["scripts/conformal_intervals.py"]),
    ("fx_select", ["scripts/fx_target_report.py", "select"]),
    ("fx_report", ["scripts/fx_target_report.py", "report"]),
    ("rolling", ["scripts/rolling_report.py"]),
    ("rolling_fold1", ["scripts/rolling_fold1_check.py"]),
    ("regime", ["scripts/regime_report.py"]),
    ("transfer", ["scripts/transfer_report.py"]),
    ("acled_check", ["scripts/acled_report.py", "check"]),
    ("acled", ["scripts/acled_report.py", "report"]),
    ("acled_crosscheck", ["scripts/acled_crosscheck.py"]),
    ("facts", ["scripts/project_facts.py"]),
]


def run_steps(start):
    names = [n for n, _ in STEPS]
    for name, cmd in STEPS[names.index(start):]:
        if name == "acled_crosscheck" and not any(ACLED_RAW.glob("*.xlsx")) and not any(ACLED_RAW.glob("*.csv")):
            print(f"SKIP {name}: raw ACLED files are not in data/raw/sudan/ (ACLED terms; results/acled/crosscheck.* kept)")
            continue
        t0 = time.time()
        code = subprocess.run([PY, *cmd], cwd=ROOT, stdout=subprocess.DEVNULL).returncode
        print(f"{'OK  ' if code == 0 else 'FAIL'} {name} ({time.time() - t0:.0f} s)", flush=True)
        if code:
            sys.exit(f"step {name} failed (exit {code}); fix it and resume with --from {name}")


def beats(dms, pkey="p_value"):
    """Verdict rule (preregistered for Phase 4; fixed in committed scripts for Phase 3): DM* < 0 and p < alpha in every seed given."""
    return all(d["dm_hln"] < 0 and d[pkey] < ALPHA for d in dms)


def tally(rows):
    """rows: (label, seed-mean test MASE, seed-level DM vs persistence per loss)."""
    best = min(rows, key=lambda r: r[1])
    return {"configurations": len(rows),
            "beat_persistence_absolute": sum(beats(r[2]["absolute"]) for r in rows),
            "beat_persistence_squared": sum(beats(r[2]["squared"]) for r in rows),
            "lowest_mean_mase": round(best[1], 3), "lowest_mean_mase_configuration": best[0]}


def mean(v):
    return sum(v) / len(v)


def headline():
    load = lambda p: json.loads((ROOT / p).read_text())  # noqa: E731
    out = {}

    rows = []
    for setting, r in load("results/final/test_results.json")["results"].items():
        for key, c in r["configs"].items():
            seeds = c["seeds"]
            rows.append((f"{setting} {key}", mean([s["mase"] for s in seeds]),
                         {loss: [s[f"dm_{loss}"] for s in seeds] for loss in ("absolute", "squared")}))
    out["phase3_main_split"] = {"evidence": "results/final/test_table.md", "seeds": 3, **tally(rows)}
    rows = [(f"{setting} {b}", c["mase"], {loss: [c[f"dm_{loss}"]] for loss in ("absolute", "squared")})
            for setting, r in load("results/final/test_results.json")["results"].items()
            for b, c in r["baselines"].items() if b != "persistence"]
    out["statistical_baselines"] = {"evidence": "results/final/test_table.md", "seeds": 1, **tally(rows)}

    rows = []
    for setting, r in load("results/rolling/rolling_results.json")["results"].items():
        for m in ("lstm", "gru", "tsmixer", "patchtst", "tft", "ensemble"):
            p = r["pooled"][m]
            rows.append((f"{setting} {m}", p["mase"], {loss: [p[f"dm_{loss}"]] for loss in ("absolute", "squared")}))
    out["rolling_origin_pooled"] = {"evidence": "results/rolling/rolling_table.md", "seeds": 1, **tally(rows)}

    rows = []
    for setting, r in load("results/chronos/chronos_results.json")["results"].items():
        for size, c in r["sizes"].items():
            seeds = [s["test"] for s in c["seeds"]]
            rows.append((f"{setting} chronos-{size}", mean([s["mase"] for s in seeds]),
                         {loss: [s[f"dm_{loss}"] for s in seeds] for loss in ("absolute", "squared")}))
    out["chronos_zero_shot"] = {"evidence": "results/chronos/chronos_table.md", "seeds": 3, **tally(rows)}

    rows = []
    for setting, r in load("results/fx_target/fx_results.json")["results"].items():
        for m, c in r.items():
            seeds = [s["fx"] for s in c["seeds"]]
            rows.append((f"{setting} {m} (FX route)", mean([s["mase"] for s in seeds]),
                         {loss: [s[f"dm_{loss}"] for s in seeds] for loss in ("absolute", "squared")}))
    out["fx_target"] = {"evidence": "results/fx_target/fx_table.md", "seeds": 3, **tally(rows)}

    acled = load("results/acled/acled_results.json")
    rows = [(f"main lag{c['lag']} {c['model']} h{c['h']} (with conflict)", mean([s["with"]["mase"] for s in c["seeds"]]),
             {loss: [s["with"]["vs_persistence"][loss] for s in c["seeds"]] for loss in ("absolute", "squared")})
            for c in acled["main"]]
    out["acled_main_split"] = {"evidence": "results/acled/acled_table.md", "seeds": 3, **tally(rows)}
    rows = [(f"rolling lag{c['lag']} {c['model']} h{c['h']} (with conflict)", c["with"]["mase"],
             {loss: [c["with"]["vs_persistence"][loss]] for loss in ("absolute", "squared")}) for c in acled["rolling"]]
    out["acled_rolling_pooled"] = {"evidence": "results/acled/acled_table.md", "seeds": 1, **tally(rows)}

    rows = []
    for c in load("results/regime/regime_results.json")["cells"]:
        for arm in ("with", "without"):
            rows.append((f"{c['where']} {c['model']} h{c['h']} ({arm} regime dummies)",
                         mean([s[arm]["mase"] for s in c["seeds"]]),
                         {loss: [s[arm]["vs_persistence"][loss] for s in c["seeds"]] for loss in ("absolute", "squared")}))
    out["regime_dummies"] = {"evidence": "results/regime/regime_table.md", "seeds": 3, **tally(rows)}

    rows = []
    for c in load("results/transfer/transfer_results.json")["cells"]:
        for arm in ("transfer", "scratch"):
            rows.append((f"{c['model']} h{c['h']} ({arm})", mean([s[arm]["mase"] for s in c["seeds"]]),
                         {loss: [s[arm]["vs_persistence"][loss] for s in c["seeds"]] for loss in ("absolute", "squared")}))
    out["transfer"] = {"evidence": "results/transfer/transfer_table.md", "seeds": 3, **tally(rows)}

    total = {k: sum(v[k] for v in out.values()) for k in ("configurations", "beat_persistence_absolute", "beat_persistence_squared")}
    L = ["# Headline: does anything beat persistence? (generated by scripts/make_tables.py)", "",
         "One row per experiment. A configuration beats persistence if the Diebold–Mariano statistic (HLN, Newey–West "
         f"lag h − 1) is negative with p < {ALPHA} in every seed (one seed for the rolling rows), on the test windows, "
         "before any multiple-testing correction. MASE = MAE / MAE(persistence) on the same windows.", "",
         "| experiment | seeds | forecast configurations | beat persistence, absolute loss | beat persistence, squared loss | "
         "lowest seed-mean MASE | configuration | evidence |", "|---|---:|---:|---:|---:|---:|---|---|"]
    for name, v in out.items():
        L.append(f"| {name} | {v['seeds']} | {v['configurations']} | {v['beat_persistence_absolute']} | "
                 f"{v['beat_persistence_squared']} | {v['lowest_mean_mase']:.3f} | {v['lowest_mean_mase_configuration']} | "
                 f"`{v['evidence']}` |")
    L += ["", f"**Total:** {total['configurations']} forecast configurations; {total['beat_persistence_absolute']} beat "
          f"persistence on absolute loss and {total['beat_persistence_squared']} on squared loss.", "",
          "Configurations overlap between rows (for example, the regime 'with' arm on the main split is the Phase 3 run), "
          "so the total counts evaluations, not distinct models."]
    dest = ROOT / "results/summary"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "headline.json").write_text(json.dumps({"alpha": ALPHA, "experiments": out, "total": total}, indent=2) + "\n")
    (dest / "headline.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def ensembles():
    """Members and SLSQP weights of every validation-selected ensemble (results/ensembles/selected_*.json)."""
    L = ["# Validation-selected ensembles (generated by scripts/make_tables.py)", "",
         "Top-2 of the five models' validation-selected configurations by validation MAE; weights fitted by SLSQP on "
         "validation predictions. No test information enters either step.", "",
         "| track | h | seed | members | weights | val MAE | test MAE |", "|---|---:|---:|---|---|---:|---:|"]
    for track in ("global", "sudan"):
        for h in (1, 7, 30):
            for s in (1, 2, 3):
                e = json.loads((ROOT / f"results/ensembles/selected_{track}_h{h}_s{s}.json").read_text())
                L.append(f"| {track} | {h} | {s} | {' + '.join(e['selected'])} | "
                         f"{' / '.join(f'{w:.3f}' for w in e['weights'])} | {e['metrics']['val']['mae']:.2f} | "
                         f"{e['metrics']['test']['mae']:.2f} |")
    (ROOT / "results/summary/ensembles.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=STEPS[0][0], choices=[n for n, _ in STEPS] + ["headline"])
    args = ap.parse_args()
    if args.start != "headline":
        run_steps(args.start)
    headline()
    ensembles()


if __name__ == "__main__":
    main()
