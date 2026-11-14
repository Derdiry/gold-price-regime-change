"""
Data and model facts quoted in README.md and docs/CLAIMS.md, computed from the
data files and the saved run records (no training).

  data        row counts of the raw file and of the daily series
  fx          SDG/USD at the first and last observation and at its peak, and
              the implied loss of the pound's USD value
  variance    R² of OLS regressions of log price on log SDG/USD and on log USD/oz
  range       Sudan and global training maximum vs the test targets
  models      parameter counts per (model, track, horizon, condition), from
              results/runs/*_s1.json, and the number of saved training runs

Writes results/summary/facts.{json,md}.

    python scripts/project_facts.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.processors import chronological_split  # noqa: E402

OUT = ROOT / "results/summary"


def r2(y, x):
    """R² of an OLS fit of y on [1, x]."""
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def data_rows():
    raw = pd.read_csv(ROOT / "data/raw/new_Data.csv")
    v2 = pd.read_csv(ROOT / "data/processed/v2/sudan_merged.csv")
    return {"raw_rows": len(raw), "v2_daily_rows": len(v2), "v2_observed_rows": int((v2["is_ffilled"] == 0).sum())}


def fx():
    v2 = pd.read_csv(ROOT / "data/processed/v2/sudan_merged.csv", parse_dates=["date"])
    obs = v2[v2["is_ffilled"] == 0]
    first, last, peak = obs.iloc[0], obs.iloc[-1], obs.loc[obs["fx_rate"].idxmax()]
    return {"first": {"date": str(first.date.date()), "sdg_per_usd": float(first.fx_rate)},
            "last": {"date": str(last.date.date()), "sdg_per_usd": float(last.fx_rate)},
            "peak": {"date": str(peak.date.date()), "sdg_per_usd": float(peak.fx_rate)},
            "usd_value_lost_first_to_last_pct": float(100 * (1 - first.fx_rate / last.fx_rate)),
            "usd_value_lost_first_to_peak_pct": float(100 * (1 - first.fx_rate / peak.fx_rate)),
            "years_first_to_last": round((last.date - first.date).days / 365.25, 2),
            **fx_steps(obs)}


def fx_steps(obs):
    """How SDG/USD moves between consecutive observed rows (forward-filled days excluded)."""
    obs = obs.reset_index(drop=True)
    ratio = (obs["fx_rate"] / obs["fx_rate"].shift()).iloc[1:]
    gap_end = obs.index[obs["date"] >= pd.Timestamp("2021-01-02")][0]
    after = ratio[ratio.index > gap_end].nlargest(2)
    step = lambda i: {"from": str(obs.date[i - 1].date()), "fx_from": float(obs.fx_rate[i - 1]),  # noqa: E731
                      "to": str(obs.date[i].date()), "fx_to": float(obs.fx_rate[i])}
    return {"observed_steps": int(len(ratio)), "observed_steps_unchanged": int((ratio == 1).sum()),
            "median_abs_step_pct": float(100 * (ratio - 1).abs().median()),
            "across_gap": step(gap_end), "largest_steps_after_gap": [step(i) for i in after.index]}


def variance():
    v2 = pd.read_csv(ROOT / "data/processed/v2/sudan_merged.csv")
    obs = v2[v2["is_ffilled"] == 0]
    logp = np.log(obs["price_sdg"].to_numpy(float))
    return {"r2_logprice_on_logfx_observed": r2(logp, np.log(obs["fx_rate"].to_numpy(float))),
            "r2_logprice_on_logintl_observed": r2(logp, np.log(obs["intl_price_usd"].to_numpy(float))),
            "observed_rows": len(obs)}


def ranges():
    out = {}
    sudan = pd.read_csv(ROOT / "data/processed/v2/sudan_merged.csv", parse_dates=["date"]).set_index("date")
    tr, _, te = chronological_split(sudan)
    out["sudan"] = {"train_max": float(tr["price_sdg"].max()), "test_min": float(te["price_sdg"].min()),
                    "test_max": float(te["price_sdg"].max()),
                    "test_min_over_train_max": float(te["price_sdg"].min() / tr["price_sdg"].max()),
                    "test_max_over_train_max": float(te["price_sdg"].max() / tr["price_sdg"].max()),
                    "train_end": str(tr.index.max().date()), "test_start": str(te.index.min().date())}
    path = ROOT / "data/processed/v2/global_merged.csv"
    if path.exists():
        g = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        target = "gold_close" if "gold_close" in g.columns else g.columns[0]
        tr, _, te = chronological_split(g)
        out["global"] = {"target": target, "train_max": float(tr[target].max()), "test_days": len(te),
                         "test_days_above_train_max": int((te[target] > tr[target].max()).sum())}
    else:
        prev = json.loads((OUT / "facts.json").read_text()) if (OUT / "facts.json").exists() else {}
        out["global"] = prev.get("range", {}).get("global", "global_merged.csv not present (Yahoo/FRED-derived, gitignored)")
    return out


def models():
    sizes = {}
    for p in sorted((ROOT / "results/runs").glob("*_s1.json")):
        r = json.loads(p.read_text())
        sizes[r["key"]] = r["n_params"]["total"]
    runs = sorted(ROOT.glob("results/**/runs/*.json"))
    patch = {k: v for k, v in sizes.items() if k.startswith("patchtst_")}
    chronos = json.loads((ROOT / "results/chronos/chronos_results.json").read_text())["n_params"]
    return {"n_params_seed1": sizes, "patchtst_min": min(patch.values()), "patchtst_max": max(patch.values()),
            "all_models_min": min(sizes.values()), "all_models_max": max(sizes.values()),
            "chronos_n_params": chronos, "saved_training_run_records": len(runs),
            "phase3_run_records": len(list((ROOT / "results/runs").glob("*.json")))}


def main():
    facts = {"data": data_rows(), "fx": fx(), "variance": variance(), "range": ranges(), "models": models()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    d, f, v, r, m = facts["data"], facts["fx"], facts["variance"], facts["range"], facts["models"]
    g = r["global"]
    L = ["# Data and model facts (generated by scripts/project_facts.py)", "",
         "## Sudan data", "",
         f"- Raw file: {d['raw_rows']} rows. Daily series: {d['v2_daily_rows']} rows, of which {d['v2_observed_rows']} observed.",
         "", "## SDG/USD", "",
         f"- First observation {f['first']['date']}: {f['first']['sdg_per_usd']:.2f}; last {f['last']['date']}: "
         f"{f['last']['sdg_per_usd']:.2f}; peak {f['peak']['date']}: {f['peak']['sdg_per_usd']:.2f}.",
         f"- USD value of the pound lost, first → last observation ({f['years_first_to_last']} years): "
         f"{f['usd_value_lost_first_to_last_pct']:.1f}%; first → peak: {f['usd_value_lost_first_to_peak_pct']:.1f}%.",
         f"- Between consecutive observed rows ({f['observed_steps']} steps), SDG/USD is unchanged in "
         f"{f['observed_steps_unchanged']}; median absolute step {f['median_abs_step_pct']:.2f}%. Zero-change days in the "
         "daily series therefore come mostly from forward-filling days without a quote.",
         f"- Across the 404-day gap: {f['across_gap']['fx_from']:.2f} ({f['across_gap']['from']}) → "
         f"{f['across_gap']['fx_to']:.2f} ({f['across_gap']['to']}); the data do not show when inside the gap it moved.",
         "- Largest steps between consecutive observations after the gap: " + "; ".join(
             f"{x['fx_from']:.2f} ({x['from']}) → {x['fx_to']:.2f} ({x['to']})" for x in f["largest_steps_after_gap"]) + ".",
         "", "## Price and exchange rate (OLS on the observed rows)", "",
         f"- On the {v['observed_rows']} observed rows: R² of log price on log SDG/USD {v['r2_logprice_on_logfx_observed']:.4f}; "
         f"R² of log price on log USD/oz {v['r2_logprice_on_logintl_observed']:.4f}.",
         "", "## Test range vs training range (70/15/15 split)", "",
         f"- Sudan: training maximum {r['sudan']['train_max']:.0f} SDG/g (training ends {r['sudan']['train_end']}); test "
         f"targets {r['sudan']['test_min']:.0f}–{r['sudan']['test_max']:.0f} SDG/g = "
         f"{r['sudan']['test_min_over_train_max']:.1f}–{r['sudan']['test_max_over_train_max']:.1f}× the training maximum.",
         (f"- Global: {g['test_days_above_train_max']} of {g['test_days']} test days above the training maximum "
          f"{g['train_max']:.2f} USD/oz ({g['target']})." if isinstance(g, dict) else f"- Global: {g}."),
         "", "## Model sizes (seed-1 run records)", "",
         f"- {m['saved_training_run_records']} saved training-run records under results/ "
         f"({m['phase3_run_records']} in Phase 3, results/runs/).",
         f"- Trained models: {m['all_models_min']:,}–{m['all_models_max']:,} parameters; PatchTST "
         f"{m['patchtst_min']:,}–{m['patchtst_max']:,}.",
         f"- Chronos: mini {m['chronos_n_params']['mini']:,}, small {m['chronos_n_params']['small']:,}.", "",
         "| run (seed 1) | parameters |", "|---|---:|", *[f"| {k} | {n:,} |" for k, n in m["n_params_seed1"].items()]]
    (OUT / "facts.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
