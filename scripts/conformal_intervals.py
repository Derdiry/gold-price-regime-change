"""
Phase 4.6: split-conformal prediction intervals (calibrated on validation) for
the Phase 3 final configurations and the Phase 2 baselines, and Chronos sample
quantiles, scored on test.

Rules are fixed in results/conformal/PREREGISTRATION.md, which must be
committed before this script runs.

Outputs: results/conformal/{intervals.json, intervals.md}

    python scripts/conformal_intervals.py
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.data.fetchers import load_config  # noqa: E402
from src.models.ensemble import apply_ensemble  # noqa: E402
from src.utils.provenance import provenance  # noqa: E402

TRACKS, HORIZONS, SEEDS = ["sudan", "global"], [1, 7, 30], [1, 2, 3]
MODELS = ["lstm", "gru", "tsmixer", "patchtst", "tft"]
BASELINES = {"sudan": ["persistence", "drift", "arima", "ets", "vecm"], "global": ["persistence", "drift", "arima", "ets"]}
ALPHAS = {80: 0.2, 90: 0.1}
CHRONOS_LEVELS = {80: (0.10, 0.90), 90: (0.05, 0.95)}
SCORES = ["absolute", "relative"]
TOL = 5.0
PREDS, ENS = Path("logs/v2/preds"), Path("results/ensembles")
CHRONOS = PREDS / "chronos"
OUT = Path("results/conformal")
PREREG = OUT / "PREREGISTRATION.md"


def require_committed_preregistration():
    rel = str(PREREG)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = not subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if not (tracked and clean):
        sys.exit(f"{rel} must be committed (and unchanged) before scoring.")
    return subprocess.run(["git", "log", "-1", "--format=%H", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def conformal_q(scores: np.ndarray, alpha: float) -> np.ndarray:
    """Per-column split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score (inf if > n)."""
    n = scores.shape[0]
    r = math.ceil((n + 1) * (1 - alpha))
    if r > n:
        return np.full(scores.shape[1], np.inf)
    return np.sort(scores, axis=0)[r - 1]


def interval(fv, yv, lv, ft, lt, score, alpha):
    if score == "absolute":
        q = conformal_q(np.abs(yv - fv), alpha)
        half = np.broadcast_to(q, ft.shape)
    else:
        q = conformal_q(np.abs(yv - fv) / lv[:, None], alpha)
        half = q[None, :] * lt[:, None]
    return ft - half, ft + half


def evaluate(lo, hi, y, mask=None):
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    out = {"coverage": float(100 * inside.mean()), "width": float(width.mean()),
           "coverage_by_step": {int(k + 1): float(100 * inside[:, k].mean()) for k in {0, 6, 29} if k < y.shape[1]}}
    if mask is not None:
        out["coverage_S1"] = float(100 * inside[mask].mean())
        out["n_pairs_S1"] = int(mask.sum())
    return out


def verdict(cov, nominal):
    if cov < nominal - TOL:
        return "under-covers"
    if cov > nominal + TOL:
        return "over-covers"
    return "close to nominal"


def forecasts(track, h, sel):
    """name -> list over seeds of (val_pred, test_pred); baselines have one entry."""
    out = {}
    for model in MODELS:
        for label, cond in (("minmax", "minmax"), ("selected", sel[f"{model}_{track}_h{h}"]["condition"])):
            out[f"{model}|{label}"] = (cond, [tuple(np.load(PREDS / f"{model}_{track}_h{h}_{cond}_s{s}.npz")[k] for k in ("val_pred", "test_pred"))
                                              for s in SEEDS])
    for label in ("minmax", "selected"):
        fs = []
        for s in SEEDS:
            e = json.loads((ENS / (f"{track}_h{h}_minmax_s{s}.json" if label == "minmax" else f"selected_{track}_h{h}_s{s}.json")).read_text())
            members = {rec.split(f"_{track}_")[0]: np.load(PREDS / f"{rec}.npz") for rec in e["member_records"]}
            fs.append(tuple(apply_ensemble({m: z[k] for m, z in members.items()}, e) for k in ("val_pred", "test_pred")))
        out[f"ensemble|{label}"] = (label, fs)
    return out


def main():
    prereg_commit = require_committed_preregistration()
    cfg = load_config("configs/model_config.yaml")
    sel = json.loads(Path("results/selection/selection.json").read_text())["selection"]
    from posthoc_observed_days import observed_masks
    obs = observed_masks(cfg)
    results = {}
    for track in TRACKS:
        for h in HORIZONS:
            b = np.load(PREDS / "baselines" / f"{track}_h{h}.npz")
            yv, yt, lv, lt = b["y_val"], b["y_test"], b["last_val"], b["last_test"]
            mask = obs[h]["obs"] if track == "sudan" else None
            cell = {"n_val": int(len(yv)), "n_test": int(len(yt)), "conformal": {}, "chronos": {}}
            entries = {**forecasts(track, h, sel),
                       **{f"{n}|baseline": ("baseline", [(b[f"{n}_val"], b[f"{n}_test"])]) for n in BASELINES[track]}}
            for name, (cond, seeds) in entries.items():
                rec = {"condition": cond}
                for score in SCORES:
                    for nom, alpha in ALPHAS.items():
                        ev = [evaluate(*interval(fv, yv, lv, ft, lt, score, alpha), yt, mask) for fv, ft in seeds]
                        rec[f"{score}_{nom}"] = {"seeds": ev, "coverage_mean": float(np.mean([e["coverage"] for e in ev])),
                                                 "width_mean": float(np.mean([e["width"] for e in ev]))}
                cell["conformal"][name] = rec
            for name, rec in cell["conformal"].items():
                for key, r in rec.items():
                    if key == "condition":
                        continue
                    nom = int(key.split("_")[1])
                    r["verdict"] = verdict(r["coverage_mean"], nom)
                    r["width_vs_persistence"] = r["width_mean"] / cell["conformal"]["persistence|baseline"][key]["width_mean"]
            # Chronos sample quantiles (uncalibrated), if its forecasts exist
            for size in ("mini", "small"):
                files = [CHRONOS / f"{size}_{track}_s{s}.npz" for s in SEEDS]
                if not all(f.exists() for f in files):
                    continue
                from run_chronos import windows
                pos, origins, _ = windows(track, cfg)
                rows = np.searchsorted(origins["test"], pos[(h, "test")])
                rec = {}
                for nom, (ql, qh) in CHRONOS_LEVELS.items():
                    ev = []
                    for f in files:
                        z = np.load(f)
                        lev = list(np.round(z["quantile_levels"], 2))
                        Q = z["test_quantiles"][rows][:, :, :h]
                        ev.append(evaluate(Q[:, lev.index(ql)], Q[:, lev.index(qh)], yt, mask))
                    cov = float(np.mean([e["coverage"] for e in ev]))
                    width = float(np.mean([e["width"] for e in ev]))
                    rec[f"quantile_{nom}"] = {"seeds": ev, "coverage_mean": cov, "width_mean": width, "verdict": verdict(cov, nom),
                                              "width_vs_persistence": width / cell["conformal"]["persistence|baseline"][f"absolute_{nom}"]["width_mean"]}
                cell["chronos"][size] = rec
            results[f"{track}_h{h}"] = cell

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "intervals.json").write_text(json.dumps(
        {"provenance": provenance(cfg), "preregistration_commit": prereg_commit, "preregistration_text": PREREG.read_text(),
         "tolerance_pp": TOL, "results": results}, indent=2, default=float))
    write_markdown(results, prereg_commit)


def ms(vals, fmt):
    v = np.asarray(vals, float)
    return fmt.format(v.mean()) + (f" ± {fmt.format(v.std(ddof=1))}" if len(v) > 1 else "")


def write_markdown(results, prereg_commit):
    L = ["# Prediction intervals on test (generated by scripts/conformal_intervals.py)", "",
         f"Rules fixed in `{PREREG}` (commit `{prereg_commit[:7]}`) before scoring. Split conformal calibrated on the "
         "validation windows, per forecast step. `abs` = ŷ ± q (price units); `rel` = ŷ ± q·P_t (q from |e|/P_t). "
         f"Coverage pooled over all (window, step) pairs; mean ± std over seeds. Verdict: within ±{TOL:.0f} pp of nominal = close; "
         "else under/over-covers. Width ratio = mean width / persistence's width under the same score and level.", ""]
    for key, cell in results.items():
        track, h = key.split("_h")
        s1 = track == "sudan"
        L += [f"## {track} h={h}", "", f"Calibration windows {cell['n_val']}, test windows {cell['n_test']}.", "",
              "| forecast | condition | abs 80% cov | abs 90% cov | abs 90% width | ×pers | verdict abs 80 / 90 | rel 80% cov | rel 90% cov | rel 90% width | verdict rel 80 / 90 |"
              + (" S1 cov abs/rel 90% |" if s1 else ""),
              "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|" + ("---:|" if s1 else "")]
        for name, rec in cell["conformal"].items():
            a8, a9, r8, r9 = (rec[k] for k in ("absolute_80", "absolute_90", "relative_80", "relative_90"))
            cov = lambda r: ms([e["coverage"] for e in r["seeds"]], "{:.1f}")  # noqa: E731
            row = (f"| {name.split('|')[0]} | {rec['condition']} | {cov(a8)} | {cov(a9)} | {ms([e['width'] for e in a9['seeds']], '{:.1f}')} | "
                   f"{a9['width_vs_persistence']:.2f} | {a8['verdict']} / {a9['verdict']} | {cov(r8)} | {cov(r9)} | "
                   f"{ms([e['width'] for e in r9['seeds']], '{:.1f}')} | {r8['verdict']} / {r9['verdict']} |")
            if s1:
                row += f" {ms([e['coverage_S1'] for e in a9['seeds']], '{:.1f}')} / {ms([e['coverage_S1'] for e in r9['seeds']], '{:.1f}')} |"
            L.append(row)
        if cell["chronos"]:
            L += ["", "Chronos sample quantiles (uncalibrated; 10–90% for 80%, 5–95% for 90%):", "",
                  "| model | 80% cov | 90% cov | 90% width | ×pers (abs conformal) | verdict 80 / 90 |" + (" S1 cov 90% |" if s1 else ""),
                  "|---|---:|---:|---:|---:|---|" + ("---:|" if s1 else "")]
            for size, rec in cell["chronos"].items():
                q8, q9 = rec["quantile_80"], rec["quantile_90"]
                row = (f"| chronos-{size} | {ms([e['coverage'] for e in q8['seeds']], '{:.1f}')} | {ms([e['coverage'] for e in q9['seeds']], '{:.1f}')} | "
                       f"{ms([e['width'] for e in q9['seeds']], '{:.1f}')} | {q9['width_vs_persistence']:.2f} | {q8['verdict']} / {q9['verdict']} |")
                if s1:
                    row += f" {ms([e['coverage_S1'] for e in q9['seeds']], '{:.1f}')} |"
                L.append(row)
        L.append("")
    (OUT / "intervals.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
