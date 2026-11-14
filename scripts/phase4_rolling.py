"""
Phase 4.1 orchestrator: rolling-origin folds 1-5, seed 1.

Rules: results/rolling/PREREGISTRATION.md. Steps, in order (crash-safe:
re-running resumes; finished baseline folds are skipped, train.py --resume
never retrains a finished model):
  1. baselines per fold (scripts/run_baselines.py --fold k), both tracks
  2. per fold, track and horizon: each model under its Phase 3 selected
     condition (models sharing a condition run in one train.py call)
  3. report (scripts/rolling_report.py)
The memory gate (exit 3) is retried after 10 min, at most 6 times; any other
non-zero exit stops the orchestrator.

    python scripts/phase4_rolling.py [--dry-run]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from phase4_fx_target import ROOT, PY, run

TRACKS, HORIZONS, MODELS, FOLDS = ["sudan", "global"], [1, 7, 30], ["lstm", "gru", "tsmixer", "patchtst", "tft"], [1, 2, 3, 4, 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sel = json.loads((ROOT / "results/selection/selection.json").read_text())["selection"]
    for k in FOLDS:
        done = all((ROOT / f"results/rolling/fold{k}/baselines/{t}_h{h}.json").exists() for t in TRACKS for h in HORIZONS)
        if not done:
            run([PY, "scripts/run_baselines.py", "--fold", str(k)], f"[baselines fold {k}]", args.dry_run)
    cmds = []
    for k in FOLDS:
        for track in TRACKS:
            for h in HORIZONS:
                by_cond = {}
                for m in MODELS:
                    by_cond.setdefault(sel[f"{m}_{track}_h{h}"]["condition"], []).append(m)
                for c, ms in by_cond.items():
                    cmds.append([PY, "train.py", "--fold", str(k), "--track", track, "--horizon", str(h),
                                 "--condition", c, "--model", *ms, "--seeds", "1", "--resume"])
    for i, cmd in enumerate(cmds, 1):
        run(cmd, f"[neural {i}/{len(cmds)}]", args.dry_run)
    run([PY, "scripts/rolling_report.py"], "[report]", args.dry_run)
    print(f"{datetime.now():%F %T} PHASE4 ROLLING DONE", flush=True)


if __name__ == "__main__":
    main()
