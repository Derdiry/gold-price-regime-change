"""
Phase 3, stage 2: final configurations, seeds 1-3, test evaluated once.

Final configurations (results/selection/selection.json):
  - MinMax for every model (the original-setup condition)
  - each (model, track, horizon)'s validation-selected condition

Steps, in order (crash-safe: re-running resumes; train.py --resume reuses or
test-scores saved checkpoints and never retrains a finished model):
  1. seed 1: first and only test evaluation of every final configuration
     (from the saved best-validation checkpoint; no retraining)
  2. seeds 2-3: train and score every final configuration on val and test
  3. sensitivity: test-score the stage-1 MinMax 'keep last' runs (Sudan LSTM/GRU)
  4. sensitivity: Sudan LSTM/GRU under their selected condition, 'keep last', seeds 1-3
Stops at the first non-zero exit (3 = memory gate).

    python scripts/phase3_stage2.py            # run everything
    python scripts/phase3_stage2.py --dry-run  # print the commands
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / "venv/bin/python")
TRACKS, HORIZONS, MODELS = ["sudan", "global"], [1, 7, 30], ["lstm", "gru", "tsmixer", "patchtst", "tft"]


def commands() -> list[list[str]]:
    sel = json.loads((ROOT / "results/selection/selection.json").read_text())["selection"]
    cmds = []
    for seeds in (["1"], ["2", "3"]):
        for track in TRACKS:
            for h in HORIZONS:
                cmds.append(["--track", track, "--horizon", str(h), "--condition", "minmax",
                             "--model", "all", "--seeds", *seeds])
                for m in MODELS:
                    c = sel[f"{m}_{track}_h{h}"]["condition"]
                    if c != "minmax":
                        cmds.append(["--track", track, "--horizon", str(h), "--condition", c,
                                     "--model", m, "--seeds", *seeds])
    for m in ("lstm", "gru"):
        for h in HORIZONS:
            cmds.append(["--track", "sudan", "--horizon", str(h), "--condition", "minmax", "--model", m,
                         "--seeds", "1", "--duplicate-rule", "last"])
    for m in ("lstm", "gru"):
        for h in HORIZONS:
            c = sel[f"{m}_sudan_h{h}"]["condition"]
            cmds.append(["--track", "sudan", "--horizon", str(h), "--condition", c, "--model", m,
                         "--seeds", "1", "2", "3", "--duplicate-rule", "last"])
    return [[PY, "train.py", *c, "--resume"] for c in cmds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cmds = commands()
    for i, cmd in enumerate(cmds, 1):
        line = " ".join(cmd[2:])
        print(f"{datetime.now():%F %T} START [{i}/{len(cmds)}] {line}", flush=True)
        if args.dry_run:
            continue
        code = subprocess.run(cmd, cwd=ROOT).returncode
        print(f"{datetime.now():%F %T} END exit={code} [{i}/{len(cmds)}] {line}", flush=True)
        if code != 0:
            print(f"STOPPING: exit {code}", flush=True)
            sys.exit(code)
    print(f"{datetime.now():%F %T} PHASE3 STAGE2 DONE", flush=True)


if __name__ == "__main__":
    main()
