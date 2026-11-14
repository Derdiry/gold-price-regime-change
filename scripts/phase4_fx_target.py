"""
Phase 4.7 orchestrator: Sudan models with SDG/USD as the target.

Rules: results/fx_target/PREREGISTRATION.md. Steps, in order (crash-safe:
re-running resumes; train.py --resume never retrains a finished model):
  1. seed 1, validation only: 5 models x {minmax, log_return, revin} x h in {1, 7, 30}
  2. selection on validation MAE of the rebuilt price (scripts/fx_target_report.py select)
  3. seeds 1-3 of each (model, h)'s selected condition; seed 1 is test-scored
     from its saved checkpoint, seeds 2-3 are trained and scored
  4. report (scripts/fx_target_report.py report)
If the memory gate refuses to start a run (exit 3), waits 10 min and retries,
at most 6 times; any other non-zero exit stops the orchestrator.

    python scripts/phase4_fx_target.py [--dry-run]
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / "venv/bin/python")
HORIZONS, MODELS = [1, 7, 30], ["lstm", "gru", "tsmixer", "patchtst", "tft"]
CONDITIONS = ["minmax", "log_return", "revin"]
FX = ["--track", "sudan", "--target", "fx_rate", "--resume"]
SELECTION = ROOT / "results/fx_target/selection/selection.json"
MEMORY_EXIT, MEMORY_WAIT, MEMORY_TRIES = 3, 600, 6


def run(cmd, label, dry):
    for attempt in range(1, MEMORY_TRIES + 1):
        print(f"{datetime.now():%F %T} START {label} {' '.join(cmd[1:])}", flush=True)
        if dry:
            return
        code = subprocess.run(cmd, cwd=ROOT).returncode
        print(f"{datetime.now():%F %T} END exit={code} {label}", flush=True)
        if code == 0:
            return
        if code == MEMORY_EXIT and attempt < MEMORY_TRIES:
            print(f"memory gate: waiting {MEMORY_WAIT}s (attempt {attempt}/{MEMORY_TRIES})", flush=True)
            time.sleep(MEMORY_WAIT)
            continue
        print(f"STOPPING: exit {code}", flush=True)
        sys.exit(code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    stage1 = [[PY, "train.py", *FX, "--horizon", str(h), "--condition", c, "--model", "all", "--seeds", "1", "--val-only"]
              for h in HORIZONS for c in CONDITIONS]
    for i, cmd in enumerate(stage1, 1):
        run(cmd, f"[stage1 {i}/{len(stage1)}]", args.dry_run)
    run([PY, "scripts/fx_target_report.py", "select"], "[select]", args.dry_run)
    if args.dry_run and not SELECTION.exists():
        print("(dry run: stage 2 depends on the selection)")
        return
    sel = json.loads(SELECTION.read_text())["selection"]
    stage2 = [[PY, "train.py", *FX, "--horizon", str(h), "--condition", sel[f"{m}_h{h}"]["condition"], "--model", m,
               "--seeds", "1", "2", "3"] for h in HORIZONS for m in MODELS]
    for i, cmd in enumerate(stage2, 1):
        run(cmd, f"[stage2 {i}/{len(stage2)}]", args.dry_run)
    run([PY, "scripts/fx_target_report.py", "report"], "[report]", args.dry_run)
    print(f"{datetime.now():%F %T} PHASE4 FX DONE", flush=True)


if __name__ == "__main__":
    main()
