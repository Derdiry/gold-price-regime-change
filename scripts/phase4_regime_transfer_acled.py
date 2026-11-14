"""
Phase 4.4 -> 4.5 -> 4.3 orchestrator.

Rules: results/{regime,transfer,acled}/PREREGISTRATION.md. Steps, in order
(crash-safe: re-running resumes; train.py --resume never retrains a finished
model):
  4.4  folds 4-5: seeds 2-3 with regime dummies, seeds 1-3 without;
       main split: seeds 1-3 without; report
  4.5  per h: global pretraining, Sudan from scratch, Sudan fine-tuning
       (GRU, PatchTST, 18 shared inputs, log_return, seeds 1-3); report
  4.3  main split, lags 1-3, seeds 1-3, validation only; lag selection on
       validation; test scoring of every main-split run from its checkpoint;
       rolling folds 1-5 with the selected lag, seed 1; report
The memory gate (exit 3) is retried after 10 min, at most 6 times; any other
non-zero exit of a training run or of the 4.3 lag selection stops the
orchestrator. A failing report is logged ("REPORT FAILED") and does not stop
later training; reports can be re-run on their own.

    python scripts/phase4_regime_transfer_acled.py [--dry-run] [--only 4.4 4.5 4.3]
"""

import argparse
import json
import subprocess
from datetime import datetime

from phase4_fx_target import PY, ROOT, run

HORIZONS, FOLDS = [1, 7, 30], [1, 2, 3, 4, 5]
SUDAN = ["--track", "sudan", "--resume"]


def report(cmd, label, dry):
    print(f"{datetime.now():%F %T} START {label} {' '.join(cmd[1:])}", flush=True)
    if dry:
        return
    code = subprocess.run(cmd, cwd=ROOT).returncode
    print(f"{datetime.now():%F %T} END exit={code} {label}" + ("" if code == 0 else " REPORT FAILED"), flush=True)


def by_condition(models, h):
    sel = json.loads((ROOT / "results/selection/selection.json").read_text())["selection"]
    out = {}
    for m in models:
        out.setdefault(sel[f"{m}_sudan_h{h}"]["condition"], []).append(m)
    return out


def regime(dry):
    cmds = []
    for k in (4, 5):
        for h in HORIZONS:
            for c, ms in by_condition(["tft", "lstm"], h).items():
                base = [PY, "train.py", *SUDAN, "--fold", str(k), "--horizon", str(h), "--condition", c, "--model", *ms]
                cmds += [base + ["--seeds", "2", "3"], base + ["--ablation", "no_regime", "--seeds", "1", "2", "3"]]
    for h in HORIZONS:
        for c, ms in by_condition(["tft", "lstm"], h).items():
            cmds.append([PY, "train.py", *SUDAN, "--horizon", str(h), "--condition", c, "--model", *ms,
                         "--ablation", "no_regime", "--seeds", "1", "2", "3"])
    for i, cmd in enumerate(cmds, 1):
        run(cmd, f"[4.4 {i}/{len(cmds)}]", dry)
    report([PY, "scripts/regime_report.py"], "[4.4 report]", dry)
    print(f"{datetime.now():%F %T} PHASE4.4 DONE", flush=True)


def transfer(dry):
    shared = ["--feature-set", "shared", "--condition", "log_return", "--model", "gru", "patchtst", "--seeds", "1", "2", "3", "--resume"]
    cmds = []
    for h in HORIZONS:
        cmds += [[PY, "train.py", "--track", "global", "--horizon", str(h), *shared],
                 [PY, "train.py", "--track", "sudan", "--horizon", str(h), *shared],
                 [PY, "train.py", "--track", "sudan", "--horizon", str(h), *shared, "--transfer-from-global"]]
    for i, cmd in enumerate(cmds, 1):
        run(cmd, f"[4.5 {i}/{len(cmds)}]", dry)
    report([PY, "scripts/transfer_report.py"], "[4.5 report]", dry)
    print(f"{datetime.now():%F %T} PHASE4.5 DONE", flush=True)


def acled(dry):
    def main_cmds(extra):
        return [[PY, "train.py", *SUDAN, "--conflict-lag", str(k), "--horizon", str(h), "--condition", c, "--model", *ms,
                 "--seeds", "1", "2", "3", *extra]
                for k in (1, 2, 3) for h in HORIZONS for c, ms in by_condition(["tft", "lstm"], h).items()]
    stage_a = main_cmds(["--val-only"])
    for i, cmd in enumerate(stage_a, 1):
        run(cmd, f"[4.3 val {i}/{len(stage_a)}]", dry)
    run([PY, "scripts/acled_report.py", "select"], "[4.3 select]", dry)
    stage_b = main_cmds([])
    for i, cmd in enumerate(stage_b, 1):
        run(cmd, f"[4.3 test {i}/{len(stage_b)}]", dry)
    sel = ROOT / "results/acled/selection/lag_selection.json"
    k_star = json.loads(sel.read_text())["selected_lag"] if sel.exists() else 1
    roll = [[PY, "train.py", *SUDAN, "--fold", str(j), "--conflict-lag", str(k_star), "--horizon", str(h), "--condition", c,
             "--model", *ms, "--seeds", "1"]
            for j in FOLDS for h in HORIZONS for c, ms in by_condition(["tft", "lstm"], h).items()]
    for i, cmd in enumerate(roll, 1):
        run(cmd, f"[4.3 rolling lag {k_star} {i}/{len(roll)}]", dry)
    report([PY, "scripts/acled_report.py", "report"], "[4.3 report]", dry)
    print(f"{datetime.now():%F %T} PHASE4.3 DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="+", choices=["4.4", "4.5", "4.3"], default=["4.4", "4.5", "4.3"])
    args = ap.parse_args()
    for step, fn in (("4.4", regime), ("4.5", transfer), ("4.3", acled)):
        if step in args.only:
            fn(args.dry_run)
    print(f"{datetime.now():%F %T} PHASE4 REGIME/TRANSFER/ACLED DONE", flush=True)


if __name__ == "__main__":
    main()
