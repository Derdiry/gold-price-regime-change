#!/bin/bash
# Phase 3, stage 1: seed-1 transform-selection runs, validation only (test never predicted).
# Order (author's instruction): all Sudan selection runs, then the Sudan duplicate-rule
# sensitivity runs (LSTM, GRU, MinMax, "keep last"), then all global selection runs.
# Crash-safe: re-running this script resumes; finished models are reused (--resume).
# Stops at the first non-zero exit (exit 3 = memory gate refused to start).
set -u
cd "$(dirname "$0")/.."
PY=venv/bin/python
run() {
  echo "$(date '+%F %T') START $*"
  $PY train.py "$@" --seeds 1 --val-only --resume
  code=$?
  echo "$(date '+%F %T') END exit=$code $*"
  if [ $code -ne 0 ]; then echo "STOPPING: exit $code"; exit $code; fi
}
for cond in minmax log_return revin; do
  for h in 1 7 30; do run --track sudan --condition $cond --horizon $h; done
done
for model in lstm gru; do
  for h in 1 7 30; do run --track sudan --condition minmax --model $model --horizon $h --duplicate-rule last; done
done
for cond in minmax log_return revin; do
  for h in 1 7 30; do run --track global --condition $cond --horizon $h; done
done
echo "$(date '+%F %T') PHASE3 SELECTION DONE"
