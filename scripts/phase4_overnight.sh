#!/usr/bin/env bash
# Phase 4 overnight chain (approved 2026-10-29): Chronos predict (already running) ->
# Chronos score -> conformal intervals with Chronos quantiles -> FX target (4.7).
# Stops at the first failure. Log: logs/v2/phase4/overnight.out
set -u
cd "$(dirname "$0")/.."
PY=venv/bin/python
stamp() { date "+%F %T"; }
while pgrep -f "^venv/bin/python -u scripts/run_chronos.py predict" > /dev/null; do sleep 60; done
n=$(ls logs/v2/preds/chronos/*_s[123].npz 2>/dev/null | wc -l)
echo "$(stamp) chronos predict finished: $n/12 prediction files"
if [ "$n" -ne 12 ]; then
  echo "$(stamp) re-running chronos predict (resumes missing files)"
  $PY -u scripts/run_chronos.py predict || { echo "$(stamp) STOP: chronos predict exit $?"; exit 1; }
fi
$PY -u scripts/run_chronos.py score > logs/v2/phase4/chronos_score.out 2>&1 || { echo "$(stamp) STOP: chronos score failed"; exit 1; }
echo "$(stamp) chronos scored"
$PY -u scripts/conformal_intervals.py > logs/v2/phase4/conformal.out 2>&1 || { echo "$(stamp) STOP: conformal failed"; exit 1; }
echo "$(stamp) conformal (with Chronos quantiles) done"
$PY -u scripts/phase4_fx_target.py > logs/v2/phase4/fx_target.out 2>&1 || { echo "$(stamp) STOP: FX orchestrator failed"; exit 1; }
echo "$(stamp) OVERNIGHT CHAIN DONE"
