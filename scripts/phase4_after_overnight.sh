#!/usr/bin/env bash
# Queued 2026-10-29 (approved): Phase 4.1 rolling-origin, after the overnight chain
# (Chronos -> conformal -> FX) has finished successfully. Log: logs/v2/phase4/rolling.out
set -u
cd "$(dirname "$0")/.."
while pgrep -f "^bash scripts/phase4_overnight.sh" > /dev/null; do sleep 60; done
if ! grep -q "OVERNIGHT CHAIN DONE" logs/v2/phase4/overnight.out; then
  echo "$(date '+%F %T') overnight chain did not finish cleanly; rolling-origin not started"; exit 1
fi
echo "$(date '+%F %T') starting rolling-origin"
venv/bin/python -u scripts/phase4_rolling.py
