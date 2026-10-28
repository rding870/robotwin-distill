#!/usr/bin/env bash
# Reproducibility check: the same teacher config twice on a frozen-seed subset, then compare_runs.py.
# Usage: scripts/run_repro_check.sh <prefix> [--wait-pid PID]   (runs <prefix>_a, then <prefix>_b, never concurrently)
set -uo pipefail
cd "$(dirname "$0")/.."
PREFIX=${1:?prefix}; shift
if [ "${1:-}" = "--wait-pid" ]; then while kill -0 "$2" 2>/dev/null; do sleep 30; done; shift 2; fi
TASKS=pick_dual_bottles,stamp_seal,place_mouse_pad,place_fan,place_burger_fries
ARGS=(--nfe 30 --tasks $TASKS --max-seeds-per-task 10)
for R in a b; do
  scripts/run_teacher_frozen.sh "${PREFIX}_$R" "${ARGS[@]}" > "logs/${PREFIX}_$R.log" 2>&1 < /dev/null
  rc=$?
  echo "$(date -Is) ${PREFIX}_$R exit $rc"
  [ $rc -eq 0 ] || [ $rc -eq 3 ] || { echo "run ${PREFIX}_$R crashed; stopping"; exit $rc; }   # 3 = finished but invalid
done
source env.sh
python scripts/compare_runs.py "eval/results/${PREFIX}_a" "eval/results/${PREFIX}_b" --out "eval/results/${PREFIX}_report.md" > /dev/null
echo "$(date -Is) compare exit $? -> eval/results/${PREFIX}_report.md"
