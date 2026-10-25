#!/usr/bin/env bash
# Appends a progress line for scripts/run_alpha_grid_replan.sh every 30 min until it completes or fails.
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
AB=results/ablation_and_alpha_grid
LOG=results/notes/state/alpha-ablation-2026-10-19-progress.md
while true; do
  ph=$(cat $AB/phase 2>/dev/null || echo unknown)
  alive=$(kill -0 "$(cat $AB/chain.pid)" 2>/dev/null && echo alive || echo dead)
  eq=$($PY leaf/reproduction/experiments/alpha_kappa_equivalence.py status 2>/dev/null)
  ab=$($PY leaf/reproduction/experiments/ablation_and_alpha_grid.py status 2>/dev/null)
  gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';')
  err=$(grep -lE "Traceback" $AB/logs/*.log results/alpha_kappa_equivalence/logs/*.log 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')
  echo "- $(date '+%F %T') phase=$ph chain=$alive | $eq | $ab | gpu: $gpu${err:+ | Traceback in: $err}" >> $LOG
  case $ph in completed|failed) exit 0;; esac
  [ $alive = dead ] && exit 1
  sleep 1800
done
