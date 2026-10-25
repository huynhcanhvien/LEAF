#!/usr/bin/env bash
# Re-run main-comparison timing (backbones without latency_done.json) once scripts/run_main_comparison.sh has finished its timing phase.
cd "$(dirname "$0")/.."
export LEAF_EXCLUSIVE_TIMING=1
LOG=results/logs
until grep -q 'ALL DONE' $LOG/orchestrate.log; do sleep 60; done
echo "[$(date '+%F %T')] main-comparison timing (resumed, fast exclusive path)" >> $LOG/orchestrate.log
CUDA_VISIBLE_DEVICES=0 ${PYTHON:-python} leaf/reproduction/experiments/main_comparison.py time >> $LOG/main_time.log 2>&1 && echo "[$(date '+%F %T')] main-comparison timing done (resumed)" >> $LOG/orchestrate.log || echo "[$(date '+%F %T')] main-comparison timing FAILED (resumed)" >> $LOG/orchestrate.log
echo "[$(date '+%F %T')] ALL TIMING DONE" >> $LOG/orchestrate.log
