#!/usr/bin/env bash
# Main comparison and latency pricing: accuracy first on both GPUs (one process per card), then every latency
# measurement with nothing else running, one device at a time. Restartable (finished work is skipped).
#   bash scripts/run_main_comparison.sh        (after `leaf experiments run main_comparison plan` and `... latency_pricing plan`)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export LEAF_EXCLUSIVE_TIMING=1
LOG=results/logs; mkdir -p "$LOG"
stamp() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG/orchestrate.log"; }
run() { local name=$1; shift; "$@" >> "$LOG/$name.log" 2>&1 || stamp "$name FAILED"; }
both() {
  local name=$1; shift
  run ${name}_gpu0 env CUDA_VISIBLE_DEVICES=0 "$@" & local a=$!
  run ${name}_gpu1 env CUDA_VISIBLE_DEVICES=1 "$@" & local b=$!
  wait $a $b
}
campaign_jobs() { pgrep -f 'leaf/reproduction/experiments/(main_comparison|latency_pricing)\.py' | wc -l; }

stamp "start"
while [ "$(campaign_jobs)" -gt 0 ]; do sleep 30; done     # workers started earlier finish their queue first
both main_worker $PY leaf/reproduction/experiments/main_comparison.py worker;    stamp "main comparison: accuracy done"
both pricing_worker $PY leaf/reproduction/experiments/latency_pricing.py worker; stamp "latency pricing: accuracy done"
touch results/logs/accuracy_phase_done

stamp "timing phase (one device at a time)"
run main_time env CUDA_VISIBLE_DEVICES=0 $PY leaf/reproduction/experiments/main_comparison.py time;                                 stamp "main comparison: timing done"
run pricing_final_gpu env CUDA_VISIBLE_DEVICES=0 $PY leaf/reproduction/experiments/latency_pricing.py final_time --devices cuda; stamp "latency pricing: GPU timing done"
run pricing_final_cpu env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $PY leaf/reproduction/experiments/latency_pricing.py final_time --devices cpu; stamp "latency pricing: CPU timing done"

run pricing_compare env CUDA_VISIBLE_DEVICES= $PY leaf/reproduction/experiments/latency_pricing.py compare
stamp "ALL DONE"
