#!/usr/bin/env bash
# Efficiency and search-cost measurements (leaf/reproduction/experiments/efficiency_and_cost.py). Restartable.
# Probe wall-clock on both GPUs (GPU jobs pinned to CPU cores 0-7), GPU overheads, then the CPU stages on the pinned
# timing cores 8-15 with the GPUs idle, then one exclusive batch-32 timing session with nothing else running.
#   bash scripts/run_efficiency_and_cost.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
EXP=leaf/reproduction/experiments/efficiency_and_cost.py
export LEAF_EXCLUSIVE_TIMING=1
LOG=results/logs; mkdir -p "$LOG"
stamp() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG/efficiency_orchestrate.log"; }
run() { local name=$1; shift; "$@" >> "$LOG/efficiency_$name.log" 2>&1 || { stamp "$name FAILED"; exit 1; }; }

stamp "start"
( run probe_gpu0 env CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 $PY $EXP probe_rate --gpu 0 &&
  run overheads_gpu env CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 $PY $EXP overheads --device cuda ) & a=$!
run probe_gpu1 env CUDA_VISIBLE_DEVICES=1 taskset -c 0-7 $PY $EXP probe_rate --gpu 1 & b=$!
wait $a $b;                                                                                   stamp "probe rate and GPU overheads done"
run plan_time env CUDA_VISIBLE_DEVICES= taskset -c 8-15 $PY $EXP plan_time;                    stamp "planning time done"
run overheads_cpu env CUDA_VISIBLE_DEVICES= $PY $EXP overheads --device cpu;                   stamp "CPU overheads done"
run time_bs32 env CUDA_VISIBLE_DEVICES=0 $PY $EXP time_bs32;                                   stamp "batch-32 session done"
run report env CUDA_VISIBLE_DEVICES= $PY $EXP report;                                          stamp "ALL DONE"
