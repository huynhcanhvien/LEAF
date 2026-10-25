#!/usr/bin/env bash
# CLIP comparison, extra baseline points (leaf/reproduction/experiments/clip_comparison.py): cost-only points near every
# LEAF latency (pilot timing on GPU0, sealed), accuracy and retrieval of the new points on both GPUs, a new exclusive
# timing session of every arm, report at bs32 and bs8. Restartable. Run after scripts/run_clip_comparison.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
EXP=leaf/reproduction/experiments/clip_comparison.py
export LEAF_EXCLUSIVE_TIMING=1
BB=${BACKBONES:-clip_b16}           # BACKBONES=clip_l14 for CLIP ViT-L/14 (comma-separated list)
LOG=results/logs; mkdir -p "$LOG"
stamp() { echo "[$(date '+%F %T')] [$BB] $*" | tee -a "$LOG/clip_orchestrate.log"; }
run() { local name=$1; shift; "$@" >> "$LOG/${BB//,/_}_$name.log" 2>&1 || { stamp "$name FAILED"; exit 1; }; }
both() {
  local name=$1; shift
  run ${name}_gpu0 env CUDA_VISIBLE_DEVICES=0 "$@" & local a=$!
  run ${name}_gpu1 env CUDA_VISIBLE_DEVICES=1 "$@" & local b=$!
  wait $a $b
}

stamp "densify start"
run clip_densify env CUDA_VISIBLE_DEVICES=0 $PY $EXP densify --backbones "$BB";      stamp "dense points sealed"
both clip_worker_dense $PY $EXP worker --backbones "$BB";                            stamp "dense accuracy and retrieval done"
run clip_retime env CUDA_VISIBLE_DEVICES=0 $PY $EXP retime --backbones "$BB";        stamp "retiming done"
run clip_report2 env CUDA_VISIBLE_DEVICES= $PY $EXP report --backbones "$BB";        stamp "DENSIFY ALL DONE"
