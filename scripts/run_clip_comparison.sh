#!/usr/bin/env bash
# CLIP comparison (leaf/reproduction/experiments/clip_comparison.py): search on both GPUs, plan and seal, accuracy and
# retrieval on both GPUs (one process per card), then every latency measurement with nothing else running. Restartable.
#   bash scripts/run_clip_comparison.sh                          (CLIP ViT-B/16, the paper)
#   BACKBONES=clip_l14 bash scripts/run_clip_comparison.sh       (CLIP ViT-L/14)
# Once, on the CPU: leaf data clip-text --backbone <backbone> --task all; leaf data prepare --split flickr_test; ... --split coco_test
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

stamp "start"
run clip_search $PY $EXP search --gpus 0,1 --backbones "$BB";                        stamp "search done (probes, budgets, SALT, DP)"
run clip_plan env CUDA_VISIBLE_DEVICES= $PY $EXP plan --backbones "$BB";             stamp "plans sealed"
both clip_worker $PY $EXP worker --backbones "$BB";                                  stamp "accuracy and retrieval done"
run clip_time env CUDA_VISIBLE_DEVICES=0 $PY $EXP time --backbones "$BB";            stamp "timing done"
run clip_report env CUDA_VISIBLE_DEVICES= $PY $EXP report --backbones "$BB";         stamp "ALL DONE"
