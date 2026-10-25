#!/usr/bin/env bash
# Batch-size sensitivity (batch 32 / 64 / 128) of the main comparison for DeiT-B, DeiT-S and ViT-B (MAE):
# one exclusive timing session per backbone on GPU 0, started only when both cards are idle (the stage waits itself),
# then the CPU report. Logs: results/logs/batch_sensitivity_20261025/. Nothing in results/ is overwritten.
set -u
cd "$(dirname "$0")/.."
LOG=results/logs/batch_sensitivity_20261025; mkdir -p "$LOG"
PY=python
BS=leaf/reproduction/experiments/batch_sensitivity.py
stamp() { date '+%H:%M:%S'; }
run() { local gpu=$1 name=$2; shift 2; echo "$(stamp) start [$name] GPU=$gpu: $*" >> "$LOG/orchestrator.log"; CUDA_VISIBLE_DEVICES=$gpu "$@" > "$LOG/$name.log" 2>&1; local rc=$?; echo "$(stamp) end   [$name] rc=$rc" >> "$LOG/orchestrator.log"; return $rc; }

echo "$(stamp) orchestrator started" >> "$LOG/orchestrator.log"
run 0 time $PY $BS time --backbones deit_b,deit_s,mae_b
run 0 report $PY $BS report --backbones deit_b,deit_s,mae_b
echo "$(stamp) orchestrator finished" >> "$LOG/orchestrator.log"
