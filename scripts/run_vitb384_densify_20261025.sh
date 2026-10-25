#!/usr/bin/env bash
# ViT-B/16-384: extra baseline points near every LEAF latency (clip_comparison.densify protocol: cost-only pilot timing,
# sealed before any accuracy), then val/V2 of the new points on both cards, a new timing session of every arm on GPU 0,
# and the report. Waits for the batch-sensitivity orchestrator to finish first (its timing needs both cards idle).
# Logs: results/logs/vitb384_20261025/ (densify_*.log). Nothing in results/ is overwritten.
set -u
cd "$(dirname "$0")/.."
export LEAF_BACKBONES=leaf/reproduction/experiments/backbones_vitb384.json
LOG=results/logs/vitb384_20261025; mkdir -p "$LOG"
PY=python
VB=leaf/reproduction/experiments/vitb384_comparison.py
stamp() { date '+%H:%M:%S'; }
run() { local gpu=$1 name=$2; shift 2; echo "$(stamp) start [$name] GPU=$gpu: $* (free $(df -h / | awk 'NR==2{print $4}'))" >> "$LOG/orchestrator.log"; CUDA_VISIBLE_DEVICES=$gpu "$@" > "$LOG/$name.log" 2>&1; local rc=$?; echo "$(stamp) end   [$name] rc=$rc" >> "$LOG/orchestrator.log"; if [ $rc -ne 0 ]; then echo "$(stamp) ABORT after [$name]" >> "$LOG/orchestrator.log"; touch "$LOG/ABORTED_densify"; fi; return $rc; }
check() { if [ -e "$LOG/ABORTED_densify" ]; then echo "$(stamp) densify orchestrator stopped" >> "$LOG/orchestrator.log"; exit 1; fi; }

echo "$(stamp) densify orchestrator started (waiting for batch sensitivity)" >> "$LOG/orchestrator.log"
while ! grep -q "orchestrator finished" results/logs/batch_sensitivity_20261025/orchestrator.log 2>/dev/null; do sleep 120; done
run 0 densify_pilot $PY $VB densify; check
run 0 densify_worker_gpu0 $PY $VB worker &
run 1 densify_worker_gpu1 $PY $VB worker &
wait; check
run 0 densify_retime $PY $VB retime; check
run 0 densify_report $PY $VB report
echo "$(stamp) densify orchestrator finished" >> "$LOG/orchestrator.log"
