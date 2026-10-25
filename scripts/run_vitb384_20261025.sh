#!/usr/bin/env bash
# ViT-B/16-384 comparison (augreg_b384, 577 tokens), unattended, 2026-10-25.
# Probes and accuracy use both cards; every timing step (budgets, SALT workloads, final session) runs on GPU 0 with GPU 1 idle,
# because `leaf search` runs its timing steps after the probe shards have finished and the final timing stage waits for an idle card.
# Afterwards, if scripts/run_batch_sensitivity_20261025.sh exists, it is run (batch-size sensitivity, GPU 0 only).
# Logs: results/logs/vitb384_20261025/. Nothing in results/ is overwritten (every stage skips existing outputs).
set -u
cd "$(dirname "$0")/.."
export LEAF_BACKBONES=leaf/reproduction/experiments/backbones_vitb384.json
LOG=results/logs/vitb384_20261025; mkdir -p "$LOG"
PY=python
VB=leaf/reproduction/experiments/vitb384_comparison.py
stamp() { date '+%H:%M:%S'; }
run() { local gpu=$1 name=$2; shift 2; echo "$(stamp) start [$name] GPU=$gpu: $* (free $(df -h / | awk 'NR==2{print $4}'))" >> "$LOG/orchestrator.log"; CUDA_VISIBLE_DEVICES=$gpu "$@" > "$LOG/$name.log" 2>&1; local rc=$?; echo "$(stamp) end   [$name] rc=$rc" >> "$LOG/orchestrator.log"; if [ $rc -ne 0 ]; then echo "$(stamp) ABORT after [$name]" >> "$LOG/orchestrator.log"; touch "$LOG/ABORTED"; fi; return $rc; }
check() { if [ -e "$LOG/ABORTED" ]; then echo "$(stamp) orchestrator stopped" >> "$LOG/orchestrator.log"; exit 1; fi; }

echo "$(stamp) orchestrator started" >> "$LOG/orchestrator.log"
# phase 1: the search (probes sharded over both cards; budgets / SALT timing on GPU 0 afterwards)
run 0,1 search $PY $VB search --gpus 0,1; check
# phase 2: seal every arm, then val / v2 on both cards
run 0 plan $PY $VB plan; check
run 0 worker_gpu0 $PY $VB worker &
run 1 worker_gpu1 $PY $VB worker &
wait; check
# phase 3: one timing session on GPU 0 (waits for an idle card), then the report
run 0 time $PY $VB time; check
run 0 report $PY $VB report
echo "$(stamp) vitb384 finished" >> "$LOG/orchestrator.log"
# phase 4: batch-size sensitivity, if its script exists by now
if [ -x scripts/run_batch_sensitivity_20261025.sh ]; then
  echo "$(stamp) start batch sensitivity" >> "$LOG/orchestrator.log"
  scripts/run_batch_sensitivity_20261025.sh
  echo "$(stamp) end batch sensitivity rc=$?" >> "$LOG/orchestrator.log"
fi
echo "$(stamp) orchestrator finished" >> "$LOG/orchestrator.log"
