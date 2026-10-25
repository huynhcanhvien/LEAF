#!/usr/bin/env bash
# DiffRate's own search on ViT-B/16-384 (official code, 3-epoch recipe), then its arms in the 384 comparison:
# waits for the densify chain to finish (both cards free), smoke-tests two steps, runs one budget per card in parallel
# (~24 h), collects the rates, seals the searched schedules + GMAC-planned LEAF, val/V2 on both cards, a new timing
# session of every arm on GPU 0, and the report. Logs: results/logs/vitb384_20261025/ (dr_*.log).
set -u
cd "$(dirname "$0")/.."
export LEAF_BACKBONES=leaf/reproduction/experiments/backbones_vitb384.json
LOG=results/logs/vitb384_20261025; mkdir -p "$LOG"
PY=python
DR=leaf/reproduction/experiments/diffrate_vitb384_search.py
VB=leaf/reproduction/experiments/vitb384_comparison.py
stamp() { date '+%H:%M:%S'; }
run() { local gpu=$1 name=$2; shift 2; echo "$(stamp) start [$name] GPU=$gpu: $* (free $(df -h / | awk 'NR==2{print $4}'))" >> "$LOG/orchestrator.log"; CUDA_VISIBLE_DEVICES=$gpu "$@" > "$LOG/$name.log" 2>&1; local rc=$?; echo "$(stamp) end   [$name] rc=$rc" >> "$LOG/orchestrator.log"; if [ $rc -ne 0 ]; then echo "$(stamp) ABORT after [$name]" >> "$LOG/orchestrator.log"; touch "$LOG/ABORTED_dr"; fi; return $rc; }
check() { if [ -e "$LOG/ABORTED_dr" ]; then echo "$(stamp) diffrate orchestrator stopped" >> "$LOG/orchestrator.log"; exit 1; fi; }

echo "$(stamp) diffrate orchestrator started (waiting for the densify chain)" >> "$LOG/orchestrator.log"
while ! grep -q "densify orchestrator finished\|densify orchestrator stopped" "$LOG/orchestrator.log"; do sleep 120; done
run 0 dr_smoke $PY $DR search --budgets 27.1 --max-steps 2; check
run 0 dr_search_27 $PY $DR search --budgets 27.1 &
run 1 dr_search_35 $PY $DR search --budgets 35.2 &
wait; check
run 0 dr_rates $PY $DR rates; check
run 0 dr_plan $PY $VB plan-dr-search; check
run 0 dr_worker_gpu0 $PY $VB worker &
run 1 dr_worker_gpu1 $PY $VB worker &
wait; check
run 0 dr_retime $PY $VB retime; check
run 0 dr_report $PY $VB report
echo "$(stamp) diffrate orchestrator finished" >> "$LOG/orchestrator.log"
