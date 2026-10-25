#!/usr/bin/env bash
# ViT-B/16-384, phase 2 (2026-10-25): DiffRate's own search first (user's priority: equal-GMAC comparison), then the
# densified baseline grid, one accuracy pass and ONE new timing session for every new arm.
#   1. wait for the batch-sensitivity chain (its timing needs both cards idle)
#   2. DiffRate search smoke test (2 steps), then one budget per card in parallel (~24 h), rates
#   3. seal the searched DiffRate schedules + GMAC-planned LEAF at their GMAC (plans_dr_search.json)
#   4. densify: cost-only pilot timing on GPU 0 (card must be idle), sealed in plans_dense.json
#   5. val / v2 of every new arm on both cards
#   6. new exclusive timing session of every arm on GPU 0, then the report
# Logs: results/logs/vitb384_20261025/ (p2_*.log). Nothing in results/ is overwritten.
set -u
cd "$(dirname "$0")/.."
export LEAF_BACKBONES=leaf/reproduction/experiments/backbones_vitb384.json
LOG=results/logs/vitb384_20261025; mkdir -p "$LOG"
PY=python
DR=leaf/reproduction/experiments/diffrate_vitb384_search.py
VB=leaf/reproduction/experiments/vitb384_comparison.py
stamp() { date '+%H:%M:%S'; }
run() { local gpu=$1 name=$2; shift 2; echo "$(stamp) start [$name] GPU=$gpu: $* (free $(df -h / | awk 'NR==2{print $4}'))" >> "$LOG/orchestrator.log"; CUDA_VISIBLE_DEVICES=$gpu "$@" > "$LOG/$name.log" 2>&1; local rc=$?; echo "$(stamp) end   [$name] rc=$rc" >> "$LOG/orchestrator.log"; if [ $rc -ne 0 ]; then echo "$(stamp) ABORT after [$name]" >> "$LOG/orchestrator.log"; touch "$LOG/ABORTED_p2"; fi; return $rc; }
check() { if [ -e "$LOG/ABORTED_p2" ]; then echo "$(stamp) phase2 orchestrator stopped" >> "$LOG/orchestrator.log"; exit 1; fi; }

echo "$(stamp) phase2 orchestrator started (waiting for batch sensitivity)" >> "$LOG/orchestrator.log"
while ! grep -q "orchestrator finished" results/logs/batch_sensitivity_20261025/orchestrator.log 2>/dev/null; do sleep 120; done
run 0 p2_dr_smoke $PY $DR search --budgets 27.1 --max-steps 2 --micro 16; check
run 0 p2_dr_search_27 $PY $DR search --budgets 27.1 --micro 16 &
run 1 p2_dr_search_35 $PY $DR search --budgets 35.2 --micro 16 &
wait; check
run 0 p2_dr_rates $PY $DR rates; check
run 0 p2_dr_plan $PY $VB plan-dr-search; check
run 0 p2_densify_pilot $PY $VB densify; check
run 0 p2_worker_gpu0 $PY $VB worker &
run 1 p2_worker_gpu1 $PY $VB worker &
wait; check
run 0 p2_retime $PY $VB retime; check
run 0 p2_report $PY $VB report
echo "$(stamp) phase2 orchestrator finished" >> "$LOG/orchestrator.log"
