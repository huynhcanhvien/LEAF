#!/usr/bin/env bash
# ViT-B/16-384, phase 3 (2026-10-25): four DiffRate budgets (23.8, 27.1, 30.8, 35.2 GFLOPs = the GMAC of the LEAF
# schedules at 0.48-0.72x the dense latency). The searches for 27.1 (GPU 0) and 35.2 (GPU 1) were started by phase 2
# at 14:14 and keep running; this script waits for them, runs 23.8 (GPU 0) and 30.8 (GPU 1), then seals all four with
# GMAC-planned LEAF, densifies the baseline grid (pilot timing, GPU 0 idle), evaluates every new arm on both cards,
# re-times every arm in one session on GPU 0 and reports. Logs: results/logs/vitb384_20261025/ (p3_*.log).
set -u
cd "$(dirname "$0")/.."
export LEAF_BACKBONES=leaf/reproduction/experiments/backbones_vitb384.json
LOG=results/logs/vitb384_20261025; mkdir -p "$LOG"
PY=python
DR=leaf/reproduction/experiments/diffrate_vitb384_search.py
VB=leaf/reproduction/experiments/vitb384_comparison.py
R=results/diffrate_vitb384_search
stamp() { date '+%H:%M:%S'; }
run() { local gpu=$1 name=$2; shift 2; echo "$(stamp) start [$name] GPU=$gpu: $* (free $(df -h / | awk 'NR==2{print $4}'))" >> "$LOG/orchestrator.log"; CUDA_VISIBLE_DEVICES=$gpu "$@" > "$LOG/$name.log" 2>&1; local rc=$?; echo "$(stamp) end   [$name] rc=$rc" >> "$LOG/orchestrator.log"; if [ $rc -ne 0 ]; then echo "$(stamp) ABORT after [$name]" >> "$LOG/orchestrator.log"; touch "$LOG/ABORTED_p3"; fi; return $rc; }
check() { if [ -e "$LOG/ABORTED_p3" ]; then echo "$(stamp) phase3 orchestrator stopped" >> "$LOG/orchestrator.log"; exit 1; fi; }

echo "$(stamp) phase3 orchestrator started (waiting for the 27.1 and 35.2 searches)" >> "$LOG/orchestrator.log"
while pgrep -f "diffrate_vitb384_search.py search --budgets (27.1|35.2)" >/dev/null || [ ! -e "$R/27.1/rates.json" ] || [ ! -e "$R/35.2/rates.json" ]; do
  if ! pgrep -f "diffrate_vitb384_search.py search --budgets (27.1|35.2)" >/dev/null && { [ ! -e "$R/27.1/rates.json" ] || [ ! -e "$R/35.2/rates.json" ]; }; then
    echo "$(stamp) a first-pair search ended without rates.json; resuming it from its checkpoint" >> "$LOG/orchestrator.log"
    [ -e "$R/27.1/rates.json" ] || run 0 p3_dr_resume_27 $PY $DR search --budgets 27.1 --micro 16 &
    [ -e "$R/35.2/rates.json" ] || run 1 p3_dr_resume_35 $PY $DR search --budgets 35.2 --micro 16 &
    wait; check
  fi
  sleep 300
done
echo "$(stamp) first pair done" >> "$LOG/orchestrator.log"
run 0 p3_dr_search_23 $PY $DR search --budgets 23.8 --micro 16 &
run 1 p3_dr_search_30 $PY $DR search --budgets 30.8 --micro 16 &
wait; check
run 0 p3_dr_rates $PY $DR rates; check
run 0 p3_dr_plan $PY $VB plan-dr-search; check
run 0 p3_densify_pilot $PY $VB densify; check
run 0 p3_worker_gpu0 $PY $VB worker &
run 1 p3_worker_gpu1 $PY $VB worker &
wait; check
run 0 p3_retime $PY $VB retime; check
run 0 p3_report $PY $VB report
echo "$(stamp) phase3 orchestrator finished" >> "$LOG/orchestrator.log"
