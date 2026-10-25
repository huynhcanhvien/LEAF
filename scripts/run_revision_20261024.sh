#!/usr/bin/env bash
# Revision experiments of 2026-10-24 (DP verification, SALT low-budget coverage, avg-pool final station), fully unattended.
# GPU accuracy work uses both cards; every timing session runs last, on GPU 0 only, after both cards are idle.
# Logs: results/logs/revision_20261024/. Nothing in results/ is overwritten (every stage skips existing outputs).
set -u
cd "$(dirname "$0")/.."
LOG=results/logs/revision_20261024; mkdir -p "$LOG"
PY=python
DV=leaf/reproduction/experiments/dp_verify.py
SL=leaf/reproduction/experiments/salt_low_coverage.py
AP=leaf/reproduction/experiments/avgpool_final_station.py
ALL=deit_t,deit_s,deit_b,augreg_s,augreg_b,mae_b,mae_l
stamp() { date '+%H:%M:%S'; }
wait_pat() { while pgrep -f "$1" >/dev/null; do sleep 30; done; }
run() { local gpu=$1 name=$2; shift 2; echo "$(stamp) start [$name] GPU=$gpu: $*" >> "$LOG/orchestrator.log"; CUDA_VISIBLE_DEVICES=$gpu "$@" > "$LOG/$name.log" 2>&1; echo "$(stamp) end   [$name] rc=$?" >> "$LOG/orchestrator.log"; }

echo "$(stamp) orchestrator started" >> "$LOG/orchestrator.log"
wait_pat "dp_verify.py plan"                                   # plans of the other five backbones
wait_pat "dp_verify.py worker --backbones deit_b"; wait_pat "dp_verify.py worker --backbones mae_b"
# phase 1: measured F of every candidate, remaining backbones on both cards
run 0 dv_worker_gpu0 $PY $DV worker --backbones deit_s,deit_t,augreg_s &
run 1 dv_worker_gpu1 $PY $DV worker --backbones augreg_b,mae_l &
wait
run 0 dv_select $PY $DV select --backbones $ALL
# phase 2: val and V2 of the selected schedules; avg-pool re-plans
run 0 dv_val_gpu0 $PY $DV val --backbones deit_b,deit_s,deit_t,augreg_s &
run 1 dv_val_gpu1 $PY $DV val --backbones mae_b,augreg_b,mae_l &
wait
run 1 ap_worker $PY $AP worker --backbones mae_b,mae_l
# phase 3: timing sessions, GPU 0 only, both cards idle (each stage waits for idle cards itself)
run 0 salt_time $PY $SL time --backbones $ALL
run 0 dv_time $PY $DV time
run 0 ap_time $PY $AP time --backbones mae_b,mae_l
# phase 4: SALT refit, reports, re-plan and its timing
run 0 salt_fit $PY $SL fit --backbones $ALL
run 0 salt_report $PY $SL report --backbones $ALL
run 0 salt_replan $PY $SL replan --backbones $ALL
run 0 salt_time_replan $PY $SL time_replan --backbones $ALL
run 0 dv_report $PY $DV report
run 0 ap_report $PY $AP report --backbones mae_b,mae_l
echo "$(stamp) orchestrator finished" >> "$LOG/orchestrator.log"
