#!/usr/bin/env bash
# Calibration-set size ablation (1,000 / 2,500 / 5,000 / 10,000 probe images) for DeiT-B, DeiT-S and ViT-B (MAE), unattended.
# Odd-half probes on both cards (DeiT-S shared through claims), val/V2 on both cards, timing last on GPU 0 only.
# Logs: results/logs/calib_size_20261024/. Nothing in results/ is overwritten (every stage skips existing outputs).
set -u
cd "$(dirname "$0")/.."
LOG=results/logs/calib_size_20261024; mkdir -p "$LOG"
PY=python
CS=leaf/reproduction/experiments/calib_size.py
BBS=deit_b,deit_s,mae_b
stamp() { date '+%H:%M:%S'; }
run() { local gpu=$1 name=$2; shift 2; echo "$(stamp) start [$name] GPU=$gpu: $*" >> "$LOG/orchestrator.log"; CUDA_VISIBLE_DEVICES=$gpu "$@" > "$LOG/$name.log" 2>&1; local rc=$?; echo "$(stamp) end   [$name] rc=$rc" >> "$LOG/orchestrator.log"; if [ $rc -ne 0 ]; then echo "$(stamp) ABORT after [$name]" >> "$LOG/orchestrator.log"; touch "$LOG/ABORTED"; fi; return $rc; }
check() { if [ -e "$LOG/ABORTED" ]; then echo "$(stamp) orchestrator stopped" >> "$LOG/orchestrator.log"; exit 1; fi; }

echo "$(stamp) orchestrator started" >> "$LOG/orchestrator.log"
# phase 1: odd-half probes (DeiT-B on GPU 0, MAE-B on GPU 1, then both cards share DeiT-S)
run 0 probe_gpu0 $PY $CS probe --backbones deit_b,deit_s &
run 1 probe_gpu1 $PY $CS probe --backbones mae_b,deit_s &
wait; check
run 0 collect $PY $CS collect --backbones $BBS
# phase 2: re-plan (CPU) and val/V2 of the re-planned schedules
run 0 plan $PY $CS plan --backbones $BBS; check
run 0 val_gpu0 $PY $CS val --backbones deit_b &
run 1 val_gpu1 $PY $CS val --backbones mae_b,deit_s &
wait; check
# phase 3: one timing session, GPU 0 only, both cards idle (the stage waits for idle cards itself)
run 0 time $PY $CS time --backbones $BBS; check
run 0 report $PY $CS report --backbones $BBS
echo "$(stamp) orchestrator finished" >> "$LOG/orchestrator.log"
