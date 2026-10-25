#!/usr/bin/env bash
# Out-of-domain probes (1,000 MS-COCO images) for DeiT-B, unattended: waits for the ViT-B/384 densify chain (its
# timing needs both cards idle), probes on both cards, plans on CPU, evaluates val/V2 on both cards, times on GPU 0 with
# GPU 1 idle (the stage waits itself), reports. Logs: results/logs/probe_domain_20261025/. Nothing in results/ is overwritten.
set -u
cd "$(dirname "$0")/.."
LOG=results/logs/probe_domain_20261025; mkdir -p "$LOG"
PY=python
PD=leaf/reproduction/experiments/probe_domain.py
stamp() { date '+%H:%M:%S'; }
run() { local gpu=$1 name=$2; shift 2; echo "$(stamp) start [$name] GPU=$gpu: $*" >> "$LOG/orchestrator.log"; CUDA_VISIBLE_DEVICES=$gpu "$@" > "$LOG/$name.log" 2>&1; local rc=$?; echo "$(stamp) end   [$name] rc=$rc" >> "$LOG/orchestrator.log"; if [ $rc -ne 0 ]; then echo "$(stamp) ABORT after [$name]" >> "$LOG/orchestrator.log"; touch "$LOG/ABORTED"; fi; return $rc; }
check() { if [ -e "$LOG/ABORTED" ]; then echo "$(stamp) orchestrator stopped" >> "$LOG/orchestrator.log"; exit 1; fi; }

echo "$(stamp) orchestrator started (waiting for the densify chain)" >> "$LOG/orchestrator.log"
while ! grep -q "densify orchestrator finished\|densify orchestrator stopped" results/logs/vitb384_20261025/orchestrator.log 2>/dev/null; do sleep 120; done
run 0,1 search $PY $PD search --gpus 0,1; check
run 0 plan $PY $PD plan; check
run 0 val $PY $PD val; check
run 0 time $PY $PD time; check
run 0 report $PY $PD report
echo "$(stamp) orchestrator finished" >> "$LOG/orchestrator.log"
