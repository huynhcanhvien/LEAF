#!/usr/bin/env bash
# Latency-pricing timing chain (restartable: completed backbones are skipped via *_done.json markers).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
LOG=results/latency_pricing/logs
mkdir -p "$LOG"
CUDA_VISIBLE_DEVICES=0 $PY leaf/reproduction/experiments/latency_pricing.py gpu_time >> "$LOG/gpu_time.log" 2>&1
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= $PY leaf/reproduction/experiments/latency_pricing.py cpu_time >> "$LOG/cpu_time.log" 2>&1
CUDA_VISIBLE_DEVICES= $PY leaf/reproduction/experiments/latency_pricing.py analyze >> "$LOG/analyze.log" 2>&1
echo done > results/latency_pricing/pricing_chain_done
