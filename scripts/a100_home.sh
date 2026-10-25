#!/usr/bin/env bash
# Home side of the A100 sessions (scripts/colab/LEAF_A100.ipynb): unpack the archives the notebook saved and run the
# CPU comparisons. Usage:
#   bash scripts/a100_home.sh transfer second_device_transfer.tar.gz     # pricing arms at device latency -> results/second_device/compare_transfer.json
#   bash scripts/a100_home.sh main a100_timing_<device>.tar.gz          # main comparison at device latency -> results/a100_timing/report_<device>.json
#   bash scripts/a100_home.sh diffrate <diffrate_vitb384_search dir>    # copy the searched rates in, seal + evaluate at home (needs the 384 caches)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
phase=${1:?phase: transfer | main | diffrate}; archive=${2:?archive or directory}
case "$phase" in
  transfer)
    cp -n "$archive" "results/archives/$(basename "$archive")" 2>/dev/null || true
    bash scripts/run_second_device.sh home-transfer "$archive" ;;
  main)
    cp -n "$archive" "results/archives/$(basename "$archive")" 2>/dev/null || true
    $PY leaf/reproduction/experiments/a100_timing.py unpack --archive "$archive"
    CUDA_VISIBLE_DEVICES= $PY leaf/reproduction/experiments/a100_timing.py compare ;;
  diffrate)
    mkdir -p results/diffrate_vitb384_search
    cp -rn "$archive"/. results/diffrate_vitb384_search/
    export LEAF_BACKBONES=leaf/reproduction/experiments/backbones_vitb384.json
    $PY leaf/reproduction/experiments/diffrate_vitb384_search.py rates
    $PY leaf/reproduction/experiments/vitb384_comparison.py plan-dr-search
    echo "now: worker on both cards, retime on GPU 0 (both cards idle), report:"
    echo "  CUDA_VISIBLE_DEVICES=0 $PY leaf/reproduction/experiments/vitb384_comparison.py worker & CUDA_VISIBLE_DEVICES=1 $PY leaf/reproduction/experiments/vitb384_comparison.py worker; wait"
    echo "  CUDA_VISIBLE_DEVICES=0 $PY leaf/reproduction/experiments/vitb384_comparison.py retime && $PY leaf/reproduction/experiments/vitb384_comparison.py report" ;;
  *) echo "unknown phase $phase" >&2; exit 1 ;;
esac
