#!/usr/bin/env bash
# Second accelerator (leaf/reproduction/experiments/second_device.py): the latency-pricing test on another GPU.
# Four phases, alternating between the device (e.g. a Kaggle T4) and the home machine (2 GPUs, ImageNet caches):
#   device-1  [device]  unpack the kit, smoke test, time the workload set        -> second_device_1.tar.gz (download)
#   home-1    [home]    unpack, fit SALT, plan and seal -> second_device_2.tar.gz (upload); then accuracy on both GPUs
#   device-2  [device]  unpack, one timing session of every planned arm          -> second_device_3.tar.gz (download)
#   home-2    [home]    unpack, SALT vs GMAC planning at equal measured latency  -> results/second_device/compare.json
# Transfer mode, one device session (about an hour on a T4), no re-planning:
#   device-transfer [device]  unpack the kit, smoke test, time the sealed pricing arms -> second_device_transfer.tar.gz
#   home-transfer   [home]    unpack, SALT-planned vs GMAC-planned at equal device latency -> compare_transfer.json
# Usage: bash scripts/run_second_device.sh <phase> [archive]
# The device runs in FP16 unless LEAF_DTYPE is set (T4 and other sm_75 GPUs have no BF16).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
EXP=leaf/reproduction/experiments/second_device.py
phase=${1:?phase: device-1 | home-1 | device-2 | home-2}; archive=${2:-}
case "$phase" in
  device-1)
    export LEAF_DTYPE=${LEAF_DTYPE:-float16}
    $PY $EXP unpack --archive "${archive:-scripts/kaggle/second_device_kit.tar.gz}"
    CUDA_VISIBLE_DEVICES=0 $PY $EXP check
    CUDA_VISIBLE_DEVICES=0 $PY $EXP time_workloads
    $PY $EXP pack --name second_device_1 ;;
  home-1)
    $PY $EXP unpack --archive "${archive:?archive from device-1}"
    CUDA_VISIBLE_DEVICES= $PY $EXP plan
    $PY $EXP pack --name second_device_2          # upload now: session 2 does not need the accuracy below
    CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 $PY $EXP worker & a=$!
    CUDA_VISIBLE_DEVICES=1 taskset -c 0-7 $PY $EXP worker & b=$!
    wait $a $b ;;
  device-2)
    export LEAF_DTYPE=${LEAF_DTYPE:-float16}
    $PY $EXP unpack --archive "${archive:?archive from home-1}"
    CUDA_VISIBLE_DEVICES=0 $PY $EXP time_final
    $PY $EXP pack --name second_device_3 ;;
  home-2)
    $PY $EXP unpack --archive "${archive:?archive from device-2}"
    CUDA_VISIBLE_DEVICES= $PY $EXP compare ;;
  device-transfer)
    export LEAF_DTYPE=${LEAF_DTYPE:-float16}
    $PY $EXP unpack --archive "${archive:-scripts/kaggle/second_device_kit.tar.gz}"
    CUDA_VISIBLE_DEVICES=0 $PY $EXP check
    CUDA_VISIBLE_DEVICES=0 $PY $EXP time_transfer
    $PY $EXP pack --name second_device_transfer ;;
  home-transfer)
    $PY $EXP unpack --archive "${archive:?archive from device-transfer}"
    CUDA_VISIBLE_DEVICES= $PY $EXP compare_transfer ;;
  *) echo "unknown phase $phase" >&2; exit 1 ;;
esac
