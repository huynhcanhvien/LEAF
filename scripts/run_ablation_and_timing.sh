#!/usr/bin/env bash
# 2026-10-20 (user): continuation of scripts/run_alpha_grid_replan.sh with the component ablations inserted before timing,
# so that the grid groups, the core arms and leaf_alpha4 are timed in one session. Steps: let the running MAE-L probe workers finish,
# complete wave 2 (probes, seal, accuracy); core phase on DeiT-B/MAE-B (restore the sealed core plans, seal forecast_only / alpha_grid / theta_grid,
# surrogate_check on calibration + val, accuracy); bs32 timing last on GPU 0 alone; report.
# v2 (2026-10-20, user: fill every number the paper still lacked): also seals final_station_all on all seven backbones (final-block
# station removed from every leaf_alpha4 and GMAC-planned schedule that has one; identical predictions + timed in the same session).
# Start with: setsid nohup bash scripts/run_ablation_and_timing.sh &
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
AB=results/ablation_and_alpha_grid
LOGS=$AB/logs; mkdir -p $LOGS
LOG=results/notes/state/alpha-ablation-2026-10-19-progress.md
CORE_BBS="deit_b mae_b"
echo $$ > $AB/chain.pid
note() { echo "- $(date '+%F %T') $*" >> $LOG; }
setphase() { echo "$1" > $AB/phase; note "phase: $1"; }

wait_free() { while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)
  [ "$used" -lt 1500 ] && return; sleep 120; done; }

run_pair() { local name=$1; shift; local pids=()
  for g in 0 1; do wait_free $g
    CUDA_VISIBLE_DEVICES=$g $PY "$@" >> $LOGS/$name.gpu$g.log 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait $p || note "$name: a worker exited with an error (see $LOGS/$name.gpu*.log)"; done; }

probe_missing() { $PY -c "
import sys; sys.path.insert(0,'.'); import leaf.reproduction.experiments.ablation_and_alpha_grid as a; from leaf.evaluation.runner import config_hash
print(sum(not (a.ROOT/bb/'probe_done'/(config_hash(j['config'])+'.json')).exists() for bb in '$BBS'.split() for s in a.probe_sets(bb) for j in a.probe_jobs(bb,s)))"; }
acc_missing() { $PY -c "
import sys; sys.path.insert(0,'.'); import leaf.reproduction.experiments.ablation_and_alpha_grid as a
print(sum(not (a.ROOT/'completed'/s/(t['id']+'.json')).exists() for t in a.tasks() for s in a.SPLITS))"; }
h_missing() { $PY -c "
import sys; sys.path.insert(0,'.'); import leaf.reproduction.experiments.ablation_and_alpha_grid as a
print(sum(not (a.ROOT/bb/d/(r['id']+'.json')).exists() for bb in a.P1_BBS for r in a.h_schedules(bb)['schedules'] for d in ('surrogate_check_done','surrogate_check_val')))"; }

retry() { local name=$1 miss=$2 rel=$3; shift 3
  for i in 1 2 3; do
    m=$($miss); [ "$m" = 0 ] && { note "$name: complete"; return 0; }
    [ $i -gt 1 ] && { $PY $rel >> $LOGS/$name.release.log 2>&1; note "$name: retry $i, $m missing"; }
    run_pair $name "$@"
  done
  m=$($miss); [ "$m" = 0 ] && { note "$name: complete"; return 0; }
  note "$name: STILL $m missing after 3 attempts"; return 1; }

note "full chain v2 started (pid $$): core phase inserted before timing; waiting for the running MAE-L probe workers"
# ---- wave 2 (MAE-L): finish probes, seal, accuracy
setphase probe-wave2
while pgrep -f "leaf/reproduction/experiments/ablation_and_alpha_grid.py probe" > /dev/null; do sleep 60; done
BBS=mae_l
retry probe2 probe_missing "leaf/reproduction/experiments/ablation_and_alpha_grid.py release" leaf/reproduction/experiments/ablation_and_alpha_grid.py probe --bbs mae_l
CUDA_VISIBLE_DEVICES=0 $PY leaf/reproduction/experiments/ablation_and_alpha_grid.py probe --bbs mae_l >> $LOGS/probe.collect.log 2>&1
setphase plan-wave2
$PY leaf/reproduction/experiments/ablation_and_alpha_grid.py plan --bbs mae_l >> $LOGS/plan_wave2.log 2>&1
note "plan wave 2: $(grep -c sealed $LOGS/plan_wave2.log) groups sealed, $(grep -c waiting $LOGS/plan_wave2.log) waiting"
setphase accuracy-wave2
retry accuracy2 acc_missing "leaf/reproduction/experiments/ablation_and_alpha_grid.py release" leaf/reproduction/experiments/ablation_and_alpha_grid.py worker
$PY leaf/reproduction/experiments/ablation_and_alpha_grid.py report >> $LOGS/report.log 2>&1 && note "interim report (accuracy only) after wave 2"

# ---- core phase (DeiT-B, MAE-B): component ablations and sensitivity
setphase plan-core
for bb in $CORE_BBS; do
  for f in $AB/$bb/plans_core_2026-10-20/*.json; do [ -e $AB/$bb/plans/$(basename $f) ] || cp $f $AB/$bb/plans/; done
done
ABL_PHASE=core $PY leaf/reproduction/experiments/ablation_and_alpha_grid.py plan --bbs $CORE_BBS >> $LOGS/plan_core.log 2>&1
ABL_PHASE=core $PY leaf/reproduction/experiments/ablation_and_alpha_grid.py plan --bbs deit_t deit_s deit_b augreg_s augreg_b mae_b mae_l --sets final_station_all >> $LOGS/plan_core.log 2>&1
note "plan core: $(grep -c sealed $LOGS/plan_core.log) groups sealed, $(grep -c waiting $LOGS/plan_core.log) waiting; restored $(ls $AB/deit_b/plans | wc -l) + $(ls $AB/mae_b/plans | wc -l) plan files"
setphase surrogate-core
retry surrogate h_missing "leaf/reproduction/experiments/ablation_and_alpha_grid.py release" leaf/reproduction/experiments/ablation_and_alpha_grid.py surrogate
setphase accuracy-core
retry accuracy-core acc_missing "leaf/reproduction/experiments/ablation_and_alpha_grid.py release" leaf/reproduction/experiments/ablation_and_alpha_grid.py worker
$PY leaf/reproduction/experiments/ablation_and_alpha_grid.py report >> $LOGS/report.log 2>&1 && note "interim report (accuracy only) after core"

# ---- latency, last, GPU 0 only (GPU 1 left idle): every sealed config of all seven backbones in one session
setphase timing
wait_free 0; wait_free 1
CUDA_VISIBLE_DEVICES=0 $PY leaf/reproduction/experiments/ablation_and_alpha_grid.py time >> $LOGS/time.log 2>&1 || { note "timing failed (see $LOGS/time.log)"; setphase failed; exit 1; }

setphase report
$PY leaf/reproduction/experiments/ablation_and_alpha_grid.py report >> $LOGS/report.log 2>&1 && note "report written: results/reports/ablation-v3-2026-10-19.md"
setphase completed
