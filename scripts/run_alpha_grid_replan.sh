#!/usr/bin/env bash
# 2026-10-20 (user): grid phase of the ablation plan. Finish the alpha equivalence check,
# then re-plan the main schedules of all seven backbones on wider grids (leaf: alpha 7 values, grid_theta: theta 5 values, grid_theta_alpha: both)
# and evaluate val/V2: wave 1 = six backbones, wave 2 = MAE-L (slowest probes). Accuracy on both GPUs (one process per
# card); bs32 timing last on GPU 0 alone; then the reports.
# Start with: setsid nohup bash scripts/run_alpha_grid_replan.sh &
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
EQ=results/alpha_kappa_equivalence
AB=results/ablation_and_alpha_grid
LOGS=$AB/logs; mkdir -p $LOGS $EQ/logs
LOG=results/notes/state/alpha-ablation-2026-10-19-progress.md
WAVE1="deit_t deit_s deit_b augreg_s augreg_b mae_b"
WAVE2="mae_l"
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

eq_missing() { $PY -c "
import json,pathlib; r=pathlib.Path('$EQ'); t=json.load(open(r/'tasks.json'))['tasks']
print(sum(not (r/'completed'/'alpha'/(x['id']+'.json')).exists() for x in t))"; }
eq_ctl_missing() { $PY -c "
import sys; sys.path.insert(0,'.'); import leaf.reproduction.experiments.alpha_kappa_equivalence as e
print(sum(not (e.ROOT/'completed'/'control'/(i+'.json')).exists() for i in e.changed_ids()))"; }
probe_missing() { $PY -c "
import sys; sys.path.insert(0,'.'); import leaf.reproduction.experiments.ablation_and_alpha_grid as a; from leaf.evaluation.runner import config_hash
print(sum(not (a.ROOT/bb/'probe_done'/(config_hash(j['config'])+'.json')).exists() for bb in '$BBS'.split() for s in a.probe_sets(bb) for j in a.probe_jobs(bb,s)))"; }
acc_missing() { $PY -c "
import sys; sys.path.insert(0,'.'); import leaf.reproduction.experiments.ablation_and_alpha_grid as a
print(sum(not (a.ROOT/'completed'/s/(t['id']+'.json')).exists() for t in a.tasks() for s in a.SPLITS))"; }

retry() { local name=$1 miss=$2 rel=$3; shift 3
  for i in 1 2 3; do
    m=$($miss); [ "$m" = 0 ] && { note "$name: complete"; return 0; }
    [ $i -gt 1 ] && { $PY $rel >> $LOGS/$name.release.log 2>&1; note "$name: retry $i, $m missing"; }
    run_pair $name "$@"
  done
  m=$($miss); [ "$m" = 0 ] && { note "$name: complete"; return 0; }
  note "$name: STILL $m missing after 3 attempts"; return 1; }

note "grid chain started (pid $$); waiting for running equivalence workers"
# ---- 1. alpha equivalence: let the workers of the previous chain finish, then complete and report
setphase equivalence
while pgrep -f "leaf/reproduction/experiments/alpha_kappa_equivalence.py worker" > /dev/null; do sleep 60; done
$PY leaf/reproduction/experiments/alpha_kappa_equivalence.py release >> $EQ/logs/report.log 2>&1
retry equivalence eq_missing "leaf/reproduction/experiments/alpha_kappa_equivalence.py release" leaf/reproduction/experiments/alpha_kappa_equivalence.py worker
$PY leaf/reproduction/experiments/alpha_kappa_equivalence.py report >> $EQ/logs/report.log 2>&1
note "equivalence report: $(python3 -c "import json;s=json.load(open('$EQ/summary.json'));print(f\"identical {s['identical']}/{s['n_compared']}, max changed {s['max_changed_pred']}, max |dtop1| {s['max_abs_delta_top1']}\")")"
if [ "$(eq_ctl_missing)" != 0 ]; then
  setphase equivalence-control
  retry equivalence-control eq_ctl_missing "leaf/reproduction/experiments/alpha_kappa_equivalence.py release" leaf/reproduction/experiments/alpha_kappa_equivalence.py worker --control-changed
  $PY leaf/reproduction/experiments/alpha_kappa_equivalence.py report >> $EQ/logs/report.log 2>&1 && note "equivalence report rewritten with the control arm"
fi

# ---- 2. waves: probes -> seal -> accuracy
for W in 1 2; do
  [ $W = 1 ] && BBS=$WAVE1 || BBS=$WAVE2
  setphase "probe-wave$W"
  retry probe$W probe_missing "leaf/reproduction/experiments/ablation_and_alpha_grid.py release" leaf/reproduction/experiments/ablation_and_alpha_grid.py probe --bbs $BBS
  CUDA_VISIBLE_DEVICES=0 $PY leaf/reproduction/experiments/ablation_and_alpha_grid.py probe --bbs $BBS >> $LOGS/probe.collect.log 2>&1   # probes/<set>.json
  setphase "plan-wave$W"
  $PY leaf/reproduction/experiments/ablation_and_alpha_grid.py plan --bbs $BBS >> $LOGS/plan_wave$W.log 2>&1
  note "plan wave $W: $(grep -c sealed $LOGS/plan_wave$W.log) groups sealed, $(grep -c waiting $LOGS/plan_wave$W.log) waiting"
  setphase "accuracy-wave$W"
  retry accuracy$W acc_missing "leaf/reproduction/experiments/ablation_and_alpha_grid.py release" leaf/reproduction/experiments/ablation_and_alpha_grid.py worker
  $PY leaf/reproduction/experiments/ablation_and_alpha_grid.py report >> $LOGS/report.log 2>&1 && note "interim report (accuracy only) after wave $W"
done

# ---- 3. latency, last, GPU 0 only (GPU 1 left idle)
setphase timing
wait_free 0; wait_free 1
CUDA_VISIBLE_DEVICES=0 $PY leaf/reproduction/experiments/ablation_and_alpha_grid.py time >> $LOGS/time.log 2>&1 || { note "timing failed (see $LOGS/time.log)"; setphase failed; exit 1; }

# ---- 4. report
setphase report
$PY leaf/reproduction/experiments/ablation_and_alpha_grid.py report >> $LOGS/report.log 2>&1 && note "report written: results/reports/ablation-v3-2026-10-19.md"
setphase completed
